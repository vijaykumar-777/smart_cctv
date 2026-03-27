import cv2
import time
import numpy as np
import threading
from collections import defaultdict
from reid_model import ReIDFeatureExtractor


# ── Configuration ──
MIN_FEATURES_FOR_MATCH = 5     # Require 5 stable features before matching
CROSS_CAM_THRESHOLD    = 0.58
SAME_CAM_THRESHOLD     = 0.76
COLOR_ASSIST_LOW       = 0.40  # Bottom of borderline band (use color to confirm)
COLOR_ASSIST_HIGH      = 0.58  # Top of borderline band (== CROSS_CAM_THRESHOLD)
COLOR_CONFIRM_SCORE    = 0.72  # Min histogram correlation to accept a borderline match
TIME_GATE_SECONDS      = 60
MIN_CROP_HEIGHT        = 100   # Drop small crops — they lack enough ReID resolution
EMA_ALPHA              = 0.2
GALLERY_EXPIRY_SECONDS = 300


# ── Color histogram helper ──────────────────────────────────────────────────
def _color_hist(crop):
    """
    Compute a normalized 2-D HSV histogram (Hue × Saturation) from a BGR crop.
    Ignores Value channel so it is robust to lighting changes.
    Returns None if the crop is invalid.
    """
    if crop is None or crop.size == 0:
        return None
    try:
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [36, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist
    except Exception:
        return None


def _hist_similarity(h1, h2):
    """Bhattacharyya distance converted to a 0-1 similarity (1 = identical)."""
    if h1 is None or h2 is None:
        return 0.0
    dist = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
    return float(1.0 - dist)  # invert so higher == more similar


# ── Per-person identity record ──────────────────────────────────────────────
class PersonIdentity:
    def __init__(self, global_id, initial_feature, initial_hist, cam_id):
        self.global_id      = global_id
        self.gallery_feature = initial_feature

        # Multi-view feature bank — stores structurally distinct poses
        self.feature_bank   = [initial_feature] if initial_feature is not None else []
        self.max_bank_size  = 5

        # Color histogram gallery (averaged over observations)
        self.color_hist     = initial_hist

        self.cameras_seen   = {cam_id}
        self.last_seen_time = time.time()
        self.last_seen_cam  = cam_id
        self.frame_count    = 1 if initial_feature is not None else 0

    # ── Feature update ──────────────────────────────────────────────────────
    def add_feature(self, feature, hist=None):
        if feature is None:
            return

        if self.gallery_feature is None:
            self.gallery_feature = feature
            self.feature_bank.append(feature)
        else:
            # EMA blend keeps the gallery profile up-to-date with appearance drift
            updated = EMA_ALPHA * feature + (1.0 - EMA_ALPHA) * self.gallery_feature
            norm    = np.linalg.norm(updated)
            if norm > 0:
                self.gallery_feature = updated / norm

            # Only add to the bank if this is a novel viewpoint (low similarity to all stored)
            if self.feature_bank:
                max_sim = max(float(np.dot(feature, f)) for f in self.feature_bank)
                if max_sim < 0.85:  # novel angle / pose
                    if len(self.feature_bank) >= self.max_bank_size:
                        self.feature_bank.pop(0)  # evict oldest distinct view
                    self.feature_bank.append(feature)
            else:
                self.feature_bank.append(feature)

        # Update color histogram with a simple running average
        if hist is not None:
            if self.color_hist is None:
                self.color_hist = hist.copy()
            else:
                self.color_hist = 0.8 * self.color_hist + 0.2 * hist
                cv2.normalize(self.color_hist, self.color_hist)

        self.frame_count += 1

    def is_mature(self):
        """True once we have accumulated enough frames to trust this identity."""
        return self.frame_count >= MIN_FEATURES_FOR_MATCH

    def touch(self, cam_id):
        self.last_seen_time = time.time()
        self.last_seen_cam  = cam_id
        self.cameras_seen.add(cam_id)


# ── Main tracker ────────────────────────────────────────────────────────────
class GlobalTracker:
    def __init__(self):
        self._lock          = threading.Lock()
        self._next_global_id = 1

        # O(1) set of active global IDs per camera
        self._active_global_ids_per_cam = defaultdict(set)

        # (cam_id, local_track_id) -> global_id
        self._local_to_global = {}

        # global_id -> PersonIdentity
        self._identities = {}

        self._reid = ReIDFeatureExtractor(model_name='osnet_ain_x1_0')

    # ── Cosine similarity ────────────────────────────────────────────────────
    @staticmethod
    def _cosine_sim(feat_a, feat_b):
        if feat_a is None or feat_b is None:
            return -1.0
        return float(np.dot(feat_a, feat_b))

    # ── Public API ───────────────────────────────────────────────────────────
    def resolve(self, cam_id, local_track_id, crop, cls_id=0):
        """
        Map (cam_id, local_track_id) to a persistent global ID.
        Heavy model inference runs BEFORE acquiring the lock so that
        camera threads do not block each other.
        """
        key = (cam_id, local_track_id)

        # ── 1. Extract outside the lock (expensive GPU work) ─────────────────
        feat = None
        hist = None
        if cls_id == 0:
            if crop is not None and crop.shape[0] >= MIN_CROP_HEIGHT:
                feat = self._reid.extract(crop)
                if feat is not None:
                    hist = _color_hist(crop)

        # ── 2. Everything else is fast dict work — hold the lock briefly ──────
        with self._lock:

            # Already have a mapping for this local track
            if key in self._local_to_global:
                gid      = self._local_to_global[key]
                identity = self._identities.get(gid)
                if identity:
                    identity.touch(cam_id)
                    if feat is not None:
                        identity.add_feature(feat, hist)
                return gid

            # Non-person objects or no usable feature → new ID, no matching
            if cls_id != 0 or feat is None:
                return self._assign_new(key, cam_id, feature=None, hist=None)

            # Try to find a match in the gallery
            match_gid = self._find_match(feat, hist, cam_id)

            if match_gid is not None:
                self._local_to_global[key] = match_gid
                self._active_global_ids_per_cam[cam_id].add(match_gid)
                identity = self._identities[match_gid]
                identity.touch(cam_id)
                identity.add_feature(feat, hist)
                return match_gid
            else:
                return self._assign_new(key, cam_id, feature=feat, hist=hist)

    # ── Internal helpers ─────────────────────────────────────────────────────
    def _assign_new(self, key, cam_id, feature, hist):
        gid = self._next_global_id
        self._next_global_id += 1
        self._identities[gid] = PersonIdentity(gid, feature, hist, cam_id)
        self._local_to_global[key] = gid
        self._active_global_ids_per_cam[cam_id].add(gid)
        return gid

    def _find_match(self, query_feat, query_hist, query_cam_id):
        """
        Search the gallery for the best matching PersonIdentity.

        Matching strategy (in order):
          1. Multi-view feature bank max-pooling  → confident match
          2. Spatio-temporal discount             → penalise old sightings
          3. Color histogram confirmation         → rescue borderline ReID scores
        """
        best_gid         = None
        best_score       = -1.0
        borderline_gid   = None      # best candidate in the color-assist band
        borderline_score = -1.0
        now              = time.time()

        for gid, identity in self._identities.items():

            # Skip immature identities (not enough frames to trust yet)
            if identity.gallery_feature is None or not identity.is_mature():
                continue

            # Skip IDs already active on this camera (O(1) lookup)
            if gid in self._active_global_ids_per_cam[query_cam_id]:
                continue

            # Temporal gate — ignore identities unseen for too long
            time_gap = max(0.0, now - identity.last_seen_time)
            if time_gap > TIME_GATE_SECONDS:
                continue

            # ── Multi-view max-pooling across the feature bank ──────────────
            bank_scores = [self._cosine_sim(query_feat, f) for f in identity.feature_bank]
            bank_scores.append(self._cosine_sim(query_feat, identity.gallery_feature))
            base_score  = max(bank_scores)

            # ── Temporal discount (up to −0.10 over TIME_GATE_SECONDS) ───────
            temporal_discount = (time_gap / TIME_GATE_SECONDS) * 0.10
            score = base_score - temporal_discount

            # Choose threshold based on whether the person was last seen on this cam
            is_same_cam = (identity.last_seen_cam == query_cam_id)
            threshold   = SAME_CAM_THRESHOLD if is_same_cam else CROSS_CAM_THRESHOLD

            if score >= threshold and score > best_score:
                best_score = score
                best_gid   = gid

            # Track the best candidate in the borderline band for color assist
            elif COLOR_ASSIST_LOW <= score < COLOR_ASSIST_HIGH and score > borderline_score:
                borderline_score = score
                borderline_gid   = gid

        # ── Confident match found ────────────────────────────────────────────
        if best_gid is not None:
            print(f"🔗 ReID Match  → gid={best_gid}  score={best_score:.3f}")
            return best_gid

        # ── Borderline match — use color histogram to confirm ────────────────
        if borderline_gid is not None and query_hist is not None:
            stored_hist  = self._identities[borderline_gid].color_hist
            color_sim    = _hist_similarity(query_hist, stored_hist)

            if color_sim >= COLOR_CONFIRM_SCORE:
                print(
                    f"🎨 ReID ColorAssist → gid={borderline_gid}  "
                    f"reid={borderline_score:.3f}  color={color_sim:.3f}"
                )
                return borderline_gid
            else:
                print(
                    f"❌ ReID Borderline rejected → gid={borderline_gid}  "
                    f"reid={borderline_score:.3f}  color={color_sim:.3f} (too low)"
                )

        return None

    # ── Stale track cleanup (call from camera_worker on each frame) ──────────
    def cleanup_stale(self, cam_id, active_local_ids):
        """
        Remove local→global mappings for tracks that ByteTrack has dropped,
        and sync the active-global-IDs set for this camera.
        Call once per frame per camera worker with the current set of live local IDs.
        """
        with self._lock:
            active_local_set  = set(active_local_ids)
            stale_keys        = []
            live_global_ids   = set()

            for (c, tid) in list(self._local_to_global.keys()):
                if c != cam_id:
                    continue
                if tid not in active_local_set:
                    stale_keys.append((c, tid))
                else:
                    live_global_ids.add(self._local_to_global[(c, tid)])

            for key in stale_keys:
                del self._local_to_global[key]

            # Keep the active-IDs set in sync
            self._active_global_ids_per_cam[cam_id] = live_global_ids

            # Periodic gallery expiry (~5 % of frames)
            if np.random.random() < 0.05:
                now           = time.time()
                active_anywhere = set(self._local_to_global.values())
                expired = [
                    gid for gid, identity in self._identities.items()
                    if (now - identity.last_seen_time) > GALLERY_EXPIRY_SECONDS
                    and gid not in active_anywhere
                ]
                for gid in expired:
                    del self._identities[gid]