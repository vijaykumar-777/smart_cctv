import numpy as np
import threading
from reid_model import ReIDFeatureExtractor


class GlobalTracker:
    """
    Cross-camera person re-identification using Fast ReID (OSNet).
    
    Uses deep appearance embeddings (512-dim) from a pre-trained ReID model
    to match the same person across different camera feeds.
    """

    def __init__(self, match_threshold=0.50):
        """
        Args:
            match_threshold: Minimum cosine similarity to consider a match (0-1).
                             Higher = stricter matching (fewer false positives).
        """
        self._lock = threading.Lock()
        self._next_global_id = 1
        self.match_threshold = match_threshold

        # (cam_id, local_id) -> global_id
        self._local_to_global = {}

        # global_id -> { "feature": np.array(512,), "source_cam": str, "all_cams": set }
        self._global_profiles = {}

        # Initialize the Fast ReID feature extractor
        self._reid = ReIDFeatureExtractor(model_name='osnet_x0_25')

    def _cosine_similarity(self, feat_a, feat_b):
        """Compute cosine similarity between two L2-normalized feature vectors."""
        if feat_a is None or feat_b is None:
            return -1.0
        return float(np.dot(feat_a, feat_b))

    def resolve(self, cam_id, local_track_id, crop, cls_id=0):
        """
        Map a (cam_id, local_track_id) to a persistent global ID.
        
        Returns:
            global_id: Integer global ID that persists across cameras
        """
        with self._lock:
            key = (cam_id, local_track_id)

            # Already mapped — return existing global ID
            if key in self._local_to_global:
                gid = self._local_to_global[key]
                # Track which cameras have seen this person
                if gid in self._global_profiles:
                    self._global_profiles[gid]["all_cams"].add(cam_id)
                return gid

            # New local track — extract ReID features
            if cls_id != 0:
                # Non-person objects: just assign a new global ID (no ReID)
                gid = self._next_global_id
                self._next_global_id += 1
                self._local_to_global[key] = gid
                self._global_profiles[gid] = {
                    "feature": None,
                    "source_cam": cam_id,
                    "all_cams": {cam_id},
                }
                return gid

            new_feature = self._reid.extract(crop)
            if new_feature is None:
                # Can't extract features — assign new ID
                gid = self._next_global_id
                self._next_global_id += 1
                self._local_to_global[key] = gid
                self._global_profiles[gid] = {
                    "feature": None,
                    "source_cam": cam_id,
                    "all_cams": {cam_id},
                }
                return gid

            # Try to match against ALL existing global profiles
            best_gid = None
            best_score = -1.0

            for gid, profile in self._global_profiles.items():
                if profile["feature"] is None:
                    continue

                # Skip profiles that are already mapped FROM this same camera
                # (to avoid self-matching within the same camera's tracks)
                already_mapped_from_this_camera = False
                for (c, _), g in self._local_to_global.items():
                    if g == gid and c == cam_id:
                        already_mapped_from_this_camera = True
                        break
                if already_mapped_from_this_camera:
                    continue

                score = self._cosine_similarity(new_feature, profile["feature"])
                if score > best_score:
                    best_score = score
                    best_gid = gid

            if best_gid is not None and best_score >= self.match_threshold:
                # Matched — reuse global ID
                self._local_to_global[key] = best_gid
                self._global_profiles[best_gid]["all_cams"].add(cam_id)
                print(f"🔗 ReID MATCH: {cam_id} local={local_track_id} → global={best_gid} (score={best_score:.3f})")
                return best_gid
            else:
                # No match — assign new global ID
                gid = self._next_global_id
                self._next_global_id += 1
                self._local_to_global[key] = gid
                self._global_profiles[gid] = {
                    "feature": new_feature,
                    "source_cam": cam_id,
                    "all_cams": {cam_id},
                }
                if best_gid is not None:
                    print(f"🆕 ReID NEW: {cam_id} local={local_track_id} → global={gid} (best_score={best_score:.3f} < {self.match_threshold})")
                return gid

    def cleanup_stale(self, active_keys):
        """Remove local mappings that are no longer actively tracked."""
        with self._lock:
            stale = [k for k in self._local_to_global if k not in active_keys]
            for k in stale:
                del self._local_to_global[k]
