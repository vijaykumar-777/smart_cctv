import cv2
import numpy as np
import threading


class GlobalTracker:
    """
    Cross-camera person re-identification using HSV color histograms.
    
    Each camera's ByteTrack assigns local IDs. This module maps them to
    globally unique IDs by matching appearance (clothing color) across cameras.
    """

    def __init__(self, match_threshold=0.45, hist_update_alpha=0.3):
        """
        Args:
            match_threshold: Minimum histogram correlation to consider a match (0-1).
                             Lower = more aggressive matching.
            hist_update_alpha: EMA blending factor for updating stored histograms.
        """
        self._lock = threading.Lock()
        self._next_global_id = 1
        self.match_threshold = match_threshold
        self.hist_update_alpha = hist_update_alpha

        # (cam_id, local_id) -> global_id
        self._local_to_global = {}

        # global_id -> { "histogram": np.array, "cam_id": str, "last_seen_frame": int }
        self._global_profiles = {}

    def _compute_histogram(self, crop):
        """Compute a normalized HSV color histogram from a bounding box crop."""
        if crop is None or crop.size == 0:
            return None
        try:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            # Use H and S channels (ignore V for lighting invariance)
            hist = cv2.calcHist(
                [hsv], [0, 1], None,
                [30, 32],          # 30 hue bins, 32 saturation bins
                [0, 180, 0, 256]
            )
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            return hist.flatten()
        except Exception:
            return None

    def _compare_histograms(self, hist_a, hist_b):
        """Compare two histograms using correlation (-1 to 1, higher = more similar)."""
        if hist_a is None or hist_b is None:
            return -1.0
        try:
            return cv2.compareHist(
                hist_a.reshape(-1, 1).astype(np.float32),
                hist_b.reshape(-1, 1).astype(np.float32),
                cv2.HISTCMP_CORREL
            )
        except Exception:
            return -1.0

    def resolve(self, cam_id, local_track_id, crop, cls_id=0):
        """
        Map a (cam_id, local_track_id) to a persistent global ID.
        
        Args:
            cam_id: Camera identifier string
            local_track_id: ByteTrack's local track ID
            crop: BGR image crop of the detected object
            cls_id: Class ID (only persons cls_id=0 get cross-camera matching)
            
        Returns:
            global_id: Integer global ID that persists across cameras
        """
        with self._lock:
            key = (cam_id, local_track_id)

            # Already mapped — update histogram and return
            if key in self._local_to_global:
                gid = self._local_to_global[key]
                self._update_profile(gid, crop, cam_id)
                return gid

            # New local track — try to match against existing global profiles
            new_hist = self._compute_histogram(crop)

            best_gid = None
            best_score = -1.0

            # Only attempt cross-camera matching for persons
            if cls_id == 0 and new_hist is not None:
                for gid, profile in self._global_profiles.items():
                    # Match against profiles from OTHER cameras
                    if profile["cam_id"] != cam_id:
                        score = self._compare_histograms(new_hist, profile["histogram"])
                        if score > best_score:
                            best_score = score
                            best_gid = gid

            if best_gid is not None and best_score >= self.match_threshold:
                # Matched — reuse global ID
                self._local_to_global[key] = best_gid
                self._update_profile(best_gid, crop, cam_id)
                return best_gid
            else:
                # No match — assign new global ID
                gid = self._next_global_id
                self._next_global_id += 1
                self._local_to_global[key] = gid
                self._global_profiles[gid] = {
                    "histogram": new_hist,
                    "cam_id": cam_id,
                }
                return gid

    def _update_profile(self, gid, crop, cam_id):
        """Update stored histogram with exponential moving average."""
        if gid not in self._global_profiles:
            return
        new_hist = self._compute_histogram(crop)
        if new_hist is None:
            return

        profile = self._global_profiles[gid]
        old_hist = profile["histogram"]
        if old_hist is not None:
            alpha = self.hist_update_alpha
            profile["histogram"] = (1 - alpha) * old_hist + alpha * new_hist
        else:
            profile["histogram"] = new_hist
        profile["cam_id"] = cam_id

    def cleanup_stale(self, active_keys):
        """Remove local mappings that are no longer actively tracked."""
        with self._lock:
            stale = [k for k in self._local_to_global if k not in active_keys]
            for k in stale:
                del self._local_to_global[k]
