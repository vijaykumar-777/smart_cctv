from datetime import datetime
import os
import cv2
import numpy as np
from db import log_entry, log_exit, log_suspicious, log_detection_index

EXIT_THRESHOLD = 2  # seconds before confirming exit
SUSPICIOUS_THRESHOLD = 10  # seconds stay to mark suspicious

SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# COCO class ID to name mapping (matches detector.py)
CLASS_NAMES = {0: "person", 2: "car", 3: "bike", 5: "bus", 7: "truck"}


def _extract_color_label(frame, x1, y1, x2, y2):
    """Crop the bounding box region and determine dominant color via HSV histogram."""
    try:
        crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size == 0:
            return "unknown"
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        mean_s = float(np.mean(s))
        mean_v = float(np.mean(v))

        # Low saturation → achromatic
        if mean_s < 30:
            if mean_v < 60:
                return "black"
            elif mean_v < 130:
                return "gray"
            elif mean_v < 200:
                return "silver"
            else:
                return "white"

        # Chromatic — use hue histogram
        hist = cv2.calcHist([h], [0], None, [180], [0, 180])
        dominant_hue = int(np.argmax(hist))

        if dominant_hue < 10 or dominant_hue >= 165:
            return "red"
        elif dominant_hue < 22:
            return "orange"
        elif dominant_hue < 35:
            return "yellow"
        elif dominant_hue < 80:
            return "green"
        elif dominant_hue < 130:
            return "blue"
        elif dominant_hue < 165:
            return "purple"
        else:
            return "gray"
    except Exception:
        return "unknown"


class EventManager:
    def __init__(self, camera_id=1):
        self.camera_id = camera_id
        self.active_tracks = {}
        self._suspicious_callbacks = []
        # track_id -> {
        #     entry_time,
        #     last_seen,
        #     snapshot_path,
        #     suspicious_flag,
        #     color_label,
        #     cls_id
        # }

    def on_suspicious(self, callback):
        """Register a callback: callback(track_id, duration)"""
        self._suspicious_callbacks.append(callback)

    def update(self, tracked_objects, frame):
        current_time = datetime.now()
        current_ids = {track_id for (_, _, _, _, track_id, _) in tracked_objects}

        # 🔹 HANDLE NEW ENTRIES
        for (x1, y1, x2, y2, track_id, cls_id) in tracked_objects:
            if track_id not in self.active_tracks:
                entry_time = current_time

                # Save snapshot
                person_img = frame[y1:y2, x1:x2]
                snapshot_path = os.path.join(
                    SNAPSHOT_DIR,
                    f"id_{track_id}_{entry_time.timestamp()}.jpg"
                )
                cv2.imwrite(snapshot_path, person_img)

                # Extract dominant color
                color_label = _extract_color_label(frame, x1, y1, x2, y2)
                object_class = CLASS_NAMES.get(cls_id, "object")

                log_entry(track_id)

                # Log to detection index
                try:
                    import camera_groups
                    group = camera_groups.get_group_for_camera(str(self.camera_id))
                    group_id = group["group_id"] if group else None
                    floor = group["floor"] if group else None
                except Exception:
                    group_id = None
                    floor = None

                log_detection_index(
                    cam_id=str(self.camera_id),
                    group_id=group_id,
                    floor=floor,
                    zone_name=None,
                    object_class=object_class,
                    color_label=color_label,
                    track_id=track_id,
                    timestamp=entry_time.isoformat(),
                    clip_path=None,
                    thumb_path=snapshot_path
                )

                self.active_tracks[track_id] = {
                    "entry_time": entry_time,
                    "last_seen": current_time,
                    "snapshot": snapshot_path,
                    "suspicious": False,
                    "color_label": color_label,
                    "cls_id": cls_id
                }
            else:
                self.active_tracks[track_id]["last_seen"] = current_time

        # 🔹 CHECK FOR SUSPICIOUS STAY
        for track_id, data in self.active_tracks.items():
            duration = (current_time - data["entry_time"]).total_seconds()

            if duration > SUSPICIOUS_THRESHOLD and not data["suspicious"]:
                print(f"⚠ Suspicious Stay Detected: ID {track_id}")
                log_suspicious(track_id, self.camera_id, duration)
                data["suspicious"] = True
                # Fire callbacks
                for cb in self._suspicious_callbacks:
                    try:
                        cb(track_id, duration)
                    except Exception:
                        pass

        # 🔹 HANDLE EXIT (with delay buffer)
        for track_id in list(self.active_tracks.keys()):
            if track_id not in current_ids:
                last_seen = self.active_tracks[track_id]["last_seen"]
                time_missing = (current_time - last_seen).total_seconds()

                if time_missing > EXIT_THRESHOLD:
                    entry_time = self.active_tracks[track_id]["entry_time"]
                    log_exit(track_id, entry_time)
                    del self.active_tracks[track_id]