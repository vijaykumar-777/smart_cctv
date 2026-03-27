from datetime import datetime
import os
import cv2
from db import log_entry, log_exit, log_suspicious

EXIT_THRESHOLD = 2  # seconds before confirming exit
SUSPICIOUS_THRESHOLD = 10  # seconds stay to mark suspicious

SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


class EventManager:
    def __init__(self, camera_id=1):
        self.camera_id = camera_id
        self.active_tracks = {}  
        # track_id -> {
        #     entry_time,
        #     last_seen,
        #     snapshot_path,
        #     suspicious_flag
        # }

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

                log_entry(track_id)

                self.active_tracks[track_id] = {
                    "entry_time": entry_time,
                    "last_seen": current_time,
                    "snapshot": snapshot_path,
                    "suspicious": False
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

        # 🔹 HANDLE EXIT (with delay buffer)
        for track_id in list(self.active_tracks.keys()):
            if track_id not in current_ids:
                last_seen = self.active_tracks[track_id]["last_seen"]
                time_missing = (current_time - last_seen).total_seconds()

                if time_missing > EXIT_THRESHOLD:
                    entry_time = self.active_tracks[track_id]["entry_time"]
                    log_exit(track_id, entry_time)
                    del self.active_tracks[track_id]