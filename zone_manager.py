import cv2
import sqlite3
from datetime import datetime
from db import log_intrusion

class ZoneManager:
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.zones = []
        self.drawing = False
        self.start_point = None
        self.current_rect = None
        self.load_zones()
        self.active_intrusions = set()

    def load_zones(self):
        try:
            with sqlite3.connect("cctv.db", timeout=30) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM zones WHERE camera_id=?", (self.camera_id,))
                rows = cursor.fetchall()
                self.zones = []
                for row in rows:
                    self.zones.append({
                        "id": row[0],
                        "name": row[2],
                        "x1": row[3],
                        "y1": row[4],
                        "x2": row[5],
                        "y2": row[6],
                        "type": row[7]
                    })
        except Exception as e:
            print(f"Error loading zones: {e}")

    def save_zone(self, name, x1, y1, x2, y2, zone_type):
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        try:
            with sqlite3.connect("cctv.db", timeout=30) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO zones (camera_id, name, x1, y1, x2, y2, zone_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (self.camera_id, name, x1, y1, x2, y2, zone_type))
                conn.commit()
            self.load_zones()
        except Exception as e:
            print(f"Error saving zone: {e}")

    def delete_zone(self, zone_id):
        try:
            with sqlite3.connect("cctv.db", timeout=30) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM zones WHERE id=?", (zone_id,))
                conn.commit()
            self.load_zones()
        except Exception as e:
            print(f"Error deleting zone: {e}")

    def draw_zones(self, frame):
        for zone in self.zones:
            color = (0, 0, 255) if zone["type"] == "restricted" else (0, 255, 0)
            cv2.rectangle(frame, (zone["x1"], zone["y1"]),
                          (zone["x2"], zone["y2"]), color, 2)
            cv2.putText(frame, zone["name"],
                        (zone["x1"], zone["y1"] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def check_intrusion(self, track_id, center_x, center_y, video_file, video_time):
        for zone in self.zones:
            if zone["x1"] < center_x < zone["x2"] and zone["y1"] < center_y < zone["y2"]:
                key = (track_id, zone["id"])
                if key not in self.active_intrusions:
                    print(f"🚨 Intrusion by ID {track_id} in {zone['name']}")
                    log_intrusion(track_id, self.camera_id, zone["id"], video_file, video_time)
                    self.active_intrusions.add(key)
            else:
                key = (track_id, zone["id"])
                if key in self.active_intrusions:
                    self.active_intrusions.remove(key)