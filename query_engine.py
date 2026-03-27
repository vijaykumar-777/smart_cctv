import sqlite3
import re
from llm_parser import LLMParser



class QueryEngine:
    def __init__(self, db_path="cctv.db"):
        self.db_path = db_path

    def run_query(self, user_query):

       parser = LLMParser()
       intent = parser.parse(user_query)

       obj = intent.get("object")

       conn = sqlite3.connect(self.db_path)
       cursor = conn.cursor()

       if obj == "car":
        # for now same table (you can refine later)
        cursor.execute("""
            SELECT track_id, camera_id, zone_id,
                   timestamp, video_file, video_time
            FROM intrusion_logs
            ORDER BY timestamp DESC
            LIMIT 20
        """)

       else:
        cursor.execute("""
            SELECT track_id, camera_id, zone_id,
                   timestamp, video_file, video_time
            FROM intrusion_logs
            ORDER BY timestamp DESC
            LIMIT 20
        """)

       rows = cursor.fetchall()
       conn.close()

       return [
        {
            "type": "intrusion",
            "track_id": r[0],
            "camera_id": r[1],
            "zone_id": r[2],
            "timestamp": r[3],
            "video_file": r[4],
            "video_time": r[5],
        }
        for r in rows
    ]