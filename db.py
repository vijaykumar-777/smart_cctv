import sqlite3
from datetime import datetime

DB_NAME = "cctv.db"


def init_db():
    print("INIT_DB CALLED")
    try:
        with sqlite3.connect(DB_NAME, timeout=30) as conn:
            cursor = conn.cursor()

            # Person logs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS person_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id INTEGER,
                    entry_time TEXT,
                    exit_time TEXT,
                    duration REAL
                )
            """)

            # Cameras
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cameras (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    location TEXT
                )
            """)

            # Zones
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS zones (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id INTEGER,
                    name TEXT,
                    x1 INTEGER,
                    y1 INTEGER,
                    x2 INTEGER,
                    y2 INTEGER,
                    zone_type TEXT
                )
            """)

            # Intrusion logs with video info
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS intrusion_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id INTEGER,
                    camera_id INTEGER,
                    zone_id INTEGER,
                    timestamp TEXT,
                    video_file TEXT,
                    video_time REAL
                )
            """)

            # Suspicious stay logs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS suspicious_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id INTEGER,
                    camera_id INTEGER,
                    timestamp TEXT,
                    duration REAL
                )
            """)
            conn.commit()
    except Exception as e:
        print(f"Error in init_db: {e}")


def log_entry(track_id):
    try:
        with sqlite3.connect(DB_NAME, timeout=30) as conn:
            cursor = conn.cursor()
            entry_time = datetime.now().isoformat()

            cursor.execute("""
                INSERT INTO person_logs (track_id, entry_time)
                VALUES (?, ?)
            """, (track_id, entry_time))
            conn.commit()
    except Exception as e:
        print(f"Error in log_entry (ID {track_id}): {e}")


def log_exit(track_id, entry_time):
    try:
        with sqlite3.connect(DB_NAME, timeout=30) as conn:
            cursor = conn.cursor()
            exit_time = datetime.now()
            duration = (exit_time - entry_time).total_seconds()

            cursor.execute("""
                UPDATE person_logs
                SET exit_time = ?, duration = ?
                WHERE track_id = ? AND exit_time IS NULL
            """, (exit_time.isoformat(), duration, track_id))
            conn.commit()
    except Exception as e:
        print(f"Error in log_exit (ID {track_id}): {e}")


def log_intrusion(track_id, camera_id, zone_id, video_file, video_time):
    try:
        with sqlite3.connect(DB_NAME, timeout=30) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO intrusion_logs
                (track_id, camera_id, zone_id, timestamp, video_file, video_time)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                track_id,
                camera_id,
                zone_id,
                datetime.now().isoformat(),
                video_file,
                video_time
            ))
            conn.commit()
    except Exception as e:
        print(f"Error in log_intrusion (ID {track_id}): {e}")

def log_suspicious(track_id, camera_id, duration):
    try:
        with sqlite3.connect(DB_NAME, timeout=30) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO suspicious_logs (track_id, camera_id, timestamp, duration)
                VALUES (?, ?, ?, ?)
            """, (track_id, camera_id, datetime.now().isoformat(), duration))
            conn.commit()
    except Exception as e:
        print(f"Error in log_suspicious (ID {track_id}): {e}")
def get_recent_intrusions(limit=10):
    try:
        with sqlite3.connect(DB_NAME, timeout=30) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 'INTRUSION' as event_type, i.id, i.track_id, i.timestamp, z.name as detail 
                FROM intrusion_logs i
                LEFT JOIN zones z ON i.zone_id = z.id
                UNION ALL
                SELECT 'SUSPICIOUS_STAY' as event_type, id, track_id, timestamp, 'STAY_TIME: ' || CAST(duration AS INT) || 's' as detail
                FROM suspicious_logs
                UNION ALL
                SELECT 'PERSON_DETECTED' as event_type, id, track_id, entry_time as timestamp, 'CAMERA_FRAME' as detail 
                FROM person_logs
                ORDER BY timestamp DESC LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            logs = []
            for row in rows:
                logs.append({
                    "event_type": row[0],
                    "id": row[1],
                    "track_id": row[2],
                    "timestamp": row[3],
                    "detail": row[4] if row[4] else "Unknown Zone"
                })
            return logs
    except Exception as e:
        print(f"Error in get_recent_intrusions: {e}")
        return []

def get_stats():
    try:
        with sqlite3.connect(DB_NAME, timeout=30) as conn:
            cursor = conn.cursor()
            
            # Active people: count where exit_time is NULL
            cursor.execute("SELECT COUNT(*) FROM person_logs WHERE exit_time IS NULL")
            active_objects = cursor.fetchone()[0]
            
            # Intrusions total
            cursor.execute("SELECT COUNT(*) FROM intrusion_logs")
            total_intrusions = cursor.fetchone()[0]
            
            # Zones armed
            cursor.execute("SELECT COUNT(*) FROM zones")
            zones_armed = cursor.fetchone()[0]

            return {
                "active_objects": active_objects,
                "intrusions": total_intrusions,
                "zones_armed": zones_armed
            }
    except Exception as e:
        print(f"Error in get_stats: {e}")
        return {"active_objects": 0, "intrusions": 0, "zones_armed": 0}

def get_zones():
    try:
        with sqlite3.connect(DB_NAME, timeout=30) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, zone_type, x1, y1, x2, y2 FROM zones")
            rows = cursor.fetchall()
            zones = []
            for row in rows:
                zones.append({
                    "id": row[0],
                    "name": row[1],
                    "type": row[2],
                    "x1": row[3],
                    "y1": row[4],
                    "x2": row[5],
                    "y2": row[6]
                })
            return zones
    except Exception as e:
        print(f"Error in get_zones: {e}")
        return []
