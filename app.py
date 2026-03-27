import cv2
import threading
from flask import Flask, render_template, Response, jsonify, request
from dotenv import load_dotenv

load_dotenv()
from db import init_db, log_exit, get_stats, get_recent_intrusions, get_zones
from event import EventManager
from tracker import PersonTracker
from detector import HumanDetector
from zone_manager import ZoneManager
from intent_manager import IntentManager
from mode_manager import ModeManager

app = Flask(__name__)

VIDEO_FILE = "test.mp4"
CLASS_NAMES = {
    0: "Person",
    2: "Car",
    3: "Bike",
    5: "Bus",
    7: "Truck"
}

# Initialize the environment and persistent instances
init_db()
CAMERA_ID = 1

intent_manager = IntentManager()
mode_manager = ModeManager()
event_manager = EventManager(CAMERA_ID)
detector = HumanDetector()
tracker = PersonTracker()
zone_manager = ZoneManager(CAMERA_ID)

# Make the LLM query mode active by default for the dashboard
mode_manager.set_mode("query")

def generate_frames():
    cap = cv2.VideoCapture(VIDEO_FILE)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            # If video ends, restart it for a continuous stream demo
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        detections = detector.detect(frame)
        tracked_objects = tracker.update(frame, detections)
        event_manager.update(tracked_objects, frame)
        zone_manager.draw_zones(frame)

        for (x1, y1, x2, y2, track_id, cls_id) in tracked_objects:
            process_object = True
            
            # Filter objects if a natural language query restricts it
            if mode_manager.is_query_mode():
                process_object = intent_manager.match(cls_id)
            if mode_manager.is_query_mode() and not process_object:
                continue

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            label = CLASS_NAMES.get(cls_id, "Object")

            if process_object and cls_id == 0:
                 zone_manager.check_intrusion(track_id, center_x, center_y, VIDEO_FILE, video_time)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 4, (255, 0, 0), -1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def api_stats():
    return jsonify(get_stats())

@app.route('/api/logs')
def api_logs():
    return jsonify(get_recent_intrusions(limit=10))

@app.route('/api/zones', methods=['GET'])
def api_get_zones():
    return jsonify(get_zones())

@app.route('/api/zones', methods=['POST'])
def api_post_zones():
    data = request.json
    name = data.get('name', 'Custom Zone')
    x1_ratio = float(data.get('x1_ratio', 0))
    y1_ratio = float(data.get('y1_ratio', 0))
    x2_ratio = float(data.get('x2_ratio', 0))
    y2_ratio = float(data.get('y2_ratio', 0))
    
    cap = cv2.VideoCapture(VIDEO_FILE)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    
    x1 = int(x1_ratio * width)
    y1 = int(y1_ratio * height)
    x2 = int(x2_ratio * width)
    y2 = int(y2_ratio * height)
    
    zone_manager.save_zone(name, x1, y1, x2, y2, "restricted")
    return jsonify({"status": "success"})

@app.route('/api/zones/<int:zone_id>', methods=['DELETE'])
def api_delete_zones(zone_id):
    zone_manager.delete_zone(zone_id)
    return jsonify({"status": "success"})

@app.route('/api/intent', methods=['POST'])
def api_intent():
    data = request.json
    query = data.get('query', '')
    intent_manager.set_intent(query)
    mode_manager.set_mode("query")
    return jsonify(intent_manager.intent)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)