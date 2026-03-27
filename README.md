# 🛡️ SYNTHETIC SENTINEL | MISSION CONTROL

An AI-powered video surveillance ecosystem featuring a premium web-based dashboard, real-time YOLOv8 object detection, interactive browser-based zone management, and natural language threat analysis.

![Smart CCTV Status](https://img.shields.io/badge/MISSION-ACTIVE-success?style=for-the-badge)
![Tech](https://img.shields.io/badge/Python-Flask-orange?style=for-the-badge)
![AI](https://img.shields.io/badge/YOLO-v8-blue?style=for-the-badge)

## 🚀 Key Features

*   **🖥️ Mission Control Dashboard**: A high-end, real-time web interface built with Stitch UI, supporting live MJPEG streaming and dynamic HUD overlays.
*   **🏗️ Web Zone Manager**: Draw and name security zones directly on the live video feed from your browser. Zones are persisted in SQLite and actively monitored.
*   **🚨 Unified Threat Log**: Real-time event stream tracking **Intrusions**, **Suspicious Stays** (lingering detected), and general **Person Detections**.
*   **💬 AI Intent Engine**: Integrated Llama 3.3 (via Groq) to parse natural language commands and filter surveillance focus (e.g., "Monitor all vehicles only").
*   **📊 Forensic Logging**: Automatic logging of entry/exit times, stay durations, and zone violations with metadata linked to specific video frames.
*   **📸 Snapshot System**: Automatic JPEG snapshots captured upon person detection for quick forensic review.

## 🛠️ Tech Stack

*   **Backend**: Flask (Python), SQLite3, Llama 3.3 (Groq API)
*   **Computer Vision**: OpenCV, Ultralytics (YOLOv8), Supervision
*   **Frontend**: Tailwind CSS, Vanilla JS, HTML5 Canvas (Drawing System)
*   **Deployment**: Git-ready for edge/cloud integration.

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `app.py` | Flask server & MJPEG streaming pipeline. |
| `db.py` | Database schema, unified alert queries, and persistence logic. |
| `templates/index.html` | The "Mission Control" frontend dashboard. |
| `zone_manager.py` | Backend logic for zone-based intrusion detection. |
| `event.py` | Lifecycle management for tracked objects (Entry/Stay/Exit). |
| `intent_manager.py` | LLM-powered query parsing and mode switching. |
| `detector.py` / `tracker.py` | Core CV detection and multi-object tracking. |

## ⚙️ Setup & Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/vijaykumar-777/smart_cctv.git
    cd smart_cctv
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Configuration**:
    Add your Groq API key to a `.env` file:
    ```bash
    GROQ_API_KEY="gsk_..."
    ```

4.  **Launch Mission Control**:
    ```bash
    python3 app.py
    ```
    Visit **`http://localhost:5000`** in your browser.

## 🎮 How to Use

1.  **Monitor**: View the live AI-augmented feed in the main panel.
2.  **Secure**: Click **`[ DRAW_NEW_ZONE ]`** in the sidebar, then drag your mouse over the video to define a restricted area.
3.  **Analyze**: Use the **INTENT_COMMAND** bar to talk to the system (e.g., "Look for suspicious persons").
4.  **Audit**: Review the **THREAT_LOG** for real-time alerts on intrusions or loitering.

## 📄 License

This project is licensed under the MIT License.
