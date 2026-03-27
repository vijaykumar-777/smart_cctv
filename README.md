# 🛡️ Smart CCTV System

An AI-powered video surveillance system featuring real-time object detection, multi-object tracking, interactive zone-based intrusion alerts, and a natural language query interface.

![Smart CCTV Demo](https://img.shields.io/badge/AI-Surveillance-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-orange)

## 🚀 Key Features

*   **🔍 AI Detection & Tracking**: High-performance object detection using YOLOv8 and robust multi-object tracking.
*   **🏗️ Interactive Zone Management**: Draw restricted zones directly on the video feed to monitor sensitive areas.
*   **🚨 Intrusion Alerts**: Real-time logging and console alerts when unauthorized persons enter restricted zones.
*   **💬 Natural Language Querying**: Talk to your CCTV. Filter video feeds by asking for specific objects or events (e.g., "Show me all cars").
*   **📊 Event Logging**: Comprehensive SQLite-based logging of person entry/exit times and intrusion events with video timestamps.
*   **🎬 Dual Operation Modes**:
    *   **Full Surveillance**: Continuous tracking of all supported classes (Person, Car, Bike, etc.).
    *   **Query Mode**: Intelligent filtering powered by Groq/Llama-3 to focus only on what matters to you.

## 🛠️ Tech Stack

*   **Computer Vision**: OpenCV, Ultralytics (YOLOv8), Supervision
*   **LLM Interface**: Groq API, LangChain
*   **Database**: SQLite3
*   **Language**: Python

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `app.py` | Main entry point for the surveillance system. |
| `detector.py` | Handles object detection using YOLOv8 models. |
| `tracker.py` | Implements multi-object tracking logic. |
| `zone_manager.py` | Manages interactive zone creation and intrusion detection. |
| `llm_parser.py` | Parses natural language queries into structured JSON using Groq. |
| `intent_manager.py` | Maps user queries to object classes and filters detections. |
| `db.py` | SQLite database schema and logging functions. |
| `event.py` | Logic for tracking entry/exit events for individuals. |

## ⚙️ Setup & Installation

1.  **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd smart_cctv
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set Environment Variables**:
    Create a `.env` file or export your Groq API key:
    ```bash
    export GROQ_API_KEY="your_api_key_here"
    ```

4.  **Run the Application**:
    ```bash
    python app.py
    ```

## 🎮 How to Use

1.  **Surveillance Mode**: Choose "1" for Full Surveillance or "2" for Query Mode on startup.
2.  **Drawing Zones**: Left-click and drag on the video window to create a restricted zone.
3.  **Querying**: In Query Mode, enter a prompt like *"Show me vehicles"* to filter the stream.
4.  **Reviewing Logs**: Intrusion events and person logs are saved in `cctv.db` for later analysis.

## 📄 License

This project is licensed under the MIT License.
