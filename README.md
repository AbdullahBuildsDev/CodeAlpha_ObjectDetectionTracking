# CodeAlpha — Object Detection & Tracking (Task 4)

A **real-time object detection and tracking** system built with **YOLOv8** and a
polished **Gradio** web interface. Detect objects in images, track them across
videos with persistent IDs, or run **live detection from your webcam** — all
running **locally and 100% free** (no paid APIs).

![status](https://img.shields.io/badge/CodeAlpha-AI%20Internship-4361ee)

## ✨ Features
- 🖼️ **Image mode** — detect objects in any photo
- 🎬 **Video mode** — detect **and track** objects across frames, with
  **unique object counting** (via persistent tracking IDs)
- 📷 **Live webcam mode** — real-time detection + tracking from your camera
- 🎯 Tracking IDs using **ByteTrack** (an advanced SORT-family tracker)
- 📊 Live stats panel: total objects + per-class breakdown
- 🎚️ Adjustable confidence threshold
- 🎨 Custom-themed, modern, responsive UI

## 🛠️ Tech Stack
- **Python 3**
- **Ultralytics YOLOv8** — pre-trained object detection (80 classes, COCO)
- **ByteTrack** — multi-object tracking (built into Ultralytics)
- **OpenCV** — frame processing & custom annotations
- **Gradio** — web UI
- **PyTorch** — deep-learning backend (installed automatically)

## 🚀 How to Run

```bash
# 1. (Recommended) use a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
python app.py
```

The app opens in your browser. The YOLOv8 weights (`yolov8n.pt`, ~6 MB) download
automatically on first run, then everything works offline.

## 📖 How It Works
1. **Detect** — each frame is passed through YOLOv8, which outputs bounding
   boxes, class labels, and confidence scores.
2. **Track** — in video/webcam mode, **ByteTrack** assigns a persistent ID to
   every object so the same object keeps its ID across frames.
3. **Count** — unique tracking IDs are collected to count how many distinct
   objects appeared (total and per class).
4. **Annotate** — a custom drawer renders modern boxes, labels, and IDs.

## 📂 Project Structure
```
CodeAlpha_ObjectDetectionTracking/
├── app.py            # Gradio web app (entry point: Image / Video / Webcam)
├── src/
│   └── detector.py   # Core YOLOv8 detection + tracking engine
├── requirements.txt  # Dependencies
├── LICENSE
└── README.md
```

## 🧠 Model
Uses `yolov8n` (nano) for fast, real-time performance on a laptop. For higher
accuracy, change the model name in `app.py` to `yolov8s.pt`, `yolov8m.pt`, etc.

---
**CodeAlpha AI Internship — Task 4** · Built by Abdullah Awan
