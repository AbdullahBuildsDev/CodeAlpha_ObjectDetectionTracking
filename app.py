"""
CodeAlpha - Object Detection & Tracking (web app)
--------------------------------------------------
A polished Gradio web app around the YOLOv8 detector/tracker.

Tabs:
  1. Image   - detect objects in a photo
  2. Video   - detect + track objects across a video, with per-object counting
  3. Webcam  - live real-time detection + tracking from your camera

Everything runs locally and free. No paid APIs.

Run:  python app.py   ->  open the local URL it prints.

Author: Abdullah Awan  |  CodeAlpha AI Internship
"""

import os
import tempfile
from collections import defaultdict

import cv2
import gradio as gr

from src.detector import ObjectDetector

# Available models: trade speed for accuracy. Nano is fast and loads by default;
# bigger models detect smaller / more distant objects but run slower.
MODELS = {
    "Nano — fastest": "yolov8n.pt",
    "Small — balanced": "yolov8s.pt",
    "Medium — most accurate": "yolov8m.pt",
}

# Detectors are created on demand and cached, so each model loads only once.
_DETECTORS = {}


def get_detector(model_label):
    model_file = MODELS.get(model_label, "yolov8n.pt")
    if model_file not in _DETECTORS:
        _DETECTORS[model_file] = ObjectDetector(model_file)
    return _DETECTORS[model_file]


# --------------------------------------------------------------------- helpers
def _stats_html(title, total, by_class, extra=None):
    """Render detection stats as a clean HTML card."""
    rows = "".join(
        f"<div class='stat-row'><span>{name}</span>"
        f"<span class='stat-pill'>{count}</span></div>"
        for name, count in sorted(by_class.items(), key=lambda x: -x[1])
    ) or "<div class='stat-empty'>No objects detected yet</div>"

    extra_html = f"<div class='stat-extra'>{extra}</div>" if extra else ""
    return f"""
    <div class='stat-card'>
      <div class='stat-title'>{title}</div>
      <div class='stat-total'>{total}<span>objects</span></div>
      {extra_html}
      <div class='stat-list'>{rows}</div>
    </div>"""


# --------------------------------------------------------------------- image
def run_image(image, conf, model):
    if image is None:
        return None, _stats_html("Detection", 0, {})
    annotated, stats = get_detector(model).detect_image(image, conf=conf)
    return annotated, _stats_html("Detection", stats["total"], stats["by_class"])


# --------------------------------------------------------------------- video
def run_video(video_path, conf, model, progress=gr.Progress()):
    if not video_path:
        return None, _stats_html("Video Tracking", 0, {})

    detector = get_detector(model)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    out_path = os.path.join(tempfile.gettempdir(), "codealpha_tracked.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    id_to_class = {}        # unique tracked id -> class name (for unique counts)
    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        annotated, stats = detector.track_frame(frame, conf=conf)
        writer.write(annotated)

        # Remember each tracking id's class so we can count unique objects.
        id_to_class.update(stats.get("id_classes", {}))
        if n_frames:
            progress(frame_no / n_frames, desc=f"Tracking frame {frame_no}/{n_frames}")

    cap.release()
    writer.release()

    # Unique object count = number of distinct tracking IDs seen in the whole clip.
    unique_total = len(id_to_class)
    unique_by_class = {}
    for cname in id_to_class.values():
        unique_by_class[cname] = unique_by_class.get(cname, 0) + 1
    extra = f"🎯 {unique_total} unique objects across {frame_no} frames"
    return out_path, _stats_html("Video Tracking", unique_total,
                                 unique_by_class, extra=extra)


# --------------------------------------------------------------------- webcam
def run_webcam(frame, conf, model):
    """Live stream handler: frame is an RGB numpy array from the browser camera."""
    if frame is None:
        return None, _stats_html("Live", 0, {})
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    annotated, stats = get_detector(model).track_frame(bgr, conf=conf)
    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    extra = f"🟢 Live tracking · {len(stats.get('ids', []))} active IDs"
    return rgb, _stats_html("Live", stats["total"], stats["by_class"], extra=extra)


# --------------------------------------------------------------------- UI/CSS
CSS = """
.gradio-container {max-width: 1150px !important; margin: auto;}
#hero {text-align:center; padding: 26px 18px; border-radius: 18px; margin-bottom: 14px;
  background: linear-gradient(120deg,#4361ee 0%,#7209b7 60%,#f72585 100%); color:#fff;
  box-shadow: 0 10px 30px rgba(67,97,238,.35);}
#hero h1 {font-size: 2rem; margin:0; font-weight:800; letter-spacing:.3px;}
#hero p {opacity:.92; margin:6px 0 0;}
.stat-card {background:var(--block-background-fill); border:1px solid var(--border-color-primary);
  border-radius:16px; padding:18px; box-shadow:0 6px 18px rgba(0,0,0,.06);}
.stat-title {font-size:.8rem; text-transform:uppercase; letter-spacing:1px; opacity:.6;}
.stat-total {font-size:2.6rem; font-weight:800; line-height:1;}
.stat-total span {font-size:.85rem; font-weight:500; opacity:.6; margin-left:8px;}
.stat-extra {margin:8px 0 4px; font-size:.9rem; font-weight:600; color:#4361ee;}
.stat-list {margin-top:12px; display:flex; flex-direction:column; gap:6px;}
.stat-row {display:flex; justify-content:space-between; align-items:center;
  padding:7px 12px; border-radius:10px; background:var(--background-fill-secondary);}
.stat-pill {background:#4361ee; color:#fff; border-radius:20px; padding:2px 11px;
  font-weight:700; font-size:.85rem;}
.stat-empty {opacity:.5; padding:10px 0;}
footer {display:none !important;}
"""

theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="purple",
                       radius_size="lg", font=["Inter", "system-ui", "sans-serif"])

with gr.Blocks(title="CodeAlpha · Object Detection") as demo:
    gr.HTML("""
      <div id='hero'>
        <h1>🎥 Object Detection &amp; Tracking</h1>
        <p>Real-time AI vision with YOLOv8 · detect, track &amp; count — CodeAlpha AI Internship</p>
      </div>""")

    model_dd = gr.Dropdown(
        list(MODELS), value="Nano — fastest", label="🧠 AI Model",
        info="Bigger models detect smaller / distant objects but run slower.")

    with gr.Tabs():
        # ---- Image ----
        with gr.Tab("🖼️  Image"):
            with gr.Row():
                with gr.Column(scale=3):
                    img_in = gr.Image(label="Upload an image", type="numpy", height=380)
                    img_conf = gr.Slider(0.1, 0.9, value=0.4, step=0.05,
                                         label="Confidence threshold")
                    img_btn = gr.Button("Detect Objects", variant="primary")
                with gr.Column(scale=3):
                    img_out = gr.Image(label="Result", height=380)
                    img_stats = gr.HTML(_stats_html("Detection", 0, {}))
            img_btn.click(run_image, [img_in, img_conf, model_dd], [img_out, img_stats])

        # ---- Video ----
        with gr.Tab("🎬  Video"):
            with gr.Row():
                with gr.Column(scale=3):
                    vid_in = gr.Video(label="Upload a video")
                    vid_conf = gr.Slider(0.1, 0.9, value=0.4, step=0.05,
                                         label="Confidence threshold")
                    vid_btn = gr.Button("Track Objects", variant="primary")
                with gr.Column(scale=3):
                    vid_out = gr.Video(label="Tracked result", autoplay=True)
                    vid_stats = gr.HTML(_stats_html("Video Tracking", 0, {}))
            vid_btn.click(run_video, [vid_in, vid_conf, model_dd], [vid_out, vid_stats])

        # ---- Webcam ----
        with gr.Tab("📷  Live Webcam"):
            gr.Markdown("Allow camera access, then watch detection + tracking in real time.")
            with gr.Row():
                with gr.Column(scale=3):
                    cam_in = gr.Image(sources=["webcam"], streaming=True,
                                      type="numpy", label="Camera", height=380)
                    cam_conf = gr.Slider(0.1, 0.9, value=0.4, step=0.05,
                                         label="Confidence threshold")
                with gr.Column(scale=3):
                    cam_out = gr.Image(label="Live result", height=380)
                    cam_stats = gr.HTML(_stats_html("Live", 0, {}))
            cam_in.stream(run_webcam, [cam_in, cam_conf, model_dd], [cam_out, cam_stats],
                          stream_every=0.1, show_progress="hidden")


if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=theme, css=CSS)
