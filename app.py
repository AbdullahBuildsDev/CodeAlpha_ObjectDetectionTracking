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
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@600;700&display=swap');
.gradio-container{
  --body-background-fill:#060a12; --background-fill-primary:#060a12;
  --background-fill-secondary:rgba(255,255,255,.03);
  --block-background-fill:rgba(255,255,255,.035);
  --block-border-color:rgba(56,189,248,.14); --border-color-primary:rgba(56,189,248,.14);
  --body-text-color:#e7f3f9; --body-text-color-subdued:#8aa0b0;
  --button-primary-background-fill:linear-gradient(92deg,#22d3ee,#3b82f6);
  --button-primary-background-fill-hover:linear-gradient(92deg,#67e8f9,#60a5fa);
  --button-primary-text-color:#041018; --button-primary-border-color:transparent;
  --input-background-fill:rgba(255,255,255,.04); --block-radius:12px;
  font-family:'Inter',system-ui,sans-serif !important; max-width:1150px !important; margin:auto !important;}
body{background:
  radial-gradient(900px 500px at 82% -12%, rgba(34,211,238,.14), transparent 60%),
  radial-gradient(900px 520px at 8% -4%, rgba(59,130,246,.16), transparent 56%),
  #060a12 !important;}
#hero{padding:34px 8px 14px; text-align:center;}
#hero h1{font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:2.3rem; margin:0;
  letter-spacing:-.5px; background:linear-gradient(90deg,#67e8f9,#93c5fd);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;}
#hero p{color:#8aa0b0; margin:9px 0 0; font-size:.97rem; letter-spacing:.3px;}
.block,.form{backdrop-filter:blur(14px); box-shadow:0 10px 40px rgba(0,8,20,.4) !important;}
.block:has(.options),.form:has(.options){overflow:visible !important; backdrop-filter:none !important;}
button.primary,button[variant=primary]{font-weight:700 !important; letter-spacing:.3px;
  box-shadow:0 10px 26px rgba(34,211,238,.28) !important; transition:transform .15s ease;}
button.primary:hover{transform:translateY(-2px);}
label span{color:#a7bccb !important; font-weight:500 !important;}
ul.options,.options,div[role=listbox]{background:#0c1622 !important;
  border:1px solid rgba(56,189,248,.22) !important; border-radius:10px !important;
  box-shadow:0 14px 40px rgba(0,8,20,.6) !important; z-index:1000 !important; backdrop-filter:none !important;}
ul.options li,.options .item,li[role=option],[role=option]{color:#e7f3f9 !important; background:transparent !important;}
ul.options li:hover,.options .item:hover,.options .item.selected,li[role=option]:hover{
  background:rgba(34,211,238,.22) !important; color:#fff !important;}
.stat-card{background:rgba(255,255,255,.035); border:1px solid rgba(56,189,248,.16); border-radius:14px; padding:18px;}
.stat-title{font-size:.78rem; text-transform:uppercase; letter-spacing:2px; color:#5e9dc4;}
.stat-total{font-family:'JetBrains Mono',monospace; font-size:2.6rem; font-weight:700; line-height:1; color:#e7f3f9;}
.stat-total span{font-family:'Inter'; font-size:.85rem; font-weight:500; color:#8aa0b0; margin-left:8px;}
.stat-extra{margin:8px 0 4px; font-size:.9rem; font-weight:600; color:#67e8f9;}
.stat-list{margin-top:12px; display:flex; flex-direction:column; gap:6px;}
.stat-row{display:flex; justify-content:space-between; align-items:center;
  padding:8px 12px; border-radius:8px; background:rgba(56,189,248,.06);}
.stat-pill{background:linear-gradient(92deg,#22d3ee,#3b82f6); color:#041018; border-radius:20px;
  padding:2px 11px; font-weight:700; font-size:.85rem; font-family:'JetBrains Mono',monospace;}
.stat-empty{color:#8aa0b0; padding:10px 0;}
footer{display:none !important;}
"""

theme = gr.themes.Base(primary_hue="cyan", neutral_hue="slate")

# Force dark mode regardless of the visitor's system light/dark setting.
HEAD = """
<script>
(function(){
  var u = new URL(window.location);
  if (u.searchParams.get('__theme') !== 'dark') {
    u.searchParams.set('__theme','dark');
    window.location.replace(u.toString());
  }
})();
</script>
"""

with gr.Blocks(title="CodeAlpha · Object Detection") as demo:
    gr.HTML("""
      <div id='hero'>
        <h1>Object Detection &amp; Tracking</h1>
        <p>Real-time AI vision — detect, track &amp; count objects with YOLOv8</p>
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
    demo.launch(inbrowser=True, theme=theme, css=CSS, head=HEAD)
