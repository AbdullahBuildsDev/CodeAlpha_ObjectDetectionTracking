"""
CodeAlpha - Object Detection & Tracking (core engine)
------------------------------------------------------
Wraps a YOLOv8 model (Ultralytics) to:
  - detect objects in a single image
  - detect + TRACK objects across video frames (persistent IDs via ByteTrack)
  - count objects (total + per class + unique tracked IDs)

All processing is local and free - the YOLOv8 weights download once (~6 MB)
and then run fully offline. No paid API is used.

A custom annotator draws clean, modern boxes with a filled label tag and the
tracking ID, so the output looks polished rather than default.

Author: Abdullah Awan  |  CodeAlpha AI Internship
"""

from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO


# A pleasant, high-contrast color palette (BGR). Each class id is mapped to one
# of these deterministically, so a given class always keeps the same color.
_PALETTE = [
    (231, 76, 60), (46, 204, 113), (52, 152, 219), (155, 89, 182),
    (241, 196, 15), (26, 188, 156), (230, 126, 34), (52, 73, 94),
    (236, 64, 122), (0, 188, 212), (139, 195, 74), (255, 112, 67),
]


def _color_for(idx):
    """Return a stable BGR color for a class index."""
    r, g, b = _PALETTE[idx % len(_PALETTE)]
    return (b, g, r)  # OpenCV uses BGR


class ObjectDetector:
    def __init__(self, model_name="yolov8n.pt"):
        # yolov8n = "nano": fast, light, great for real-time demos on a laptop.
        self.model = YOLO(model_name)
        self.names = self.model.names  # {class_id: "person", ...}

    # ------------------------------------------------------------------ draw
    def _draw(self, frame, box, label, color, track_id=None):
        """Draw a single clean bounding box with a filled label tag."""
        x1, y1, x2, y2 = (int(v) for v in box)

        # Main rectangle.
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Small corner accents for a modern look.
        corner = max(10, (x2 - x1) // 8)
        for (cx, cy, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                                 (x1, y2, 1, -1), (x2, y2, -1, -1)]:
            cv2.line(frame, (cx, cy), (cx + dx * corner, cy), color, 4, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + dy * corner), color, 4, cv2.LINE_AA)

        # Label text (+ tracking id if present).
        text = f"{label}" + (f" #{track_id}" if track_id is not None else "")
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = max(y1, th + 8)
        cv2.rectangle(frame, (x1, ty - th - 8), (x1 + tw + 10, ty), color, -1, cv2.LINE_AA)
        cv2.putText(frame, text, (x1 + 5, ty - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def _annotate(self, frame, boxes, cls_ids, confs, ids=None):
        counts = Counter()
        for i, (box, c, conf) in enumerate(zip(boxes, cls_ids, confs)):
            name = self.names[int(c)]
            counts[name] += 1
            tid = int(ids[i]) if ids is not None else None
            self._draw(frame, box, f"{name} {conf:.0%}", _color_for(int(c)), tid)
        return frame, counts

    # -------------------------------------------------------------- image API
    def detect_image(self, image_rgb, conf=0.4):
        """Detect objects in a single RGB image. Returns (annotated_rgb, stats)."""
        if image_rgb is None:
            return None, {}
        frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        res = self.model.predict(frame, conf=conf, verbose=False)[0]

        if res.boxes is None or len(res.boxes) == 0:
            return image_rgb, {"total": 0, "by_class": {}}

        boxes = res.boxes.xyxy.cpu().numpy()
        cls_ids = res.boxes.cls.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        frame, counts = self._annotate(frame, boxes, cls_ids, confs)

        stats = {"total": int(len(boxes)), "by_class": dict(counts)}
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), stats

    # -------------------------------------------------------------- track API
    def track_frame(self, frame_bgr, conf=0.4):
        """Detect + track on one BGR frame (keeps IDs across calls). Returns (bgr, stats)."""
        res = self.model.track(frame_bgr, conf=conf, persist=True,
                               tracker="bytetrack.yaml", verbose=False)[0]

        if res.boxes is None or len(res.boxes) == 0:
            return frame_bgr, {"total": 0, "by_class": {}, "ids": []}

        boxes = res.boxes.xyxy.cpu().numpy()
        cls_ids = res.boxes.cls.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        ids = res.boxes.id.cpu().numpy() if res.boxes.id is not None else None

        frame_bgr, counts = self._annotate(frame_bgr, boxes, cls_ids, confs, ids)
        # Pair each tracking id with its class name (for unique per-class counts).
        id_classes = {}
        if ids is not None:
            for i, c in zip(ids, cls_ids):
                id_classes[int(i)] = self.names[int(c)]
        stats = {
            "total": int(len(boxes)),
            "by_class": dict(counts),
            "ids": list(id_classes.keys()),
            "id_classes": id_classes,
        }
        return frame_bgr, stats
