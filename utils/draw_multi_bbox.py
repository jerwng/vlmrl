import numpy as np
import cv2
import os
from typing import List, Dict, Optional


# Color palette for up to 10 distinct object types (RGB tuples)
BBOX_COLORS = [
    (255, 0, 0),      # red
    (0, 255, 0),      # green
    (0, 0, 255),      # blue
    (255, 255, 0),    # yellow
    (255, 0, 255),    # magenta
    (0, 255, 255),    # cyan
    (255, 128, 0),    # orange
    (128, 0, 255),    # purple
    (0, 255, 128),    # spring green
    (255, 128, 128),  # light red
]


def draw_multi_bbox_on_frame(
    frame: np.ndarray,          # (H, W, 3) RGB uint8
    detections: List[Dict],     # [{"bbox": [x1,y1,x2,y2], "label": str}, ...]
    thickness: int = 2,
) -> np.ndarray:                # (H, W, 3) RGB uint8, new array
    """
    Draw multiple bounding boxes on a frame, each label in a distinct color.

    Args:
        frame: (H, W, 3) uint8 RGB NumPy array
        detections: List of dicts, each with "bbox" (4-element list in 0-1000
                    scale) and "label" (string)
        thickness: Rectangle line thickness

    Returns:
        New annotated frame. If detections is empty, returns frame with
        "No objects detected" text overlay.
    """
    annotated = frame.copy()

    if len(detections) == 0:
        cv2.putText(annotated, "No objects detected", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        return annotated

    # Build label_to_color dict: assign colors in order of first appearance
    label_to_color: Dict[str, tuple] = {}
    for det in detections:
        lbl = det["label"]
        if lbl not in label_to_color:
            label_to_color[lbl] = BBOX_COLORS[len(label_to_color) % len(BBOX_COLORS)]

    h, w = frame.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        px1 = int(round(x1 / 1000.0 * w))
        py1 = int(round(y1 / 1000.0 * h))
        px2 = int(round(x2 / 1000.0 * w))
        py2 = int(round(y2 / 1000.0 * h))
        color = label_to_color[det["label"]]

        cv2.rectangle(annotated, (px1, py1), (px2, py2), color, thickness)

        # Draw label text above top-left corner (matching draw_bbox.py style)
        text = det["label"]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        # Filled dark background rectangle
        cv2.rectangle(annotated,
                      (px1, py1 - text_h - 6),
                      (px1 + text_w + 4, py1),
                      (0, 0, 0), -1)
        # White text
        cv2.putText(annotated, text, (px1 + 2, py1 - 4),
                    font, font_scale, (255, 255, 255), font_thickness)

    return annotated


def save_multi_bbox_frame(
    frame: np.ndarray,
    detections: List[Dict],
    output_path: str,
    thickness: int = 2,
) -> str:
    """Draw multi-bbox annotations and save to disk as PNG (RGB->BGR conversion). Returns output_path."""
    annotated = draw_multi_bbox_on_frame(frame, detections, thickness=thickness)
    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, bgr)
    return output_path
