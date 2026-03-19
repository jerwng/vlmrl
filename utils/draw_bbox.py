import numpy as np
import cv2
import os
from typing import Optional, Tuple


def draw_bbox_on_frame(
    frame: np.ndarray,
    bbox_coords: Optional[Tuple[int, int, int, int]],
    label: Optional[str] = None,
    u_value: Optional[float] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw a bounding box on a frame.

    Args:
        frame: (H, W, 3) uint8 RGB NumPy array
        bbox_coords: (x1, y1, x2, y2) in 0-1000 normalized scale, or None
        label: Optional text label
        u_value: Optional u value to display
        color: RGB color tuple for the rectangle
        thickness: Rectangle line thickness

    Returns:
        New (H, W, 3) uint8 RGB array with annotations drawn
    """
    annotated = frame.copy()

    if bbox_coords is None:
        cv2.putText(annotated, "No detection", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        return annotated

    x1, y1, x2, y2 = bbox_coords
    h, w = frame.shape[:2]
    px1 = int(round(x1 / 1000.0 * w))
    py1 = int(round(y1 / 1000.0 * h))
    px2 = int(round(x2 / 1000.0 * w))
    py2 = int(round(y2 / 1000.0 * h))

    cv2.rectangle(annotated, (px1, py1), (px2, py2), color, thickness)

    # Build label text
    text = None
    if label is not None and u_value is not None:
        text = f"{label} | u={u_value:.2f}"
    elif u_value is not None:
        text = f"u={u_value:.2f}"
    elif label is not None:
        text = label

    if text is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        cv2.rectangle(annotated,
                      (px1, py1 - text_h - 6),
                      (px1 + text_w + 4, py1),
                      (0, 0, 0), -1)
        cv2.putText(annotated, text, (px1 + 2, py1 - 4),
                    font, font_scale, (255, 255, 255), font_thickness)

    return annotated


def save_annotated_frame(
    frame: np.ndarray,
    bbox_coords: Optional[Tuple[int, int, int, int]],
    output_path: str,
    label: Optional[str] = None,
    u_value: Optional[float] = None,
) -> str:
    """
    Draw bbox on frame and save to disk as PNG.

    Args:
        frame: (H, W, 3) uint8 RGB NumPy array
        bbox_coords: (x1, y1, x2, y2) in 0-1000 scale, or None
        output_path: Path to save the annotated PNG
        label: Optional text label
        u_value: Optional u value to display

    Returns:
        The output path string
    """
    annotated = draw_bbox_on_frame(frame, bbox_coords, label=label, u_value=u_value)
    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, bgr)
    return output_path
