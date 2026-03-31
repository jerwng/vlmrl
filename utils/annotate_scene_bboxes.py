#!/usr/bin/env python3
"""
Ground-truth bounding box annotation for MuJoCo room scenarios.

Projects 3D object body extents through the egocentric camera to produce
2D pixel bounding boxes. Saves raw and annotated egocentric frames plus a
consolidated bboxes.json per run.

Usage:
    python vlmrl/utils/annotate_scene_bboxes.py --scenarios B01 T04
    python vlmrl/utils/annotate_scene_bboxes.py --scenarios all
"""
import os
import sys
import json
import argparse

# Set headless rendering backend BEFORE importing mujoco.Renderer.
# Use EGL on SLURM/headless clusters; falls back to osmesa if EGL unavailable.
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import cv2
import mujoco

# Add vlmrl/ to sys.path so the utils package is importable
# when this script is run as a standalone file (not as a module).
_VLMRL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _VLMRL_ROOT not in sys.path:
    sys.path.insert(0, _VLMRL_ROOT)

from utils.mujoco_bbox_projection import (
    identify_object_bodies,
    compute_body_aabb_world,
    project_aabb_to_bbox2d,
    get_camera_id,
    LABEL_COLORS,
)
from utils.update_scenario import update_scenario_include, restore_scenario_include

CAMERA_NAME = "egocentric_cam"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ground-truth 2D bounding box annotations for MuJoCo room scenarios."
    )
    parser.add_argument(
        "--scenarios", nargs="+", required=True,
        help="One or more scenario IDs (e.g. T04 B01 C01) or 'all' to process every scenario."
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/feature05",
        help="Base output directory (default: outputs/feature05)."
    )
    parser.add_argument(
        "--image_size", type=int, default=336,
        help="Egocentric frame size in pixels, width and height (default: 336)."
    )
    parser.add_argument(
        "--go2_xml", type=str,
        default="loco-mujoco/loco_mujoco/models/unitree_go2/go2.xml",
        help="Path to the Go2 XML file (default: loco-mujoco/loco_mujoco/models/unitree_go2/go2.xml)."
    )
    return parser.parse_args()


def discover_scenarios(go2_xml_path: str) -> list:
    """Return sorted list of scenario IDs that have a scene_room.xml."""
    scenarios_dir = os.path.join(os.path.dirname(os.path.abspath(go2_xml_path)), "scenarios")
    if not os.path.isdir(scenarios_dir):
        raise FileNotFoundError(f"Scenarios directory not found: {scenarios_dir}")
    ids = sorted([
        d for d in os.listdir(scenarios_dir)
        if os.path.isfile(os.path.join(scenarios_dir, d, "scene_room.xml"))
    ])
    return ids


def validate_scenario(go2_xml_path: str, scenario_id: str) -> bool:
    """Return True if scenario_id has a scene_room.xml under the scenarios directory."""
    xml_dir = os.path.dirname(os.path.abspath(go2_xml_path))
    scene_xml = os.path.join(xml_dir, "scenarios", scenario_id, "scene_room.xml")
    return os.path.isfile(scene_xml)


def render_egocentric_frame(model, data, cam_id: int, image_size: int) -> np.ndarray:
    """Render a frame from the egocentric camera. Returns (H, W, 3) uint8 RGB array."""
    # Ensure the offscreen framebuffer is large enough for the requested image size.
    # MuJoCo defaults to 480; rendering at a larger size requires bumping this first.
    if model.vis.global_.offheight < image_size:
        model.vis.global_.offheight = image_size
    if model.vis.global_.offwidth < image_size:
        model.vis.global_.offwidth = image_size
    renderer = mujoco.Renderer(model, height=image_size, width=image_size)
    renderer.update_scene(data, camera=cam_id)
    # render() returns (H, W, 3) uint8 RGB
    frame = renderer.render()
    renderer.close()
    return frame


def draw_annotations(frame: np.ndarray, detections: list, image_size: int) -> np.ndarray:
    """Draw bounding box rectangles and labels on a copy of frame. Returns RGB array."""
    annotated = frame.copy()
    if not detections:
        cv2.putText(
            annotated, "No objects in view", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
        )
        return annotated

    for det in detections:
        label = det["label"]
        x, y, w, h = det["bbox"]
        color = LABEL_COLORS.get(label, (255, 255, 255))

        # Draw bounding box rectangle (2px)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        # Draw white text on dark background above top-left corner
        text = label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        # Background rectangle: clamp so it stays in frame
        bg_x1 = x
        bg_y1 = max(0, y - text_h - 6)
        bg_x2 = min(image_size, x + text_w + 4)
        bg_y2 = y
        cv2.rectangle(annotated, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.putText(
            annotated, text, (x + 2, max(text_h + 2, y - 4)),
            font, font_scale, (255, 255, 255), font_thickness
        )
    return annotated


def process_scenario(scenario_id: str, go2_xml_path: str, output_dir: str, image_size: int) -> list:
    """
    Process a single scenario: load model, project bboxes, render and save images.

    Returns the list of detection dicts for this scenario (empty list if no visible objects).
    """
    # Load model and data
    model = mujoco.MjModel.from_xml_path(go2_xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Get camera ID
    cam_id = get_camera_id(model, CAMERA_NAME)

    # Identify room-object bodies
    object_bodies = identify_object_bodies(model)
    body_summary = ", ".join(
        f"{name} ({label})" for _, name, label in object_bodies
    ) if object_bodies else "(none)"
    print(f"  Found {len(object_bodies)} object body/bodies: {body_summary}")

    # Compute and project bounding boxes
    detections = []
    for body_id, body_name, label in object_bodies:
        corners = compute_body_aabb_world(model, data, body_id)
        result = project_aabb_to_bbox2d(
            model, data, cam_id, corners, image_size, image_size
        )
        if result is not None:
            detections.append({
                "bbox": result["bbox"],
                "label": label,
                "truncated": result["truncated"],
            })

    if detections:
        bbox_summary = ", ".join(
            f"{d['label']} {d['bbox']}" for d in detections
        )
        print(f"  Projected {len(detections)} bbox(es): {bbox_summary}")
    else:
        print("  No bboxes projected (no objects in camera frustum).")

    # Render raw frame
    frame_rgb = render_egocentric_frame(model, data, cam_id, image_size)

    # Save raw frame (RGB -> BGR for cv2.imwrite)
    scenario_out_dir = os.path.join(output_dir, scenario_id)
    os.makedirs(scenario_out_dir, exist_ok=True)
    raw_path = os.path.join(scenario_out_dir, "raw.png")
    cv2.imwrite(raw_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    print(f"  Saved: {raw_path}")

    # Draw and save annotated frame
    annotated_rgb = draw_annotations(frame_rgb, detections, image_size)
    annotated_path = os.path.join(scenario_out_dir, "annotated.png")
    cv2.imwrite(annotated_path, cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
    print(f"  Saved: {annotated_path}")

    return detections


def main():
    args = parse_args()

    # Resolve go2.xml path (support relative paths from project root)
    go2_xml_path = os.path.abspath(args.go2_xml)
    if not os.path.isfile(go2_xml_path):
        print(f"[ERROR] go2.xml not found: {go2_xml_path}")
        sys.exit(1)

    # Resolve scenario list
    if args.scenarios == ["all"]:
        scenario_ids = discover_scenarios(go2_xml_path)
        print(f"Discovered {len(scenario_ids)} scenarios: {scenario_ids}")
    else:
        scenario_ids = []
        for sid in args.scenarios:
            if validate_scenario(go2_xml_path, sid):
                scenario_ids.append(sid)
            else:
                print(f"[WARNING] Scenario '{sid}' has no scene_room.xml — skipping.")

    if not scenario_ids:
        print("[ERROR] No valid scenarios to process.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    all_bboxes = {}         # Maps scenario_id -> list of detection dicts
    original_include = None
    total = len(scenario_ids)

    try:
        for idx, scenario_id in enumerate(scenario_ids, start=1):
            print(f"\n[{idx}/{total}] Processing scenario {scenario_id}...")

            # Mutate go2.xml to include this scenario's scene_room.xml
            prev_include = update_scenario_include(go2_xml_path, scenario_id)
            if original_include is None:
                original_include = prev_include  # Save only the first (true original)

            try:
                detections = process_scenario(
                    scenario_id, go2_xml_path, args.output_dir, args.image_size
                )
                # Strip 'truncated' from JSON output (implementation detail)
                all_bboxes[scenario_id] = [
                    {"bbox": d["bbox"], "label": d["label"]}
                    for d in detections
                ]
            except Exception as e:
                print(f"  [ERROR] Failed to process {scenario_id}: {e}")
                all_bboxes[scenario_id] = []

    finally:
        # Always restore go2.xml, even if the loop crashed
        if original_include is not None:
            try:
                restore_scenario_include(go2_xml_path, original_include)
                print(f"\nRestored go2.xml to original state.")
            except Exception as e:
                print(f"[CRITICAL] Failed to restore go2.xml: {e}")
                print(f"[CRITICAL] Manual restore needed. Original line: {original_include}")

    # Write consolidated bboxes.json
    bboxes_json_path = os.path.join(args.output_dir, "bboxes.json")
    with open(bboxes_json_path, "w", encoding="utf-8") as f:
        json.dump(all_bboxes, f, indent=2)
    print(f"\nSaved consolidated JSON: {bboxes_json_path}")
    print(f"Done. Processed {total} scenario(s).")


if __name__ == "__main__":
    main()
