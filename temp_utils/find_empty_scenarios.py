#!/usr/bin/env python3
"""Analyze bboxes.json for empty scenarios and fully contained bboxes."""

import argparse
import json
from pathlib import Path


def bbox_to_xyxy(bbox):
    """Return bbox as (x1, y1, x2, y2), supporting xywh and xyxy inputs."""
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None

    try:
        a, b, c, d = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None

    # If third/fourth values are not greater than first/second, treat as [x, y, w, h].
    if c <= a or d <= b:
        w, h = c, d
        if w <= 0 or h <= 0:
            return None
        return (a, b, a + w, b + h)

    return (a, b, c, d)


def is_contained(inner, outer):
    """True when inner bbox is fully inside outer bbox (strictly not identical)."""
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer

    if ix1 < ox1 or iy1 < oy1 or ix2 > ox2 or iy2 > oy2:
        return False

    # Require strict containment to avoid counting identical boxes.
    return (ix1, iy1, ix2, iy2) != (ox1, oy1, ox2, oy2)


def find_contained_pairs(detections):
    """Find all detection pairs where one bbox is fully inside another."""
    parsed = []
    for idx, det in enumerate(detections):
        if not isinstance(det, dict):
            continue
        box = bbox_to_xyxy(det.get("bbox"))
        if box is None:
            continue
        parsed.append(
            {
                "idx": idx,
                "label": str(det.get("label", "unknown")),
                "box": box,
            }
        )

    pairs = []
    for i, inner in enumerate(parsed):
        for j, outer in enumerate(parsed):
            if i == j:
                continue
            if is_contained(inner["box"], outer["box"]):
                pairs.append((inner, outer))
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find scenarios with empty object lists and scenarios with fully "
            "contained bboxes in bboxes.json"
        )
    )
    parser.add_argument(
        "--bboxes_json",
        type=Path,
        default=Path("job_scripts_utils/annotate_bboxes_output_finetune/bboxes.json"),
        help="Path to source bboxes.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.bboxes_json.exists():
        raise FileNotFoundError(f"bboxes json not found: {args.bboxes_json}")

    data = json.loads(args.bboxes_json.read_text(encoding="utf-8"))
    empty_ids = []
    contained_by_scenario = {}

    for scenario_id, detections in data.items():
        if isinstance(detections, list) and len(detections) == 0:
            empty_ids.append(scenario_id)
            continue

        if isinstance(detections, list):
            contained_pairs = find_contained_pairs(detections)
            if contained_pairs:
                contained_by_scenario[scenario_id] = contained_pairs

    print(f"Total scenarios: {len(data)}")
    print(f"Scenarios with 0 objects: {len(empty_ids)}")
    print(
        "Scenarios with a bbox fully contained in another bbox: "
        f"{len(contained_by_scenario)}"
    )

    if empty_ids:
        print("Empty scenario IDs:")
        for scenario_id in empty_ids:
            print(scenario_id)
    else:
        print("No empty scenarios found.")

    if contained_by_scenario:
        print("Scenarios with fully contained bboxes:")
        for scenario_id in contained_by_scenario:
            print(scenario_id)

        print("Contained bbox details:")
        for scenario_id, pairs in contained_by_scenario.items():
            for inner, outer in pairs:
                print(
                    f"{scenario_id}: inner idx={inner['idx']} ({inner['label']}) "
                    f"inside outer idx={outer['idx']} ({outer['label']})"
                )
    else:
        print("No fully contained bbox pairs found.")


if __name__ == "__main__":
    main()
