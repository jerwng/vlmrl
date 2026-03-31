#!/usr/bin/env python3
"""Convert scenario-indexed bbox annotations into LoRA conversation JSON."""

import argparse
import ast
import json
from pathlib import Path


def load_open_vocab_prompt(eval_file: Path) -> str:
    """Extract OPEN_VOCAB_PROMPT string literal from eval_open_vocab.py."""
    source = eval_file.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "OPEN_VOCAB_PROMPT":
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        return node.value.value
    raise ValueError(f"OPEN_VOCAB_PROMPT not found in {eval_file}")


def build_assistant_payload(objects, image_size: int = 336):
    """Format assistant content with bbox conversion from xywh pixel coords to xyxy 0-1000 scale."""
    scale = 1000.0 / image_size
    converted = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        label = obj.get("label")
        bbox = obj.get("bbox")
        if (
            not isinstance(label, str)
            or not label.strip()
            or not isinstance(bbox, list)
            or len(bbox) != 4
        ):
            continue

        x1, y1, width, height = bbox
        if not all(isinstance(v, (int, float)) for v in (x1, y1, width, height)):
            continue

        x2 = x1 + width
        y2 = y1 + height
        # Scale pixel coordinates to 0-1000 range
        x1_s = int(round(x1 * scale))
        y1_s = int(round(y1 * scale))
        x2_s = int(round(x2 * scale))
        y2_s = int(round(y2 * scale))
        converted.append({"bbox": [x1_s, y1_s, x2_s, y2_s], "label": label.strip()})

    return json.dumps(converted, ensure_ascii=False)


def convert_dataset(input_json: Path, prompt: str, image_size: int = 336):
    data = json.loads(input_json.read_text(encoding="utf-8"))
    output = []

    for scenario_id, objects in data.items():
        output.append(
            {
                "id": scenario_id,
                "conversations": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": build_assistant_payload(objects, image_size)},
                ],
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Convert bbox scenario JSON into LoRA conversation JSON dataset."
    )
    parser.add_argument(
        "--input_json",
        type=Path,
        default=Path("job_scripts_utils/annotate_bboxes_output_finetune/bboxes.json"),
        help="Path to source bboxes.json",
    )
    parser.add_argument(
        "--eval_open_vocab",
        type=Path,
        default=Path("vlmrl/vlm/eval_open_vocab.py"),
        help="Path to eval_open_vocab.py containing OPEN_VOCAB_PROMPT",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("job_scripts_utils/annotate_bboxes_output_finetune/lora_finetune_dataset.json"),
        help="Path to write converted dataset JSON",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=336,
        help="Pixel size of rendered egocentric frames (default: 336). Used to scale coords to 0-1000.",
    )
    args = parser.parse_args()

    prompt = load_open_vocab_prompt(args.eval_open_vocab)
    records = convert_dataset(args.input_json, prompt, args.image_size)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} records to {args.output_json}")


if __name__ == "__main__":
    main()