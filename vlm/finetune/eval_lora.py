"""
eval_lora.py — Evaluate a saved LoRA checkpoint on the validation split.

Loads the base Qwen3-VL-8B-Instruct model, merges the specified LoRA
adapter, runs inference on all validation samples, and computes:
  - Mean IoU (all matched pairs)
  - Detection precision and recall @ IoU 0.5
  - Per-class precision and recall (6 object types)
  - Per-tier (FT1-FT6) precision and recall
  - JSON parse success rate

Results are printed as a formatted table and saved to a JSON file.

Usage:
    python vlmrl/vlm/finetune/eval_lora.py \
        --checkpoint_path models/vlm_lora/<run>/best/ \
        --val_json models/vlm_lora/<run>/val_split.json

Options:
    --save_visualizations   Save annotated images (GT=green, pred=blue)
    --include_baseline      Also evaluate the unfinetuned base model
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a LoRA checkpoint on the validation split."
    )
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to LoRA adapter directory (e.g., models/vlm_lora/<run>/best/)")
    parser.add_argument("--val_json", type=str, required=True,
                        help="Path to val_split.json")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Base model name or path (default: Qwen/Qwen3-VL-8B-Instruct)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Where to save eval_results.json. "
                             "Defaults to <checkpoint_path>/eval_results.json")
    parser.add_argument("--save_visualizations", action="store_true",
                        help="Save annotated images with GT (green) and predicted (blue) bboxes")
    parser.add_argument("--vis_output_dir", type=str, default=None,
                        help="Directory for visualization images. "
                             "Defaults to <checkpoint_path>/visualizations/")
    parser.add_argument("--include_baseline", action="store_true",
                        help="Also run the unfinetuned base model for baseline comparison")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Inference batch size (default: 1)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: 'cuda' or 'cuda:0' etc. (default: cuda)")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Max new tokens for generation (default: 256)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_with_lora(base_model_name: str, checkpoint_path: str, device: str):
    """
    Load base model + merge LoRA adapter.

    Returns:
        (model, processor)
    """
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    print(f"Loading processor from: {base_model_name}")
    processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)

    is_cuda = device == "cuda" or device.startswith("cuda:")
    print(f"Loading base model onto {device} in bfloat16...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if is_cuda else torch.float32,
        device_map={"": device} if is_cuda else "cpu",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from: {checkpoint_path}")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model = model.merge_and_unload()
    model.eval()
    print("LoRA adapter merged successfully.")

    return model, processor


def load_base_model_only(base_model_name: str, device: str):
    """
    Load base model without LoRA (for baseline evaluation).

    Returns:
        (model, processor)
    """
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    print(f"Loading base model (no LoRA) from: {base_model_name}")
    processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)

    is_cuda = device == "cuda" or device.startswith("cuda:")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if is_cuda else torch.float32,
        device_map={"": device} if is_cuda else "cpu",
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    model,
    processor,
    val_samples: List[dict],
    device: str,
    max_new_tokens: int,
) -> List[dict]:
    """
    Run inference on all validation samples.

    Returns:
        List of result dicts, each with:
            "id", "tier", "gt_detections", "pred_detections",
            "raw_response", "parse_success"
    """
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    from vlmrl.vlm.finetune.dataset import DETECTION_PROMPT

    results = []
    n = len(val_samples)

    for i, sample in enumerate(val_samples):
        sample_id = sample["id"]
        tier = sample_id.split("_")[0]

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Inference [{i+1}/{n}] — {sample_id}")

        # Load image
        image = Image.open(sample["image"]).convert("RGB")

        # Build user-only messages for inference
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": DETECTION_PROMPT},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )

        is_cuda = device == "cuda" or device.startswith("cuda:")
        if is_cuda:
            inputs = inputs.to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated_ids = [
            output_ids[j][inputs.input_ids.shape[1]:]
            for j in range(output_ids.shape[0])
        ]
        raw_response = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        # Parse ground-truth detections (already in 0-1000 scale)
        try:
            gt_detections = json.loads(sample["conversations"][1]["content"])
        except json.JSONDecodeError:
            gt_detections = []

        # Parse predicted detections
        pred_detections = None
        parse_success = False
        try:
            pred_detections = json.loads(raw_response)
            if isinstance(pred_detections, list):
                parse_success = True
            else:
                pred_detections = None
        except json.JSONDecodeError:
            import re
            match = re.search(r"\[.*\]", raw_response, re.DOTALL)
            if match:
                try:
                    pred_detections = json.loads(match.group())
                    if isinstance(pred_detections, list):
                        parse_success = True
                    else:
                        pred_detections = None
                except json.JSONDecodeError:
                    pred_detections = None

        results.append({
            "id": sample_id,
            "tier": tier,
            "gt_detections": gt_detections,
            "pred_detections": pred_detections if parse_success else [],
            "raw_response": raw_response,
            "parse_success": parse_success,
        })

    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """
    Compute Intersection over Union for two bboxes in [x1, y1, x2, y2] format
    with coordinates in 0-1000 scale.
    """
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def compute_metrics(results: List[dict], iou_threshold: float = 0.5) -> dict:
    """
    Compute detection metrics from inference results.

    Returns:
        {
            "overall": {"mean_iou": ..., "precision": ..., "recall": ...},
            "per_class": {label: {"precision": ..., "recall": ..., "tp": ..., "fp": ..., "fn": ...}},
            "per_tier": {tier: {"precision": ..., "recall": ..., "tp": ..., "fp": ..., "fn": ...}},
            "json_parse_rate": float,
        }
    """
    overall_tp = 0
    overall_fp = 0
    overall_fn = 0
    all_matched_ious = []

    class_stats: Dict[str, Dict[str, int]] = {}
    tier_stats: Dict[str, Dict[str, int]] = {}

    def get_stats(d, key):
        if key not in d:
            d[key] = {"tp": 0, "fp": 0, "fn": 0, "matched_ious": []}
        return d[key]

    for result in results:
        tier = result["tier"]
        gt_dets = result["gt_detections"]
        pred_dets = result["pred_detections"] if result["parse_success"] else []

        # Track which GT and pred boxes are matched
        gt_matched = [False] * len(gt_dets)
        pred_matched = [False] * len(pred_dets)

        # For each GT, find best-matching prediction (same label, highest IoU)
        for gt_idx, gt in enumerate(gt_dets):
            gt_label = gt.get("label", "")
            gt_bbox = gt.get("bbox", [])
            if len(gt_bbox) != 4:
                continue

            best_iou = 0.0
            best_pred_idx = -1

            for pred_idx, pred in enumerate(pred_dets):
                if pred_matched[pred_idx]:
                    continue
                if pred.get("label", "") != gt_label:
                    continue
                pred_bbox = pred.get("bbox", [])
                if len(pred_bbox) != 4:
                    continue
                iou = compute_iou(gt_bbox, pred_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = pred_idx

            if best_pred_idx >= 0 and best_iou >= iou_threshold:
                gt_matched[gt_idx] = True
                pred_matched[best_pred_idx] = True
                all_matched_ious.append(best_iou)
                get_stats(class_stats, gt_label)["tp"] += 1
                get_stats(class_stats, gt_label)["matched_ious"].append(best_iou)
                get_stats(tier_stats, tier)["tp"] += 1
                get_stats(tier_stats, tier)["matched_ious"].append(best_iou)
                overall_tp += 1
            else:
                get_stats(class_stats, gt_label)["fn"] += 1
                get_stats(tier_stats, tier)["fn"] += 1
                overall_fn += 1

        # Unmatched predictions are FP
        for pred_idx, pred in enumerate(pred_dets):
            if not pred_matched[pred_idx]:
                pred_label = pred.get("label", "")
                get_stats(class_stats, pred_label)["fp"] += 1
                get_stats(tier_stats, tier)["fp"] += 1
                overall_fp += 1

    def safe_div(num, den):
        return num / den if den > 0 else 0.0

    mean_iou = float(sum(all_matched_ious) / len(all_matched_ious)) if all_matched_ious else 0.0
    overall_precision = safe_div(overall_tp, overall_tp + overall_fp)
    overall_recall = safe_div(overall_tp, overall_tp + overall_fn)

    per_class = {}
    for label, stats in class_stats.items():
        per_class[label] = {
            "precision": safe_div(stats["tp"], stats["tp"] + stats["fp"]),
            "recall": safe_div(stats["tp"], stats["tp"] + stats["fn"]),
            "tp": stats["tp"],
            "fp": stats["fp"],
            "fn": stats["fn"],
        }

    per_tier = {}
    for tier, stats in tier_stats.items():
        per_tier[tier] = {
            "precision": safe_div(stats["tp"], stats["tp"] + stats["fp"]),
            "recall": safe_div(stats["tp"], stats["tp"] + stats["fn"]),
            "tp": stats["tp"],
            "fp": stats["fp"],
            "fn": stats["fn"],
        }

    json_parse_rate = sum(1 for r in results if r["parse_success"]) / len(results) if results else 0.0

    return {
        "overall": {
            "mean_iou": mean_iou,
            "precision": overall_precision,
            "recall": overall_recall,
            "tp": overall_tp,
            "fp": overall_fp,
            "fn": overall_fn,
        },
        "per_class": per_class,
        "per_tier": per_tier,
        "json_parse_rate": json_parse_rate,
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_results_table(metrics: dict) -> None:
    """Print a formatted metrics table to stdout."""
    overall = metrics["overall"]
    per_class = metrics["per_class"]
    per_tier = metrics["per_tier"]
    parse_rate = metrics["json_parse_rate"]

    print("\n=== Overall ===")
    print(f"  Mean IoU:   {overall['mean_iou']:.4f}")
    print(f"  Precision:  {overall['precision']:.4f}")
    print(f"  Recall:     {overall['recall']:.4f}")
    print(f"  JSON Parse: {100 * parse_rate:.1f}%")
    print(f"  TP: {overall['tp']}  FP: {overall['fp']}  FN: {overall['fn']}")

    print("\n=== Per-Class (IoU@0.5) ===")
    header = f"  {'Label':<20}  {'Precision':>9}  {'Recall':>6}  {'TP':>4}  {'FP':>4}  {'FN':>4}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label in sorted(per_class.keys()):
        s = per_class[label]
        print(
            f"  {label:<20}  {s['precision']:>9.4f}  {s['recall']:>6.4f}  "
            f"{s['tp']:>4}  {s['fp']:>4}  {s['fn']:>4}"
        )

    print("\n=== Per-Tier (IoU@0.5) ===")
    tier_header = f"  {'Tier':<6}  {'Precision':>9}  {'Recall':>6}  {'TP':>4}  {'FP':>4}  {'FN':>4}"
    print(tier_header)
    print("  " + "-" * (len(tier_header) - 2))
    for tier in sorted(per_tier.keys()):
        s = per_tier[tier]
        print(
            f"  {tier:<6}  {s['precision']:>9.4f}  {s['recall']:>6.4f}  "
            f"{s['tp']:>4}  {s['fp']:>4}  {s['fn']:>4}"
        )
    print()


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_visualizations(results: List[dict], val_samples: List[dict], vis_output_dir: str) -> None:
    """
    Save annotated images for each validation sample.
    GT bboxes drawn in green (0, 200, 0); predicted bboxes in blue (0, 0, 200).
    Coordinates are in 0-1000 scale; images are assumed to be 512x512.
    """
    from PIL import Image, ImageDraw, ImageFont

    os.makedirs(vis_output_dir, exist_ok=True)

    # Build a lookup from sample id to image path
    id_to_path = {s["id"]: s["image"] for s in val_samples}

    for result in results:
        sample_id = result["id"]
        img_path = id_to_path.get(sample_id)
        if img_path is None or not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        draw = ImageDraw.Draw(image)

        def scale_bbox_norm(bbox):
            # Both GT and pred are in 0-1000 scale in results (GT normalized in run_inference).
            x1, y1, x2, y2 = bbox
            return [x1 * w / 1000, y1 * h / 1000, x2 * w / 1000, y2 * h / 1000]

        # Draw GT bboxes in green
        for det in result.get("gt_detections", []):
            bbox = det.get("bbox", [])
            label = det.get("label", "")
            if len(bbox) == 4:
                sx1, sy1, sx2, sy2 = scale_bbox_norm(bbox)
                draw.rectangle([sx1, sy1, sx2, sy2], outline=(0, 200, 0), width=2)
                draw.text((sx1 + 2, sy1 + 2), f"GT:{label}", fill=(0, 200, 0))

        # Draw predicted bboxes in blue
        for det in result.get("pred_detections", []):
            bbox = det.get("bbox", [])
            label = det.get("label", "")
            if len(bbox) == 4:
                sx1, sy1, sx2, sy2 = scale_bbox_norm(bbox)
                draw.rectangle([sx1, sy1, sx2, sy2], outline=(0, 0, 200), width=2)
                draw.text((sx1 + 2, sy2 - 12), f"PD:{label}", fill=(0, 0, 200))

        out_path = os.path.join(vis_output_dir, f"{sample_id}.png")
        image.save(out_path)

    print(f"Visualizations saved to: {vis_output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Validate dependencies
    try:
        import peft  # noqa
    except ImportError:
        print("ERROR: peft is required. Install with: pip install peft>=0.12.0", file=sys.stderr)
        sys.exit(1)

    # Set default paths
    if args.output_path is None:
        args.output_path = os.path.join(args.checkpoint_path, "eval_results.json")
    if args.vis_output_dir is None:
        args.vis_output_dir = os.path.join(args.checkpoint_path, "visualizations")

    # Load validation split
    print(f"Loading validation split from: {args.val_json}")
    with open(args.val_json, "r") as f:
        val_samples = json.load(f)
    print(f"Validation samples: {len(val_samples)}")

    # Evaluate finetuned model
    print(f"\n--- Evaluating finetuned model: {args.checkpoint_path} ---")
    model, processor = load_model_with_lora(args.base_model, args.checkpoint_path, args.device)

    print(f"\nRunning inference on {len(val_samples)} validation samples...")
    results = run_inference(model, processor, val_samples, args.device, args.max_new_tokens)

    metrics = compute_metrics(results)
    print_results_table(metrics)

    output_json = {
        "checkpoint": args.checkpoint_path,
        "finetuned": metrics,
    }

    # Visualizations
    if args.save_visualizations:
        save_visualizations(results, val_samples, args.vis_output_dir)

    # Optional baseline evaluation
    if args.include_baseline:
        print(f"\n--- Evaluating base model (no LoRA): {args.base_model} ---")
        del model  # Free memory
        import torch
        torch.cuda.empty_cache()

        base_model, base_processor = load_base_model_only(args.base_model, args.device)
        print(f"\nRunning baseline inference on {len(val_samples)} validation samples...")
        base_results = run_inference(
            base_model, base_processor, val_samples, args.device, args.max_new_tokens
        )
        base_metrics = compute_metrics(base_results)
        print("\n=== Baseline Results ===")
        print_results_table(base_metrics)
        output_json["baseline"] = base_metrics

        if args.save_visualizations:
            baseline_vis_dir = os.path.join(args.vis_output_dir, "baseline")
            save_visualizations(base_results, val_samples, baseline_vis_dir)

    # Save results JSON
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()
