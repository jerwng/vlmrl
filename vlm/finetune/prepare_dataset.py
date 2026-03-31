"""
prepare_dataset.py — Dataset preprocessing and stratified train/val splitting.

Reads the master dataset JSON from lora_finetune_dataset.json, validates
image paths and bbox coordinates, performs a tier-stratified 80/20
train-validation split (default seed=42), enriches each sample with an
absolute "image" path, and writes train_split.json and val_split.json.

Usage:
    python vlmrl/vlm/finetune/prepare_dataset.py \
        --dataset_json job_scripts_utils/annotate_bboxes_output_finetune/lora_finetune_dataset.json \
        --images_root job_scripts_utils/annotate_bboxes_output_finetune \
        --output_dir job_scripts_utils/annotate_bboxes_output_finetune \
        --seed 42
"""
import argparse
import collections
import json
import os
import random
import sys
from typing import List, Tuple


def load_and_validate(dataset_json: str, images_root: str):
    """
    Load the master dataset JSON and validate each sample.

    For each sample:
    - Resolves image path to <images_root>/<id>/raw.png
    - Checks the image file exists
    - Validates all bbox coordinates satisfy 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000
    - Adds "image" key (absolute path) to valid samples

    Returns:
        valid_samples (List[dict]): samples that passed all checks
        all_data (List[dict]): raw data from JSON (for statistics)
        skip_missing (int): count of samples skipped due to missing image
        skip_invalid (int): count of samples skipped due to invalid bbox
    """
    with open(dataset_json, "r") as f:
        all_data = json.load(f)

    valid_samples = []
    skip_missing = 0
    skip_invalid = 0

    for sample in all_data:
        sample_id = sample.get("id", "<unknown>")
        img_path = os.path.abspath(os.path.join(images_root, sample_id, "raw.png"))

        # Check image exists
        if not os.path.exists(img_path):
            print(
                f"WARNING: Image not found for sample '{sample_id}': {img_path}",
                file=sys.stderr,
            )
            skip_missing += 1
            continue

        # Validate bboxes in assistant response
        try:
            assistant_content = sample["conversations"][1]["content"]
            detections = json.loads(assistant_content)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(
                f"WARNING: Could not parse assistant content for sample '{sample_id}': {e}",
                file=sys.stderr,
            )
            skip_invalid += 1
            continue

        bbox_valid = True
        for det in detections:
            bbox = det.get("bbox", [])
            if len(bbox) != 4:
                print(
                    f"WARNING: Sample '{sample_id}' has bbox with wrong length: {bbox}",
                    file=sys.stderr,
                )
                skip_invalid += 1
                bbox_valid = False
                break
            x1, y1, x2, y2 = bbox
            if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
                print(
                    f"WARNING: Sample '{sample_id}' has invalid bbox coordinates: {bbox}. "
                    f"Requires 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000",
                    file=sys.stderr,
                )
                skip_invalid += 1
                bbox_valid = False
                break

        if not bbox_valid:
            continue

        # Valid sample — add image path and tier
        enriched = dict(sample)
        enriched["image"] = img_path
        valid_samples.append(enriched)

    return valid_samples, all_data, skip_missing, skip_invalid


def stratified_split(
    samples: List[dict], train_ratio: float = 0.8, seed: int = 42
) -> Tuple[List[dict], List[dict]]:
    """
    Perform a stratified train/val split grouped by tier (FT1-FT6).

    For each tier group, shuffles with the given seed and takes
    the first round(len * train_ratio) samples for training.

    Returns:
        train_samples, val_samples
    """
    # Group by tier prefix (first underscore-delimited token of the ID)
    tier_groups: dict = collections.defaultdict(list)
    for sample in samples:
        tier = sample["id"].split("_")[0]
        tier_groups[tier].append(sample)

    train_samples = []
    val_samples = []

    for tier in sorted(tier_groups.keys()):
        group = list(tier_groups[tier])
        rng = random.Random(seed)
        rng.shuffle(group)
        n_train = max(1, round(len(group) * train_ratio))
        train_samples.extend(group[:n_train])
        val_samples.extend(group[n_train:])

    # Shuffle final lists with same seed for random ordering across tiers
    rng_final = random.Random(seed)
    rng_final.shuffle(train_samples)
    rng_final.shuffle(val_samples)

    return train_samples, val_samples


def compute_label_counts(samples: List[dict]) -> dict:
    """Count all label occurrences across all samples' assistant responses."""
    counts: dict = collections.defaultdict(int)
    for sample in samples:
        try:
            assistant_content = sample["conversations"][1]["content"]
            detections = json.loads(assistant_content)
            for det in detections:
                label = det.get("label", "<unknown>")
                counts[label] += 1
        except (KeyError, IndexError, json.JSONDecodeError):
            pass
    return dict(counts)


def compute_tier_counts(samples: List[dict]) -> dict:
    """Count samples per tier."""
    counts: dict = collections.defaultdict(int)
    for sample in samples:
        tier = sample["id"].split("_")[0]
        counts[tier] += 1
    return dict(counts)


def print_statistics(
    all_data: List[dict],
    valid_samples: List[dict],
    skip_missing: int,
    skip_invalid: int,
    train_samples: List[dict],
    val_samples: List[dict],
) -> None:
    """Print dataset statistics to stdout."""
    print(f"\n{'='*60}")
    print("Dataset Statistics")
    print(f"{'='*60}")
    print(f"Total samples in JSON:     {len(all_data)}")
    print(f"Skipped (missing image):   {skip_missing}")
    print(f"Skipped (invalid bbox):    {skip_invalid}")
    print(f"Valid samples:             {len(valid_samples)}")

    print(f"\nPer-tier counts (valid):")
    tier_counts = compute_tier_counts(valid_samples)
    for tier in sorted(tier_counts.keys()):
        print(f"  {tier}: {tier_counts[tier]}")

    print(f"\nPer-label counts (valid):")
    label_counts = compute_label_counts(valid_samples)
    for label in sorted(label_counts.keys()):
        print(f"  {label}: {label_counts[label]}")

    print(f"\nTrain split size: {len(train_samples)}")
    train_tier_counts = compute_tier_counts(train_samples)
    for tier in sorted(train_tier_counts.keys()):
        print(f"  {tier}: {train_tier_counts[tier]}")

    print(f"\nVal split size:   {len(val_samples)}")
    val_tier_counts = compute_tier_counts(val_samples)
    for tier in sorted(val_tier_counts.keys()):
        print(f"  {tier}: {val_tier_counts[tier]}")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset: validate and split lora_finetune_dataset.json into train/val."
    )
    parser.add_argument(
        "--dataset_json",
        type=str,
        default="job_scripts_utils/annotate_bboxes_output_finetune/lora_finetune_dataset.json",
        help="Path to the master dataset JSON file.",
    )
    parser.add_argument(
        "--images_root",
        type=str,
        default="job_scripts_utils/annotate_bboxes_output_finetune",
        help="Root directory containing per-sample subdirectories with raw.png.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write train_split.json and val_split.json. Defaults to images_root.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting (default: 42).",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of samples to use for training (default: 0.8).",
    )
    parser.add_argument(
        "--all_train",
        action="store_true",
        help="If set, disables 80/20 split and uses 100%% of valid samples for training (0%% validation).",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.images_root

    # Resolve paths relative to cwd
    dataset_json = os.path.abspath(args.dataset_json)
    images_root = os.path.abspath(args.images_root)
    output_dir = os.path.abspath(args.output_dir)

    print(f"Loading dataset from: {dataset_json}")
    print(f"Images root:          {images_root}")
    print(f"Output directory:     {output_dir}")
    effective_train_ratio = 1.0 if args.all_train else args.train_ratio
    print(f"Train ratio:          {effective_train_ratio}")
    print(f"All-train mode:       {args.all_train}")
    print(f"Seed:                 {args.seed}")

    # Load and validate
    valid_samples, all_data, skip_missing, skip_invalid = load_and_validate(
        dataset_json, images_root
    )

    if len(valid_samples) == 0:
        print("ERROR: No valid samples found. Check dataset_json and images_root paths.", file=sys.stderr)
        sys.exit(1)

    # Stratified split
    train_samples, val_samples = stratified_split(valid_samples, effective_train_ratio, args.seed)

    # Print statistics
    print_statistics(all_data, valid_samples, skip_missing, skip_invalid, train_samples, val_samples)

    # Write output files
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train_split.json")
    val_path = os.path.join(output_dir, "val_split.json")

    with open(train_path, "w") as f:
        json.dump(train_samples, f, indent=2)

    with open(val_path, "w") as f:
        json.dump(val_samples, f, indent=2)

    print(
        f"Saved train_split.json ({len(train_samples)} samples) and "
        f"val_split.json ({len(val_samples)} samples) to {output_dir}"
    )


if __name__ == "__main__":
    main()
