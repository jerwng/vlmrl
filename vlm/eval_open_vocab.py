#!/usr/bin/env python3
"""
Single-pass open-vocabulary object detection using Qwen3-VL.
Detects all objects in a Go2 egocentric frame (MuJoCo mode) or a static image (image mode)
in a single VLM call, saves annotated frame and detections.json to output directory.

Usage (MuJoCo mode):
    python vlmrl/vlm/eval_open_vocab.py --agent_path models/go2/.../PPOJax_saved.pkl

Usage (image mode):
    python vlmrl/vlm/eval_open_vocab.py --image_path /path/to/frame.png
"""
import argparse
import json
import os
import re
import sys
import warnings
from datetime import datetime

import cv2
import jax
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJaxCollectVLMRL
from vlm_goal_predictor import create_vlm_predictor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.draw_multi_bbox import draw_multi_bbox_on_frame

os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'


OPEN_VOCAB_PROMPT = """You are controlling a robot navigating in an indoor environment.

Detect all visible instances of these objects: couch, table, chair, bookshelf, cardboard box, trash can.

Return ONLY a valid JSON array:
[{"bbox": [x1, y1, x2, y2], "label": "object name"}, ...]

Rules:
- Coordinates are in 0-1000 scale relative to image dimensions
- Tightly bound the VISIBLE portion of each object
- Return one entry per detected instance
- If none visible: []
"""


def query_model_extended(predictor, image, prompt: str, max_new_tokens: int = 512) -> str:
    """Replicate HuggingFaceVLMPredictor.query_model() with configurable max_new_tokens."""
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    text = predictor.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = predictor.processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    )
    # Cover both "cuda" and "cuda:N" device strings
    if predictor.device == "cuda" or predictor.device.startswith("cuda:"):
        inputs = inputs.to(predictor.device)
    with torch.no_grad():
        output_ids = predictor.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
        )
    generated_ids = [
        output_ids[i][len(inputs.input_ids[i]):]
        for i in range(len(inputs.input_ids))
    ]
    return predictor.processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def parse_detections(response: str) -> list:
    """
    Parse VLM response into a list of {"bbox": [x1,y1,x2,y2], "label": str} dicts.
    Uses cascading fallbacks: direct JSON -> substring JSON -> regex extraction.
    Invalid entries (coords out of [0,1000] or x1>=x2 or y1>=y2) are discarded with a warning.
    Returns a (possibly empty) list of validated dicts.
    """
    raw = response.strip()
    parsed = None

    # Fallback 1: direct json.loads -- expects a JSON array
    try:
        candidate = json.loads(raw)
        if isinstance(candidate, list):
            parsed = candidate
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback 2: extract first '[' to last ']' substring
    if parsed is None:
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            try:
                candidate = json.loads(match.group(0))
                if isinstance(candidate, list):
                    parsed = candidate
            except (json.JSONDecodeError, ValueError):
                pass

    # Fallback 3: regex extraction of individual bbox+label pairs
    if parsed is None:
        parsed = []
        bbox_pattern = r'"bbox"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
        label_pattern = r'"label"\s*:\s*"([^"]+)"'
        for bbox_match in re.finditer(bbox_pattern, raw):
            coords = [int(bbox_match.group(i)) for i in range(1, 5)]
            # Look for nearest label in surrounding context (+/-200 chars)
            start = max(0, bbox_match.start() - 200)
            end = min(len(raw), bbox_match.end() + 200)
            context = raw[start:end]
            label_match = re.search(label_pattern, context)
            label = label_match.group(1) if label_match else "unknown"
            parsed.append({"bbox": coords, "label": label})

    # Validate each entry: coords in [0,1000], x1<x2, y1<y2
    validated = []
    for det in parsed:
        if not isinstance(det, dict):
            continue
        bbox = det.get("bbox")
        label = det.get("label")
        if not isinstance(bbox, list) or len(bbox) != 4:
            warnings.warn(f"Discarding detection with invalid bbox: {det}")
            continue
        if not isinstance(label, str) or not label.strip():
            warnings.warn(f"Discarding detection with invalid label: {det}")
            continue
        x1, y1, x2, y2 = bbox
        try:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        except (TypeError, ValueError):
            warnings.warn(f"Discarding detection with non-integer coords: {det}")
            continue
        if not all(0 <= c <= 1000 for c in [x1, y1, x2, y2]):
            warnings.warn(f"Discarding detection with out-of-range coords: {det}")
            continue
        if x1 >= x2 or y1 >= y2:
            warnings.warn(f"Discarding detection with invalid bbox ordering: {det}")
            continue
        validated.append({"bbox": [x1, y1, x2, y2], "label": label.strip()})
    return validated


def capture_mujoco_frame(agent_path: str, use_mujoco: bool, seed: int) -> np.ndarray:
    """Load Go2 agent and environment, reset, capture egocentric frame."""
    print(f"Loading agent from: {agent_path}")
    agent_conf, agent_state = PPOJaxCollectVLMRL.load_agent(agent_path)
    config = agent_conf.config
    OmegaConf.set_struct(config, False)
    config.experiment.env_params["headless"] = True

    print("Setting up Go2 environment...")
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env = factory.make(
        **config.experiment.env_params,
        **config.experiment.task_factory.params,
        default_camera_mode="ego_head",
        viewer_size=(512, 512),
        show_visual_geoms=False,
    )

    rng = jax.random.key(seed)
    if use_mujoco:
        env.reset()
        frame = env.render(record=False)
    else:
        env = PPOJaxCollectVLMRL._wrap_env(env, config.experiment)
        n_envs = 1
        keys = jax.random.split(rng, n_envs + 1)
        rng, env_keys = keys[0], keys[1:]
        obs, env_state = env.reset(env_keys)
        frame = env.mjx_render(env_state, record=False)

    print(f"Captured frame: shape={frame.shape}, dtype={frame.dtype}")
    return frame


def main():
    parser = argparse.ArgumentParser(
        description='Single-pass open-vocabulary object detection with Qwen3-VL'
    )
    parser.add_argument('--agent_path', type=str, default=None,
                        help='Path to trained Go2 agent .pkl file (MuJoCo mode)')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to static image file (image mode)')
    parser.add_argument('--vlm_device', type=str, default='cuda',
                        help='Device for VLM model (default: cuda)')
    parser.add_argument('--output_dir', type=str, default='./open_vocab_output',
                        help='Directory for output files (default: ./open_vocab_output)')
    parser.add_argument('--use_mujoco', action='store_true',
                        help='Use MuJoCo backend instead of MJX (MuJoCo mode only)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for environment reset (MuJoCo mode only)')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Custom VLM prompt (overrides OPEN_VOCAB_PROMPT)')
    args = parser.parse_args()

    # Validate: exactly one of --agent_path or --image_path must be provided
    if args.agent_path is not None and args.image_path is not None:
        parser.error("Provide exactly one of --agent_path or --image_path, not both.")
    if args.agent_path is None and args.image_path is None:
        parser.error("Provide exactly one of --agent_path or --image_path.")

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Capture or load frame
    if args.agent_path is not None:
        source = "mujoco"
        frame = capture_mujoco_frame(args.agent_path, args.use_mujoco, args.seed)
    else:
        source = "image"
        print(f"Loading image from: {args.image_path}")
        img = Image.open(args.image_path).convert("RGB")
        frame = np.array(img)
        print(f"Loaded frame: shape={frame.shape}, dtype={frame.dtype}")

    # Step 2: Save raw frame (RGB -> BGR for imwrite)
    raw_path = os.path.join(args.output_dir, "raw_frame.png")
    cv2.imwrite(raw_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"Raw frame saved to: {raw_path}")

    # Step 3: Load VLM
    print(f"Loading VLM on {args.vlm_device}...")
    predictor = create_vlm_predictor('huggingface', device=args.vlm_device)
    print("VLM loaded.")

    # Step 4: Preprocess frame (returns PIL Image for Qwen3-VL)
    image = predictor.preprocess_frame(frame)

    # Step 5: Determine prompt
    prompt = args.prompt if args.prompt else OPEN_VOCAB_PROMPT

    # Step 6: Run VLM inference with extended token budget
    print("Running VLM inference...")
    response = query_model_extended(predictor, image, prompt, max_new_tokens=512)
    print(f"VLM raw response:\n{response}\n")

    # Step 7: Parse response
    detections = parse_detections(response)
    print(f"Parsed {len(detections)} valid detection(s).")

    # Step 8: Draw bboxes and save annotated frame
    annotated = draw_multi_bbox_on_frame(frame, detections)
    annotated_path = os.path.join(args.output_dir, "annotated_frame.png")
    cv2.imwrite(annotated_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    print(f"Annotated frame saved to: {annotated_path}")

    # Step 9: Write detections.json
    output_data = {
        "metadata": {
            "source": source,
            "image_path": args.image_path,
            "agent_path": args.agent_path,
            "vlm_model": "Qwen/Qwen3-VL-8B-Instruct",
            "timestamp": datetime.now().isoformat(timespec='seconds'),
            "seed": args.seed,
            "prompt": prompt,
        },
        "raw_response": response,
        "detections": detections,
        "total_detected": len(detections),
    }
    json_path = os.path.join(args.output_dir, "detections.json")
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"detections.json saved to: {json_path}")

    # Step 10: Print summary
    print("\n" + "=" * 60)
    print("Open-Vocabulary Detection Results")
    print("=" * 60)
    print(f"Source: {source}")
    print(f"Total objects detected: {len(detections)}")
    for i, det in enumerate(detections):
        print(f"  [{i+1}] label={det['label']!r}  bbox={det['bbox']}")
    print(f"\nOutput directory: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
