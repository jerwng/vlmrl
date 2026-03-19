#!/usr/bin/env python3
"""
Bounding box visualization for VLM goal prediction.
Loads Go2 env, captures first egocentric frame, runs VLM,
saves annotated image with bounding box overlay.

Uses the same MJX rendering path as eval_goal_vlm.py / play_policy()
to ensure identical frames.
"""
import argparse
import os
import jax
import numpy as np
import cv2
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJaxCollectVLMRL
from vlm_goal_predictor import create_vlm_predictor
from bbox_vlm_predictor import BBoxVLMPredictor
from omegaconf import OmegaConf

os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'


def main():
    parser = argparse.ArgumentParser(
        description='Capture Go2 egocentric frame and visualize VLM bbox prediction')
    parser.add_argument('--agent_path', type=str, required=True,
                        help='Path to trained Go2 agent pkl file')
    parser.add_argument('--vlm_device', type=str, default='cuda',
                        help='Device for VLM model (default: cuda)')
    parser.add_argument('--output_path', type=str,
                        default='./bbox_output/first_frame_bbox.png',
                        help='Path to save annotated image')
    parser.add_argument('--use_mujoco', action='store_true',
                        help='Use MuJoCo backend instead of MJX (default: MJX)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for environment reset')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Custom VLM prompt (if not provided, uses default)')
    args = parser.parse_args()

    # Load agent config
    print(f"Loading agent from: {args.agent_path}")
    agent_conf, agent_state = PPOJaxCollectVLMRL.load_agent(args.agent_path)
    config = agent_conf.config

    # Create environment
    print("Setting up Go2 environment...")
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    OmegaConf.set_struct(config, False)
    config.experiment.env_params["headless"] = True
    env = factory.make(
        **config.experiment.env_params,
        **config.experiment.task_factory.params,
        default_camera_mode="ego_head",
        viewer_size=(336, 336),
        show_visual_geoms=False,
    )

    # Reset and render, matching play_policy() paths exactly
    rng = jax.random.key(args.seed)

    if args.use_mujoco:
        # MuJoCo backend — matches play_policy(use_mujoco=True)
        env.reset()
        frame = env.render(record=False)
    else:
        # MJX backend (default) — wrap env + batched keys, matching play_policy() lines 67-96, 118
        env = PPOJaxCollectVLMRL._wrap_env(env, config.experiment)
        n_envs = 1
        keys = jax.random.split(rng, n_envs + 1)
        rng, env_keys = keys[0], keys[1:]
        obs, env_state = env.reset(env_keys)
        frame = env.mjx_render(env_state, record=False)

    print(f"Captured frame: shape={frame.shape}, dtype={frame.dtype}")

    # Save raw frame
    output_dir = os.path.dirname(args.output_path) or '.'
    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, 'first_frame_raw.png')
    bgr_raw = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(raw_path, bgr_raw)
    print(f"Raw frame saved to: {raw_path}")

    # Create VLM predictor and wrapper
    print(f"Loading VLM on {args.vlm_device}...")
    vlm_predictor = create_vlm_predictor('huggingface', device=args.vlm_device)
    bbox_predictor = BBoxVLMPredictor(vlm_predictor)

    # Run prediction and save annotated frame
    print("Running VLM inference...")
    goal = bbox_predictor.predict_and_save(
        frame, args.output_path,
        prompt=args.prompt,
        current_heading=0.0,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Bounding Box Visualization Results")
    print("=" * 60)
    print(f"Bbox coords (0-1000 scale): {bbox_predictor.last_bbox}")
    print(f"u value: {bbox_predictor.last_u_value}")
    print(f"Goal dict: {goal}")
    print(f"Raw VLM response: {bbox_predictor.last_raw_response}")
    print(f"\nRaw frame saved to: {raw_path}")
    print(f"Annotated frame saved to: {args.output_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
