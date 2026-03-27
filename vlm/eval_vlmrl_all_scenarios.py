#!/usr/bin/env python3
"""
Batch VLM+RL navigation evaluation across all 43 Go2 evaluation scenarios.
Loads VLM once; iterates scenarios; runs navigation episodes; produces MP4 videos + results.json.

Video path control: recorder_params is passed through factory.make() to VideoRecorder.
Each evaluation creates a new environment so VideoRecorder uses the correct video_name.
"""
import argparse
import json
import os
import sys
import jax
import cv2
import numpy as np
from datetime import datetime
from omegaconf import OmegaConf
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJaxCollectVLMRL
from vlm_goal_predictor import create_vlm_predictor

# Add vlmrl/ parent to sys.path to reach vlmrl/utils/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.update_scenario import update_scenario_include, restore_scenario_include

os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'


OBJECT_TYPE_CODES = {
    "couch": "C",
    "table": "T",
    "chair": "CH",
    "bookshelf": "B",
    "cardboard box": "BX",
    "trash can": "TC",
}

SCENARIO_CONFIG = {
    # Chapter 5 - COUCH scenarios
    "C01": {"target": "couch", "distractors": []},
    "C02": {"target": "couch", "distractors": ["chair"]},
    "C03": {"target": "couch", "distractors": ["chair", "bookshelf"]},
    "C04": {"target": "couch", "distractors": ["chair", "table"]},
    "C05": {"target": "couch", "distractors": ["chair"]},
    "C06": {"target": "couch", "distractors": []},
    # Chapter 5 - TABLE scenarios
    "T01": {"target": "table", "distractors": []},
    "T02": {"target": "table", "distractors": ["chair"]},
    "T03": {"target": "table", "distractors": ["bookshelf", "cardboard box"]},
    "T04": {"target": "table", "distractors": ["bookshelf", "chair"]},
    "T05": {"target": "table", "distractors": ["bookshelf", "trash can"]},
    "T06": {"target": "table", "distractors": []},
    # Chapter 5 - CHAIR scenarios
    "CH01": {"target": "chair", "distractors": []},
    "CH02": {"target": "chair", "distractors": ["couch"]},
    "CH03": {"target": "chair", "distractors": ["couch", "table"]},
    "CH04": {"target": "chair", "distractors": ["couch"]},
    "CH05": {"target": "chair", "distractors": ["table", "bookshelf"]},
    "CH06": {"target": "chair", "distractors": ["trash can"]},
    # Chapter 5 - BOOKSHELF scenarios
    "B01": {"target": "bookshelf", "distractors": []},
    "B02": {"target": "bookshelf", "distractors": ["table"]},
    "B03": {"target": "bookshelf", "distractors": ["chair", "table"]},
    "B04": {"target": "bookshelf", "distractors": ["table", "chair", "cardboard box"]},
    "B05": {"target": "bookshelf", "distractors": ["table", "chair"]},
    "B06": {"target": "bookshelf", "distractors": []},
    # Chapter 5 - CARDBOARD BOX scenarios
    "BX01": {"target": "cardboard box", "distractors": []},
    "BX02": {"target": "cardboard box", "distractors": ["trash can"]},
    "BX03": {"target": "cardboard box", "distractors": ["table", "chair"]},
    "BX04": {"target": "cardboard box", "distractors": ["table", "trash can"]},
    "BX05": {"target": "cardboard box", "distractors": ["chair"]},
    "BX06": {"target": "cardboard box", "distractors": ["couch"]},
    # Chapter 5 - TRASH CAN scenarios
    "TC01": {"target": "trash can", "distractors": []},
    "TC02": {"target": "trash can", "distractors": ["cardboard box"]},
    "TC03": {"target": "trash can", "distractors": ["chair", "bookshelf"]},
    "TC04": {"target": "trash can", "distractors": ["chair", "table", "cardboard box"]},
    "TC05": {"target": "trash can", "distractors": ["couch"]},
    "TC06": {"target": "trash can", "distractors": []},
    # Chapter 6 - Cross-Cutting scenarios
    "X01": {"target": "chair", "distractors": ["couch"]},
    "X02": {"target": "table", "distractors": ["bookshelf"]},
    "X03": {"target": "cardboard box", "distractors": ["table"]},
    "X04": {"target": "couch", "distractors": ["chair"]},
    "X05": {"target": "cardboard box", "distractors": ["bookshelf"]},
    "X06": {"target": "chair", "distractors": []},
    "X07": {"target": "trash can", "distractors": []},
}

# Canonical scenario order (deterministic iteration)
SCENARIO_ORDER = list(SCENARIO_CONFIG.keys())


def main():
    parser = argparse.ArgumentParser(
        description='Batch VLM+RL navigation evaluation across all Go2 scenarios'
    )
    parser.add_argument('--agent_path', type=str, required=True,
                        help='Path to trained Go2 agent pkl file')
    parser.add_argument('--vlm_device', type=str, default='cuda:1',
                        help='Device for VLM model (default: cuda:1)')
    parser.add_argument('--output_dir', type=str,
                        default='./eval_scenarios_recordings',
                        help='Directory for output MP4 videos and results.json')
    parser.add_argument('--n_steps', type=int, default=500,
                        help='Number of simulation steps per evaluation episode')
    parser.add_argument('--vlm_update_freq', type=int, default=10,
                        help='How often the VLM predicts a new heading (in steps)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for environment reset')
    parser.add_argument('--scenarios', type=str, default=None,
                        help='Comma-separated subset of scenario IDs, e.g. C01,C02,X01')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use deterministic policy (no action sampling noise)')
    args = parser.parse_args()

    # Resolve scenario list
    if args.scenarios is not None:
        requested = [s.strip() for s in args.scenarios.split(',')]
        invalid = [s for s in requested if s not in SCENARIO_CONFIG]
        if invalid:
            raise ValueError(f"Unknown scenario IDs: {invalid}. "
                             f"Valid IDs: {list(SCENARIO_CONFIG.keys())}")
        scenarios_to_run = [s for s in SCENARIO_ORDER if s in requested]
    else:
        scenarios_to_run = SCENARIO_ORDER

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine go2.xml path (relative to this script's location)
    go2_xml_path = os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'loco-mujoco',
        'loco_mujoco', 'models', 'unitree_go2', 'go2.xml'
    )
    go2_xml_path = os.path.abspath(go2_xml_path)
    if not os.path.exists(go2_xml_path):
        raise FileNotFoundError(f"go2.xml not found: {go2_xml_path}")

    # Load agent config once (env is recreated per evaluation; only config is reused)
    print(f"Loading agent config from: {args.agent_path}")
    agent_conf, agent_state = PPOJaxCollectVLMRL.load_agent(args.agent_path)
    config = agent_conf.config
    OmegaConf.set_struct(config, False)
    config.experiment.env_params["headless"] = True

    # Load VLM once (loading takes ~45s; reused across all 90 evaluations)
    print(f"Loading VLM on {args.vlm_device}...")
    vlm_predictor = create_vlm_predictor('huggingface', device=args.vlm_device)
    print("VLM loaded.")

    original_include = None
    results = {}
    eval_counter = 0

    # Total evaluations = 1 target + N distractors per scenario
    total_evals = sum(
        1 + len(SCENARIO_CONFIG[sid]["distractors"])
        for sid in scenarios_to_run
    )

    print(f"Scenarios to run: {len(scenarios_to_run)}, Total evaluations: {total_evals}")
    print(f"Output dir: {args.output_dir}")
    print("-" * 60)

    try:
        for scenario_id in scenarios_to_run:
            config_entry = SCENARIO_CONFIG[scenario_id]
            scenario_target = config_entry["target"]
            distractors = config_entry["distractors"]

            # XML swap (once per scenario; all evaluations within share the same scene)
            try:
                prev_include = update_scenario_include(go2_xml_path, scenario_id)
                if original_include is None:
                    # First swap: capture the original line for restore
                    original_include = prev_include
            except Exception as e:
                print(f"[ERROR] Failed to swap XML for {scenario_id}: {e}")
                # Record error for all evaluations in this scenario
                eval_targets_for_error = [scenario_target] + distractors
                for eval_target in eval_targets_for_error:
                    eval_counter += 1
                    is_distractor = (eval_target != scenario_target)
                    if is_distractor:
                        stem = f"{scenario_id}{OBJECT_TYPE_CODES[eval_target]}"
                    else:
                        stem = scenario_id
                    results[stem] = {
                        "scenario_id": scenario_id,
                        "target": eval_target,
                        "is_distractor": is_distractor,
                        "status": "failed",
                        "error": f"XML swap failed: {str(e)}",
                    }
                continue

            # Build eval target list: [scenario_target] + distractors
            eval_targets = [scenario_target] + distractors

            for eval_target in eval_targets:
                eval_counter += 1
                is_distractor = (eval_target != scenario_target)
                role = "distractor" if is_distractor else "target"

                # Determine video filename stem
                if is_distractor:
                    type_code = OBJECT_TYPE_CODES[eval_target]
                    stem = f"{scenario_id}{type_code}"
                else:
                    stem = scenario_id

                try:
                    # Build VLM prompt (same template as eval_bbox_all_scenarios.py line 225)
                    vlm_prompt = f"""
                        You are controlling a robot navigating toward a target object.

                        Detect the target object in the image and return ONLY valid JSON:
                        {{"bbox": [x1, y1, x2, y2], "label": "object name"}}

                        Rules:
                        - Coordinates are in 0-1000 scale relative to image dimensions
                        - Tightly bound the VISIBLE portion of the object
                        - Detect even if partially occluded or cut off by image edge
                        - If target not visible: {{"bbox": null, "label": null}}

                        Target: {eval_target}
                    """

                    # Reset VLM predictor state to prevent heading bleed between evaluations
                    vlm_predictor.previous_goal_heading = None

                    # Create environment with recorder_params for this evaluation.
                    # A new env is created per evaluation (not just per scenario) because
                    # recorder_params (including video_name) are baked in at construction time.
                    recorder_params = {
                        "path": args.output_dir,
                        "tag": ".",
                        "video_name": stem,
                    }
                    factory = TaskFactory.get_factory_cls(
                        config.experiment.task_factory.name
                    )
                    env = factory.make(
                        **config.experiment.env_params,
                        **config.experiment.task_factory.params,
                        default_camera_mode="ego_head",
                        viewer_size=(336, 336),
                        recorder_params=recorder_params,
                    )
                    env.info.horizon = args.n_steps

                    # Run navigation episode (VLM updates every vlm_update_freq steps)
                    rng = jax.random.key(args.seed)
                    PPOJaxCollectVLMRL.play_policy(
                        env, agent_conf, agent_state,
                        n_envs=1, n_steps=args.n_steps,
                        record=True, rng=rng,
                        deterministic=args.deterministic,
                        vlm_predictor=vlm_predictor,
                        vlm_update_frequency=args.vlm_update_freq,
                        vlm_prompt=vlm_prompt,
                    )

                    video_path = os.path.join(args.output_dir, f"{stem}.mp4")
                    results[stem] = {
                        "scenario_id": scenario_id,
                        "target": eval_target,
                        "is_distractor": is_distractor,
                        "video_path": video_path,
                        "status": "completed",
                        "steps_run": args.n_steps,
                    }
                    print(f"[{eval_counter}/{total_evals}] {scenario_id} -> {eval_target} ({role}): completed")

                except Exception as e:
                    print(f"[{eval_counter}/{total_evals}] {scenario_id} -> {eval_target} ({role}): FAILED ({e})")
                    results[stem] = {
                        "scenario_id": scenario_id,
                        "target": eval_target,
                        "is_distractor": is_distractor,
                        "status": "failed",
                        "error": str(e),
                    }

    finally:
        # Always restore go2.xml to its original state (even if script crashes)
        if original_include is not None:
            try:
                restore_scenario_include(go2_xml_path, original_include)
                print(f"\nRestored go2.xml to original include: {original_include}")
            except Exception as e:
                print(f"[CRITICAL] Failed to restore go2.xml: {e}")
                print(f"[CRITICAL] Manual restore needed. Original line was: {original_include}")

    # Count completed vs failed evaluations
    completed_count = sum(1 for v in results.values() if v.get("status") == "completed")
    failed_count = sum(1 for v in results.values() if v.get("status") == "failed")

    output = {
        "metadata": {
            "agent_path": args.agent_path,
            "vlm_model": "Qwen/Qwen3-VL-8B-Instruct",
            "timestamp": datetime.now().isoformat(timespec='seconds'),
            "n_steps": args.n_steps,
            "vlm_update_freq": args.vlm_update_freq,
            "seed": args.seed,
            "deterministic": args.deterministic,
            "total_scenarios": len(scenarios_to_run),
            "total_evaluations": eval_counter,
            "completed_count": completed_count,
            "failed_count": failed_count,
        },
        "results": results,
    }

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to: {results_path}")
    print(f"Total evaluations: {eval_counter}, Completed: {completed_count}, Failed: {failed_count}")


if __name__ == '__main__':
    main()
