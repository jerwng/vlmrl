#!/usr/bin/env python3
"""
Batch VLM+RL navigation evaluation across all 60 Go2 evaluation scenarios.

Discovers scenarios from the scenarios/ directory, parses each scenario.md for the
target body name, and passes it to PPOJaxCollectVLMRL.play_policy for accurate
success/failure evaluation.

Produces MP4 videos + results.json + a final success/failure summary log.
"""
import argparse
import json
import os
import re
import sys
import jax
from datetime import datetime
from omegaconf import OmegaConf
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJaxCollectVLMRL
from vlm_goal_predictor import create_vlm_predictor

# Add vlmrl/ parent to sys.path to reach vlmrl/utils/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.update_scenario import create_scenario_xml_copy, remove_scenario_xml_copy

os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'

SCENARIOS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '..', '..', 'loco-mujoco',
    'loco_mujoco', 'models', 'unitree_go2', 'scenarios',
))

# Canonical object type ordering for deterministic iteration
OBJECT_TYPE_ORDER = ["chair", "couch", "bookshelf", "table", "cardboard_box", "trash_can"]


def _parse_scenario_md(scenario_dir: str) -> dict:
    """Parse scenario.md and return a dict with target_type and target_body_name."""
    md_path = os.path.join(scenario_dir, "scenario.md")
    with open(md_path, "r") as f:
        content = f.read()

    target_type_match = re.search(r"- Target type:\s*`([^`]+)`", content)
    target_body_match = re.search(r"- Target body name:\s*`([^`]+)`", content)
    difficulty_match = re.search(r"- Difficulty:\s*`([^`]+)`", content)
    distance_match = re.search(r"- Distance bucket:\s*`([^`]+)`", content)

    if target_type_match is None or target_body_match is None:
        raise ValueError(f"Could not parse target info from {md_path}")

    return {
        "target_type": target_type_match.group(1),
        "target_body_name": target_body_match.group(1),
        "difficulty": difficulty_match.group(1) if difficulty_match else "unknown",
        "distance": distance_match.group(1) if distance_match else "unknown",
    }


def _discover_scenarios(scenarios_dir: str) -> list[str]:
    """Return sorted list of scenario directory names (e.g. bookshelf_01)."""
    entries = []
    for name in os.listdir(scenarios_dir):
        if name == "README.md":
            continue
        if os.path.isdir(os.path.join(scenarios_dir, name)):
            entries.append(name)

    def sort_key(name):
        # Sort by object type canonical order, then by number
        for i, obj_type in enumerate(OBJECT_TYPE_ORDER):
            if name.startswith(obj_type):
                num = int(name[len(obj_type) + 1:])
                return (i, num)
        return (len(OBJECT_TYPE_ORDER), name)

    entries.sort(key=sort_key)
    return entries


def _print_final_summary(results: dict) -> None:
    """Print a structured final success/failure summary log."""
    successes = {k: v for k, v in results.items() if v.get("outcome") == "SUCCESS"}
    partials  = {k: v for k, v in results.items() if v.get("outcome") == "PARTIAL"}
    failures  = {k: v for k, v in results.items() if v.get("outcome") == "FAILURE"}
    errors    = {k: v for k, v in results.items() if v.get("outcome") == "ERROR"}

    total = len(results)
    print("\n" + "=" * 70)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total scenarios : {total}")
    print(f"  SUCCESS       : {len(successes)}")
    print(f"  PARTIAL       : {len(partials)}")
    print(f"  FAILURE       : {len(failures)}")
    print(f"  ERROR         : {len(errors)}")
    if total > 0:
        print(f"  Success rate  : {len(successes) / total * 100:.1f}%")
        print(f"  Partial rate  : {len(partials) / total * 100:.1f}%")
    completed = {k: v for k, v in results.items() if "vlm_time_total_s" in v}
    if completed:
        total_vlm = sum(v["vlm_time_total_s"] for v in completed.values())
        total_calls = sum(v["vlm_call_count"] for v in completed.values())
        avg_per_call = total_vlm / total_calls if total_calls > 0 else 0.0
        print(f"  VLM time      : {total_vlm:.1f}s total, {total_calls} calls, {avg_per_call:.2f}s/call avg")
    print()

    if successes:
        print(f"--- SUCCESSES ({len(successes)}) ---")
        for sid, info in sorted(successes.items()):
            dist_str = f"dist={info['dist']:.3f}m" if info.get("dist") is not None else "dist=N/A"
            vlm_str = f"vlm={info['vlm_time_total_s']:.1f}s/{info['vlm_call_count']}calls" if "vlm_time_total_s" in info else ""
            print(f"  {sid:<20}  {info['target_type']:<14}  {info['difficulty']:<8}  "
                  f"{dist_str}  steps={info['steps']}  {vlm_str}  reason: {info['reason']}")
        print()

    if partials:
        print(f"--- PARTIALS ({len(partials)}) ---")
        for sid, info in sorted(partials.items()):
            dist_str = f"dist={info['dist']:.3f}m" if info.get("dist") is not None else "dist=N/A"
            vlm_str = f"vlm={info['vlm_time_total_s']:.1f}s/{info['vlm_call_count']}calls" if "vlm_time_total_s" in info else ""
            print(f"  {sid:<20}  {info['target_type']:<14}  {info['difficulty']:<8}  "
                  f"{dist_str}  steps={info['steps']}  {vlm_str}  reason: {info['reason']}")
        print()

    if failures:
        print(f"--- FAILURES ({len(failures)}) ---")
        for sid, info in sorted(failures.items()):
            dist_str = f"dist={info['dist']:.3f}m" if info.get("dist") is not None else "dist=N/A"
            vlm_str = f"vlm={info['vlm_time_total_s']:.1f}s/{info['vlm_call_count']}calls" if "vlm_time_total_s" in info else ""
            print(f"  {sid:<20}  {info['target_type']:<14}  {info['difficulty']:<8}  "
                  f"{dist_str}  {vlm_str}  reason: {info['reason']}")
        print()

    if errors:
        print(f"--- ERRORS ({len(errors)}) ---")
        for sid, info in sorted(errors.items()):
            print(f"  {sid:<20}  reason: {info['reason']}")
        print()

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Batch VLM+RL navigation evaluation across all 60 Go2 scenarios'
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
                        help='Comma-separated subset of scenario dir names, e.g. chair_01,couch_03')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use deterministic policy (no action sampling noise)')
    parser.add_argument('--success_distance', type=float, default=1.0,
                        help='Distance threshold (m) for success evaluation (default: 1.0)')
    parser.add_argument('--success_velocity', type=float, default=0.1,
                        help='Velocity threshold (m/s) for success evaluation (default: 0.1)')
    parser.add_argument('--low_velocity_threshold', type=float, default=0.1,
                        help='Terminate when average planar speed over the window drops below this value (default: 0.1)')
    parser.add_argument('--low_velocity_window', type=int, default=100,
                        help='Window size in steps for low-velocity termination; set to 0 to disable (default: 100)')
    args = parser.parse_args()

    # Discover all scenario directories
    all_scenarios = _discover_scenarios(SCENARIOS_DIR)

    if args.scenarios is not None:
        requested = [s.strip() for s in args.scenarios.split(',')]
        invalid = [s for s in requested if s not in all_scenarios]
        if invalid:
            raise ValueError(f"Unknown scenario dirs: {invalid}. "
                             f"Available: {all_scenarios}")
        scenarios_to_run = [s for s in all_scenarios if s in requested]
    else:
        scenarios_to_run = all_scenarios

    os.makedirs(args.output_dir, exist_ok=True)

    go2_xml_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'loco-mujoco',
        'loco_mujoco', 'models', 'unitree_go2', 'go2.xml'
    ))
    if not os.path.exists(go2_xml_path):
        raise FileNotFoundError(f"go2.xml not found: {go2_xml_path}")

    print(f"Loading agent config from: {args.agent_path}")
    agent_conf, agent_state = PPOJaxCollectVLMRL.load_agent(args.agent_path)
    config = agent_conf.config
    OmegaConf.set_struct(config, False)
    config.experiment.env_params["headless"] = True

    print(f"Loading VLM on {args.vlm_device}...")
    vlm_predictor = create_vlm_predictor('twopass', device=args.vlm_device)
    print("VLM loaded.")

    run_suffix = str(os.getpid())
    results = {}
    eval_counter = 0
    total = len(scenarios_to_run)

    print(f"Scenarios to run: {total}")
    print(f"Output dir      : {args.output_dir}")
    print("-" * 60)

    for scenario_dir in scenarios_to_run:
        eval_counter += 1

        # Parse scenario metadata
        try:
            meta = _parse_scenario_md(os.path.join(SCENARIOS_DIR, scenario_dir))
        except Exception as e:
            print(f"[{eval_counter}/{total}] {scenario_dir}: FAILED to parse scenario.md ({e})")
            results[scenario_dir] = {
                "scenario": scenario_dir,
                "outcome": "ERROR",
                "reason": f"scenario.md parse error: {e}",
                "target_type": "unknown",
                "target_body_name": "unknown",
                "difficulty": "unknown",
            }
            continue

        target_type = meta["target_type"]
        target_body_name = meta["target_body_name"]
        difficulty = meta["difficulty"]
        distance = meta["distance"]

        # Create a per-job XML copy pointing at this scenario (never mutates go2.xml)
        try:
            copy_path = create_scenario_xml_copy(go2_xml_path, scenario_dir, suffix=run_suffix)
        except Exception as e:
            print(f"[{eval_counter}/{total}] {scenario_dir}: FAILED to create XML copy ({e})")
            results[scenario_dir] = {
                "scenario": scenario_dir,
                "outcome": "ERROR",
                "reason": f"XML copy failed: {e}",
                "target_type": target_type,
                "target_body_name": target_body_name,
                "difficulty": difficulty,
                "distance": distance,
            }
            continue

        # Update two-pass predictor target for this scenario
        # Replace underscores with spaces (e.g. cardboard_box -> cardboard box)
        vlm_predictor.target_object = target_type.replace("_", " ")

        # Reset VLM predictor state to prevent heading bleed between evaluations
        vlm_predictor.previous_goal_heading = None
        vlm_predictor.vlm_time_total = 0.0
        vlm_predictor.vlm_call_count = 0

        try:
            recorder_params = {
                "path": args.output_dir,
                "tag": ".",
                "video_name": scenario_dir,
            }
            factory = TaskFactory.get_factory_cls(
                config.experiment.task_factory.name
            )
            env_params = dict(config.experiment.env_params)
            env_params.pop("spec", None)  # never pass a stale spec
            env = factory.make(
                spec=copy_path,
                **env_params,
                **config.experiment.task_factory.params,
                default_camera_mode="ego_head",
                viewer_size=(336, 336),
                recorder_params=recorder_params,
            )
            env.info.horizon = args.n_steps

            rng = jax.random.key(args.seed)
            episode_result = PPOJaxCollectVLMRL.play_policy(
                env, agent_conf, agent_state,
                n_envs=1, n_steps=args.n_steps,
                record=True, rng=rng,
                deterministic=args.deterministic,
                vlm_predictor=vlm_predictor,
                vlm_update_frequency=args.vlm_update_freq,
                target_body_name=target_body_name,
                success_distance_threshold=args.success_distance,
                success_velocity_threshold=args.success_velocity,
                low_velocity_threshold=args.low_velocity_threshold if args.low_velocity_window > 0 else None,
                low_velocity_window=args.low_velocity_window,
            )

            vlm_time = vlm_predictor.vlm_time_total
            vlm_calls = vlm_predictor.vlm_call_count
            results[scenario_dir] = {
                "scenario": scenario_dir,
                "target_type": target_type,
                "target_body_name": target_body_name,
                "difficulty": difficulty,
                "distance": distance,
                "video_path": os.path.join(args.output_dir, f"{scenario_dir}.mp4"),
                "outcome": episode_result["outcome"],
                "reason": episode_result["reason"],
                "steps": episode_result["steps"],
                "speed": episode_result["speed"],
                "dist": episode_result["dist"],
                "collision": episode_result["collision"],
                "target_collision": episode_result["target_collision"],
                "vlm_time_total_s": round(vlm_time, 2),
                "vlm_call_count": vlm_calls,
                "vlm_time_per_call_s": round(vlm_time / vlm_calls, 2) if vlm_calls > 0 else None,
            }
            outcome = episode_result["outcome"]
            print(f"[{eval_counter}/{total}] {scenario_dir:<22}  {target_type:<14}  "
                  f"{difficulty:<8}  -> {outcome}  "
                  f"vlm={vlm_time:.1f}s/{vlm_calls}calls  ({episode_result['reason']})")

        except Exception as e:
            print(f"[{eval_counter}/{total}] {scenario_dir}: FAILED during rollout ({e})")
            results[scenario_dir] = {
                "scenario": scenario_dir,
                "target_type": target_type,
                "target_body_name": target_body_name,
                "difficulty": difficulty,
                "distance": distance,
                "outcome": "ERROR",
                "reason": str(e),
            }
        finally:
            remove_scenario_xml_copy(copy_path)

    # Belt-and-suspenders: remove any leftover copy from this job
    leftover = os.path.join(os.path.dirname(go2_xml_path), f"go2_{run_suffix}.xml")
    if os.path.exists(leftover):
        os.remove(leftover)

    # Aggregate counts
    outcome_counts = {}
    for v in results.values():
        oc = v.get("outcome", "ERROR")
        outcome_counts[oc] = outcome_counts.get(oc, 0) + 1

    output = {
        "metadata": {
            "agent_path": args.agent_path,
            "vlm_model": "Qwen/Qwen3-VL-8B-Instruct",
            "timestamp": datetime.now().isoformat(timespec='seconds'),
            "n_steps": args.n_steps,
            "vlm_update_freq": args.vlm_update_freq,
            "seed": args.seed,
            "deterministic": args.deterministic,
            "success_distance_threshold": args.success_distance,
            "success_velocity_threshold": args.success_velocity,
            "low_velocity_threshold": args.low_velocity_threshold,
            "low_velocity_window": args.low_velocity_window,
            "total_scenarios": eval_counter,
            "outcome_counts": outcome_counts,
        },
        "results": results,
    }

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to: {results_path}")

    _print_final_summary(results)


if __name__ == '__main__':
    main()
