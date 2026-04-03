#!/usr/bin/env python3
"""
VLM-based goal prediction for Go2 navigation.

Uses Qwen/Qwen3-VL-8B-Instruct to process environment frames
and dynamically determine heading/velocity goals for the agent.
"""
import argparse
import jax
import os
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJaxCollectVLMRL
from vlm_goal_predictor import create_vlm_predictor
from omegaconf import OmegaConf

# Set up JAX/XLA flags
os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'


def main():
    parser = argparse.ArgumentParser(description='Run Go2 with Qwen3-VL goal prediction')
    
    # Model paths
    parser.add_argument('--agent_path', type=str, required=True,
                        help='Path to trained Go2 agent pkl file')
    
    # VLM configuration
    parser.add_argument('--vlm_device', type=str, default='cuda',
                        help='Device for VLM model: "cuda", "cpu", or "cuda:0", "cuda:1", etc. (default: cuda)')
    parser.add_argument('--vlm_update_freq', type=int, default=10,
                        help='How often to update VLM prediction (in steps)')
    parser.add_argument('--vlm_prompt', type=str, default=None,
                        help='Optional custom text prompt for Qwen3-VL. If not provided, uses default prompt that requests JSON with red disk marker analysis.')
    parser.add_argument('--trust_remote_code', action='store_true', default=True,
                        help='Trust remote code when loading model (default: True)')
    parser.add_argument('--lora_adapter_path', type=str, default=None,
                        help='Path to LoRA adapter directory. If provided, loads finetuned LoRA weights '
                             'and uses the multi-object detection prompt.')
    parser.add_argument(
        '--predictor_type',
        type=str,
        default='huggingface',
        choices=['huggingface', 'twopass'],
        help='VLM predictor type to use: "huggingface" (default, single-pass) '
             'or "twopass" (two-pass with obstruction awareness).'
    )
    parser.add_argument(
        '--target_object',
        type=str,
        default='red disk marker',
        help='Name of the target object for the two-pass predictor (default: "red disk marker"). '
             'Used only when --predictor_type is "twopass".'
    )

    # Environment configuration
    parser.add_argument('--use_mujoco', action='store_true',
                        help='Use MuJoCo backend (default: Mjx)')
    parser.add_argument('--headless', action='store_true',
                        help='Run in headless mode')
    parser.add_argument('--n_steps', type=int, default=1000,
                        help='Number of steps to run')
    
    # Other options
    parser.add_argument('--deterministic', action='store_true',
                        help='Use deterministic policy')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--record', action='store_true',
                        help='Record frames and save as GIF')
    
    args = parser.parse_args()
    
    # Load agent
    print(f"Loading Go2 agent from: {args.agent_path}")
    agent_conf, agent_state = PPOJaxCollectVLMRL.load_agent(args.agent_path)
    config = agent_conf.config
    
    # Create Qwen3-VL predictor
    print(f"\nInitializing Qwen3-VL-8B-Instruct predictor on {args.vlm_device}...")
    if args.predictor_type == 'twopass' and args.lora_adapter_path is not None:
        print("WARNING: --lora_adapter_path is ignored when --predictor_type is 'twopass'.")
    if args.predictor_type == 'twopass':
        vlm_predictor = create_vlm_predictor(
            'twopass',
            device=args.vlm_device,
            trust_remote_code=args.trust_remote_code,
            target_object=args.target_object,
        )
    else:
        # 'huggingface' (default) — unchanged behavior
        vlm_predictor = create_vlm_predictor(
            'huggingface',
            device=args.vlm_device,
            trust_remote_code=args.trust_remote_code,
            lora_adapter_path=args.lora_adapter_path,
        )
    print(f"Predictor: {args.predictor_type} | Model: Qwen/Qwen3-VL-8B-Instruct (device: {args.vlm_device})")
    
    # Create environment
    print(f"\nSetting up environment...")
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    OmegaConf.set_struct(config, False)
    config.experiment.env_params["headless"] = args.headless
    env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params, default_camera_mode="ego_head", viewer_size=(336, 336))   
    
    # When using MJX, episode auto-terminates when step reaches env.info.horizon
    # Override the horizon value defined in the saved model to n_steps
    env.info.horizon = args.n_steps 
    # Print configuration
    print("\n" + "="*60)
    print("Qwen3-VL Guided Navigation Configuration")
    print("="*60)
    print(f"Agent: {args.agent_path}")
    print(f"VLM Model: Qwen/Qwen3-VL-8B-Instruct (device: {args.vlm_device})")
    print(f"Predictor Type: {args.predictor_type}")
    if args.predictor_type == 'twopass':
        print(f"Target Object: {args.target_object}")
    print(f"Update Frequency: Every {args.vlm_update_freq} steps")
    if args.lora_adapter_path:
        print(f"LoRA Adapter: {args.lora_adapter_path}")
    else:
        print("LoRA Adapter: None (using base model)")
    if args.vlm_prompt:
        print(f"Custom Prompt: {args.vlm_prompt[:80]}..." if len(args.vlm_prompt) > 80 else f"Custom Prompt: {args.vlm_prompt}")
    else:
        print("Using default prompt (requests JSON with red disk analysis)")
    print(f"Backend: {'MuJoCo' if args.use_mujoco else 'Mjx'}")
    print(f"Steps: {args.n_steps}")
    print(f"Mode: {'Headless' if args.headless else 'GUI'}")
    print("="*60 + "\n")
    
    # Set random seed
    rng = jax.random.key(args.seed)
    
    # Run policy with Qwen3-VL predictor
    print("Starting Qwen3-VL guided navigation...\n")
    if args.use_mujoco:
        PPOJaxCollectVLMRL.play_policy_mujoco(
            env, agent_conf, agent_state,
            n_steps=args.n_steps,
            deterministic=args.deterministic,
            record=args.record,
            rng=rng,
            vlm_predictor=vlm_predictor,
            vlm_update_frequency=args.vlm_update_freq,
            vlm_prompt=args.vlm_prompt
        )
    else:
        PPOJaxCollectVLMRL.play_policy(
            env, agent_conf, agent_state,
            n_envs=1,
            n_steps=args.n_steps,
            deterministic=args.deterministic,
            record=args.record,
            rng=rng,
            vlm_predictor=vlm_predictor,
            vlm_update_frequency=args.vlm_update_freq,
            vlm_prompt=args.vlm_prompt
        )
    
    print("\nQwen3-VL guided navigation complete!")


if __name__ == '__main__':
    main()
