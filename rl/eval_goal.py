import os
import argparse
import time
import jax
import numpy as np

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJaxCollectVLMRL


from omegaconf import OmegaConf

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True ')

# Set up argument parser
parser = argparse.ArgumentParser(description='Run inference with trained Go2 agent.')
parser.add_argument('--path', type=str, required=True, help='Path to the agent pkl file')
parser.add_argument('--use_mujoco', action='store_true', help='Use MuJoCo for evaluation instead of Mjx')
parser.add_argument('--deterministic', action='store_true', help='Use deterministic policy (no exploration)')
parser.add_argument('--n_steps', type=int, default=10000, help='Number of steps to run')
parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
parser.add_argument('--train_state_seed', type=int, default=0, help='Which training seed to use for evaluation')
parser.add_argument('--record', action='store_true', help='Record video of the episode')
parser.add_argument('--headless', action='store_true', help='Run in headless mode (no GUI, required for SLURM)')

# Data saving parameters
parser.add_argument('--save_data', action='store_true', help='Save episode data to files')
parser.add_argument('--save_dir', type=str, default=None, help='Directory to save training episodes (required if --save_data is used)')
parser.add_argument('--starting_episode', type=int, default=0, help='Starting episode number')
parser.add_argument('--num_episodes', type=int, default=1, help='Number of episodes to run')
parser.add_argument('--episode_length', type=int, default=100, help='Length of each episode')
parser.add_argument('--slice_obs', type=int, default=None, help='Indices of observation to save')
parser.add_argument('--save_frames_as_video', action='store_true', help='Whether to save frames as video')

# Custom goal parameters
parser.add_argument('--vel_x', type=float, default=None, help='Custom x velocity goal (m/s)')
parser.add_argument('--vel_y', type=float, default=None, help='Custom y velocity goal (m/s)')
parser.add_argument('--heading', type=float, default=None, help='Custom heading goal (radians)')
args = parser.parse_args()

# Validate arguments
if args.save_data and args.save_dir is None:
    parser.error('--save_dir is required when --save_data is used')

# Load the trained agent
print(f"Loading agent from: {args.path}")
agent_conf, agent_state = PPOJaxCollectVLMRL.load_agent(args.path)
config = agent_conf.config

# Display environment configuration
print("\n" + "="*60)
print("Environment Configuration:")
print("="*60)
print(f"Environment: {config.experiment.env_params.env_name}")
print(f"Goal Type: {config.experiment.env_params.goal_type}")
print(f"Reward Type: {config.experiment.env_params.reward_type}")
if hasattr(config.experiment.env_params, 'goal_params'):
    print("\nGoal Parameters:")
    for key, value in config.experiment.env_params.goal_params.items():
        print(f"  {key}: {value}")
print("="*60 + "\n")

# Get task factory
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

# Create environment with visualization enabled/disabled
OmegaConf.set_struct(config, False)  # Allow modifications
config.experiment.env_params["headless"] = args.headless  # Control visualization (True for SLURM)
env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params, default_camera_mode="ego_head")

if args.save_data:
    print(f"Saving data mode: Running {args.num_episodes} episodes of {args.episode_length} steps each")
    print(f"Save directory: {args.save_dir}")
else:
    print(f"Running inference for {args.n_steps} steps...")

print(f"Using {'deterministic' if args.deterministic else 'stochastic'} policy")
print(f"Backend: {'MuJoCo' if args.use_mujoco else 'Mjx'}")
print(f"Mode: {'Headless' if args.headless else 'GUI'}")
print(f"Recording: {'Yes' if args.record or args.save_frames_as_video else 'No'}")

# Setup custom goal if parameters are provided
custom_goal = None
if args.vel_x is not None or args.vel_y is not None or args.heading is not None:
    custom_goal = {
        'vel_x': args.vel_x if args.vel_x is not None else 0.0,
        'vel_y': args.vel_y if args.vel_y is not None else 0.0,
        'heading': args.heading if args.heading is not None else 0.0
    }
    print(f"\nCustom Goal:")
    print(f"  vel_x: {custom_goal['vel_x']:.3f} m/s")
    print(f"  vel_y: {custom_goal['vel_y']:.3f} m/s")
    print(f"  heading: {custom_goal['heading']:.3f} rad ({custom_goal['heading']*180/3.14159:.1f}°)")
else:
    print(f"\nUsing random goals from environment")

# Original single-run inference mode
# Set random seed for reproducibility
rng = jax.random.key(args.seed)

# Run policy
if args.use_mujoco:
    # Run with MuJoCo backend (slower but more compatible)
    PPOJaxCollectVLMRL.play_policy_mujoco(env, agent_conf, agent_state, 
                                deterministic=args.deterministic, 
                                n_steps=args.n_steps, 
                                record=args.record,
                                train_state_seed=args.train_state_seed,
                                rng=rng,
                                custom_goal=custom_goal)
else:
    # Run with Mjx backend (faster, GPU-accelerated)
    PPOJaxCollectVLMRL.play_policy(env, agent_conf, agent_state, 
                        deterministic=args.deterministic, 
                        n_steps=args.n_steps, 
                        n_envs=1, 
                        record=args.record,
                        train_state_seed=args.train_state_seed,
                        rng=rng,
                        custom_goal=custom_goal)

print("\nInference complete!")
