#!/usr/bin/env python3
"""
Unified evaluation harness for 4-condition ablation study.

Compares:
1. Pure teleop (noisy pilot, no assistance)
2. MPC baseline (CEM planner with obstacle-blind dynamics)
3. Conditioned diffusion (with intent labels + MPC proposals) - TODO
4. DiffuSHA (standard shared autonomy with partial diffusion)

Tracks comprehensive metrics across multiple seeds and episodes.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import gym
import torch

# Import actors and environment
from diffusha.actor.base import Actor, NoisyActor, ExpertActor
from diffusha.data_collection.env import make_env


class MetricsTracker:
    """Track comprehensive metrics for each episode"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset for new episode"""
        self.trajectory = []
        self.episode_return = 0.0
        self.collision = False
        self.crashed = False
        self.timeout = False
        self.success = False
        self.min_clearances = []
        self.steps = 0

    def step(self, obs, action, reward, done, info):
        """Update metrics for one step"""
        self.trajectory.append({
            'obs': obs.copy() if isinstance(obs, np.ndarray) else obs,
            'action': action.copy() if isinstance(action, np.ndarray) else action,
            'reward': reward,
        })
        self.episode_return += reward
        self.steps += 1

        # Track collision
        if info.get('collision', False):
            self.collision = True

        # Track crash (hit terrain, not goal)
        if info.get('crashed', False):
            self.crashed = True

        # Track success (landed at goal)
        if info.get('goal', '') in ['landed', 'target-reached']:
            self.success = True

        # Extract lidar readings for clearance (last 8 dims if available)
        if isinstance(obs, np.ndarray) and len(obs) >= 17:
            lidar = obs[-8:]  # Last 8 are lidar readings
            # Lidar values in [0,1], where 0=close, 1=far
            # Convert to actual distance (unnormalized)
            # For now, just track minimum normalized reading
            min_reading = np.min(lidar)
            self.min_clearances.append(min_reading)

    def get_summary(self):
        """Get episode summary"""
        # Determine outcome priority: collision > crash > timeout > success
        if self.collision:
            outcome = 'collision'
        elif self.crashed:
            outcome = 'crash'
        elif self.steps >= 1000:  # Timeout
            outcome = 'timeout'
            self.timeout = True
        elif self.success:
            outcome = 'success'
        else:
            outcome = 'other'

        return {
            'outcome': outcome,
            'success': self.success,
            'collision': self.collision,
            'crashed': self.crashed,
            'timeout': self.timeout,
            'episode_return': float(self.episode_return),
            'steps': int(self.steps),
            'mean_clearance': float(np.mean(self.min_clearances)) if self.min_clearances else 1.0,
            'min_clearance': float(np.min(self.min_clearances)) if self.min_clearances else 1.0,
        }


def evaluate_condition(
    condition_name: str,
    actor: Actor,
    env_name: str,
    n_episodes: int,
    seed: int,
    save_trajectories: bool = False
) -> Dict[str, Any]:
    """
    Evaluate one condition for n_episodes.

    Returns:
        Dictionary with episode-level results
    """
    # Create environment
    env = gym.make(env_name)
    env.seed(seed)

    # Results storage
    episodes = []

    for ep in range(n_episodes):
        tracker = MetricsTracker()
        obs = env.reset()
        done = False

        # Update MPC goal if applicable
        if hasattr(actor, 'set_goal') and hasattr(env.unwrapped, 'helipad_x1'):
            # Compute normalized helipad position
            helipad_x = (env.unwrapped.helipad_x1 + env.unwrapped.helipad_x2) / 2
            from diffusha.data_collection.env.lunar_lander import VIEWPORT_W, SCALE
            W = VIEWPORT_W / SCALE
            helipad_x_norm = (helipad_x - W/2) / (W/2)
            actor.set_goal(helipad_x_norm, 0.0)

        while not done:
            action = actor.act(obs)
            obs, reward, done, info = env.step(action)
            tracker.step(obs, action, reward, done, info)

            if tracker.steps >= 1000:  # Max episode length
                break

        # Get episode summary
        ep_summary = tracker.get_summary()

        if save_trajectories:
            ep_summary['trajectory'] = tracker.trajectory

        episodes.append(ep_summary)

    env.close()

    # Aggregate statistics
    outcomes = [ep['outcome'] for ep in episodes]
    success_rate = sum(1 for o in outcomes if o == 'success') / n_episodes
    collision_rate = sum(1 for o in outcomes if o == 'collision') / n_episodes
    crash_rate = sum(1 for o in outcomes if o == 'crash') / n_episodes
    timeout_rate = sum(1 for o in outcomes if o == 'timeout') / n_episodes

    mean_return = np.mean([ep['episode_return'] for ep in episodes])
    std_return = np.std([ep['episode_return'] for ep in episodes])

    mean_clearance = np.mean([ep['mean_clearance'] for ep in episodes])

    return {
        'condition': condition_name,
        'seed': seed,
        'n_episodes': n_episodes,
        'episodes': episodes if save_trajectories else None,
        'metrics': {
            'success_rate': float(success_rate),
            'collision_rate': float(collision_rate),
            'crash_rate': float(crash_rate),
            'timeout_rate': float(timeout_rate),
            'mean_return': float(mean_return),
            'std_return': float(std_return),
            'mean_clearance': float(mean_clearance),
        }
    }


def create_teleop_actor(env, pilot_noise=0.3, seed=0):
    """Create pure teleop actor (noisy expert)"""
    # For now, use random actor as surrogate pilot
    # In full implementation, would load SAC expert
    from diffusha.actor.base import RandomActor
    return RandomActor(env.observation_space, env.action_space, seed=seed)


def create_mpc_actor(env, model_path, horizon=10, n_samples=200, device='cuda'):
    """Create MPC actor"""
    from diffusha.baseline.mpc_cem import MPCAgent
    return MPCAgent(
        obs_space=env.observation_space,
        act_space=env.action_space,
        dynamics_model_path=model_path,
        horizon=horizon,
        n_samples=n_samples,
        device=device
    )


def create_diffusha_actor(env, diffusion_path, pilot_noise=0.3, gamma=0.4, seed=0, device='cuda'):
    """Create DiffuSHA actor (diffusion-assisted)"""
    # TODO: Implement when diffusion model is available
    # For now, return None
    return None


def run_ablation_study(
    env_name: str,
    output_dir: Path,
    n_seeds: int = 3,
    n_episodes: int = 10,
    pilot_noise: float = 0.3,
    mpc_model_path: Optional[str] = None,
    diffusha_model_path: Optional[str] = None,
    save_trajectories: bool = False,
    conditions: Optional[List[str]] = None,
    device: str = 'cuda'
):
    """
    Run full ablation study across multiple seeds.

    Args:
        env_name: Environment ID
        output_dir: Where to save results
        n_seeds: Number of random seeds
        n_episodes: Episodes per seed per condition
        pilot_noise: Noise level for surrogate pilot
        mpc_model_path: Path to MPC dynamics model (None = skip MPC)
        diffusha_model_path: Path to diffusion model (None = skip DiffuSHA)
        save_trajectories: Whether to save full trajectories
        conditions: Which conditions to run (None = all available)
        device: 'cuda' or 'cpu'
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which conditions to run
    available_conditions = {
        'teleop': True,
        'mpc': mpc_model_path is not None,
        'diffusha': diffusha_model_path is not None,
    }

    if conditions is None:
        conditions_to_run = [c for c, available in available_conditions.items() if available]
    else:
        conditions_to_run = [c for c in conditions if available_conditions.get(c, False)]

    print(f"Running ablation study:")
    print(f"  Environment: {env_name}")
    print(f"  Seeds: {n_seeds}")
    print(f"  Episodes per seed: {n_episodes}")
    print(f"  Conditions: {conditions_to_run}")
    print(f"  Output: {output_dir}")

    # Results storage
    all_results = []

    # Create sample environment for setup
    sample_env = gym.make(env_name)

    # Run each seed
    for seed in range(n_seeds):
        print(f"\n=== Seed {seed}/{n_seeds} ===")

        for condition in conditions_to_run:
            print(f"\nRunning condition: {condition}")

            # Create actor for this condition
            if condition == 'teleop':
                actor = create_teleop_actor(sample_env, pilot_noise, seed)

            elif condition == 'mpc':
                actor = create_mpc_actor(sample_env, mpc_model_path, device=device)

            elif condition == 'diffusha':
                actor = create_diffusha_actor(
                    sample_env, diffusha_model_path,
                    pilot_noise, gamma=0.4, seed=seed, device=device
                )

            if actor is None:
                print(f"  Skipping {condition} (not available)")
                continue

            # Evaluate
            result = evaluate_condition(
                condition_name=condition,
                actor=actor,
                env_name=env_name,
                n_episodes=n_episodes,
                seed=seed,
                save_trajectories=save_trajectories
            )

            all_results.append(result)

            # Print summary
            m = result['metrics']
            print(f"  Success: {m['success_rate']:.1%}, "
                  f"Collision: {m['collision_rate']:.1%}, "
                  f"Crash: {m['crash_rate']:.1%}, "
                  f"Return: {m['mean_return']:.1f}")

    sample_env.close()

    # Save results
    output_file = output_dir / 'ablation_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Compute aggregated statistics
    print_summary_statistics(all_results)

    return all_results


def print_summary_statistics(results: List[Dict]):
    """Print aggregated statistics across seeds"""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS (across all seeds)")
    print("=" * 60)

    # Group by condition
    by_condition = defaultdict(list)
    for r in results:
        by_condition[r['condition']].append(r['metrics'])

    # Print table
    print(f"\n{'Condition':<15} {'Success':>10} {'Collision':>12} {'Crash':>10} {'Return':>12}")
    print("-" * 60)

    for condition, metrics_list in sorted(by_condition.items()):
        success_rates = [m['success_rate'] for m in metrics_list]
        collision_rates = [m['collision_rate'] for m in metrics_list]
        crash_rates = [m['crash_rate'] for m in metrics_list]
        returns = [m['mean_return'] for m in metrics_list]

        print(f"{condition:<15} "
              f"{np.mean(success_rates):>9.1%} "
              f"{np.mean(collision_rates):>11.1%} "
              f"{np.mean(crash_rates):>9.1%} "
              f"{np.mean(returns):>11.1f}")


def main():
    parser = argparse.ArgumentParser(description='Run ablation study')
    parser.add_argument('--env-name', type=str, default='LunarLanderObstacle-v5',
                        help='Environment name')
    parser.add_argument('--n-seeds', type=int, default=3,
                        help='Number of random seeds')
    parser.add_argument('--n-episodes', type=int, default=10,
                        help='Episodes per seed per condition')
    parser.add_argument('--pilot-noise', type=float, default=0.3,
                        help='Noise level for surrogate pilot')
    parser.add_argument('--mpc-model', type=str, default=None,
                        help='Path to MPC dynamics model (optional)')
    parser.add_argument('--diffusha-model', type=str, default=None,
                        help='Path to DiffuSHA model (optional)')
    parser.add_argument('--out-dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--save-trajectories', action='store_true',
                        help='Save full trajectories (large files)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--conditions', type=str, nargs='+', default=None,
                        help='Specific conditions to run (default: all available)')

    args = parser.parse_args()

    # Import diffusha to trigger environment registration
    import diffusha

    run_ablation_study(
        env_name=args.env_name,
        output_dir=Path(args.out_dir),
        n_seeds=args.n_seeds,
        n_episodes=args.n_episodes,
        pilot_noise=args.pilot_noise,
        mpc_model_path=args.mpc_model,
        diffusha_model_path=args.diffusha_model,
        save_trajectories=args.save_trajectories,
        conditions=args.conditions,
        device=args.device
    )


if __name__ == "__main__":
    main()
