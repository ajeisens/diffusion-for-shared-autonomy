#!/usr/bin/env python3
"""
Analyze recorded episodes to understand issues:
1. Teleop crashes - episodes where user thought they landed but crashed
2. Heuristic spinning - examine angular velocity patterns
"""

import torch
import numpy as np
from pathlib import Path
import json

def analyze_episodes():
    # Find all episode files
    episode_dir = Path("./demo/interactive/recorded_episodes")
    if not episode_dir.exists():
        print(f"Episode directory not found: {episode_dir}")
        return

    episode_files = list(episode_dir.glob("episode_*.pkl"))
    print(f"Found {len(episode_files)} episode files")

    # Load metadata
    metadata_file = episode_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
            print("\n=== Overall Statistics ===")
            print(json.dumps(metadata['statistics'], indent=2))

    # Analyze teleop episodes
    print("\n\n=== TELEOP ANALYSIS ===")
    teleop_files = [f for f in episode_files if "_teleop.pkl" in str(f)]
    print(f"Found {len(teleop_files)} teleop episodes")

    # Find crashes
    crashed_episodes = []
    collision_episodes = []
    success_episodes = []

    for ep_file in teleop_files:
        try:
            ep = torch.load(ep_file, weights_only=False)
            meta = ep['metadata']

            if meta.get('crashed', False):
                crashed_episodes.append((ep_file.name, ep))
            if meta.get('collision', False):
                collision_episodes.append((ep_file.name, ep))
            if meta.get('success', False):
                success_episodes.append((ep_file.name, ep))
        except Exception as e:
            print(f"Error loading {ep_file}: {e}")

    print(f"\nTeleop results:")
    print(f"  Success: {len(success_episodes)}")
    print(f"  Crashed (terrain/OOB): {len(crashed_episodes)}")
    print(f"  Collision (obstacle): {len(collision_episodes)}")

    # Show some crashed episodes
    if crashed_episodes:
        print("\n--- Sample Crashed Teleop Episodes ---")
        for name, ep in crashed_episodes[:5]:
            meta = ep['metadata']
            obs = ep['observations']
            # Final state: [x, y, θ, vx, vy, ω, ...]
            final_state = obs[-1][:6]
            print(f"\n{name}:")
            print(f"  Return: {meta['episode_return']:.1f}")
            print(f"  Length: {meta['episode_length']} steps")
            print(f"  Final state: x={final_state[0]:.2f}, y={final_state[1]:.2f}, θ={final_state[2]:.2f}")
            print(f"  Final velocity: vx={final_state[3]:.2f}, vy={final_state[4]:.2f}, ω={final_state[5]:.2f}")

            # Check last info
            if ep['infos']:
                final_info = ep['infos'][-1]
                print(f"  Goal status: {final_info.get('goal', 'unknown')}")

    # Analyze heuristic episodes
    print("\n\n=== HEURISTIC ANALYSIS ===")
    heuristic_files = [f for f in episode_files if "_heuristic.pkl" in str(f)]
    print(f"Found {len(heuristic_files)} heuristic episodes")

    if heuristic_files:
        print("\n--- Heuristic Angular Velocity Analysis ---")
        for i, ep_file in enumerate(heuristic_files[:10]):  # Check first 10
            try:
                ep = torch.load(ep_file, weights_only=False)
                obs = ep['observations']
                # Angular velocity is at index 5
                angular_vels = obs[:, 5]

                max_abs_omega = np.max(np.abs(angular_vels))
                mean_abs_omega = np.mean(np.abs(angular_vels))

                print(f"\n{ep_file.name}:")
                print(f"  Max |omega|: {max_abs_omega:.3f} rad/s")
                print(f"  Mean |omega|: {mean_abs_omega:.3f} rad/s")
                print(f"  Success: {ep['metadata'].get('success', False)}")
                print(f"  Crashed: {ep['metadata'].get('crashed', False)}")
                print(f"  Collision: {ep['metadata'].get('collision', False)}")

                # Check if spinning is excessive (> 1 rad/s is pretty fast)
                if max_abs_omega > 1.0:
                    print(f"  WARNING: High angular velocity detected!")
                    # Show when it happened
                    high_omega_steps = np.where(np.abs(angular_vels) > 1.0)[0]
                    print(f"  High omega at steps: {high_omega_steps[:10].tolist()}...")

            except Exception as e:
                print(f"Error loading {ep_file}: {e}")

if __name__ == '__main__':
    analyze_episodes()
