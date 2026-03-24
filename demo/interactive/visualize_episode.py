#!/usr/bin/env python3
"""
Visualize a single recorded episode to understand what's happening.
"""

import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

def visualize_episode(filename):
    """Visualize an episode"""
    ep = torch.load(filename, weights_only=False)

    obs = ep['observations']
    actions = ep['actions']
    rewards = ep['rewards']
    meta = ep['metadata']

    print(f"=== Episode: {filename} ===")
    print(f"Mode: {meta['mode']}")
    print(f"Success: {meta.get('success', False)}")
    print(f"Collision: {meta.get('collision', False)}")
    print(f"Crashed: {meta.get('crashed', False)}")
    print(f"Return: {meta['episode_return']:.2f}")
    print(f"Length: {meta['episode_length']} steps")

    # Extract state components
    x = obs[:, 0]
    y = obs[:, 1]
    theta = obs[:, 2]
    vx = obs[:, 3]
    vy = obs[:, 4]
    omega = obs[:, 5]

    # Actions
    Fm = actions[:, 0]  # Main engine [0,1]
    Fs = actions[:, 1]  # Side engine [-1,1]

    # Final state
    print(f"\nFinal state:")
    print(f"  Position: ({x[-1]:.2f}, {y[-1]:.2f})")
    print(f"  Angle: {theta[-1]:.2f} rad ({np.degrees(theta[-1]):.1f} deg)")
    print(f"  Velocity: ({vx[-1]:.2f}, {vy[-1]:.2f})")
    print(f"  Angular velocity: {omega[-1]:.3f} rad/s")

    # Check if any info indicates what went wrong
    if ep['infos']:
        final_info = ep['infos'][-1]
        print(f"\nFinal info:")
        print(f"  Goal status: {final_info.get('goal', 'unknown')}")
        if 'collision' in final_info:
            print(f"  Collision: {final_info['collision']}")
        if 'crashed' in final_info:
            print(f"  Crashed: {final_info['crashed']}")

    # Create plots
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(f"{meta['mode']} episode - Return: {meta['episode_return']:.1f}", fontsize=14)

    # Plot 1: Trajectory
    ax = axes[0, 0]
    ax.plot(x, y, 'b-', linewidth=1, alpha=0.7)
    ax.plot(x[0], y[0], 'go', markersize=10, label='Start')
    success = meta.get('success', False)
    ax.plot(x[-1], y[-1], 'rx' if not success else 'g*', markersize=10,
            label='End (crashed)' if not success else 'End (landed)')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title('Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Angle over time
    ax = axes[0, 1]
    ax.plot(theta, 'b-', linewidth=1)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Angle (rad)')
    ax.set_title('Angle')
    ax.grid(True, alpha=0.3)

    # Plot 3: Angular velocity over time
    ax = axes[1, 0]
    ax.plot(omega, 'r-', linewidth=1)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    # Highlight high angular velocity regions
    high_omega = np.abs(omega) > 1.0
    if np.any(high_omega):
        ax.fill_between(range(len(omega)), -2, 2, where=high_omega, alpha=0.2, color='red',
                        label='|ω| > 1.0')
        ax.legend()
    ax.set_xlabel('Time step')
    ax.set_ylabel('Angular velocity (rad/s)')
    ax.set_title('Angular Velocity')
    ax.grid(True, alpha=0.3)

    # Plot 4: Actions
    ax = axes[1, 1]
    ax.plot(Fm, 'b-', linewidth=1, alpha=0.7, label='Main engine')
    ax.plot(Fs, 'r-', linewidth=1, alpha=0.7, label='Side engine')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Action')
    ax.set_title('Actions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Velocities
    ax = axes[2, 0]
    ax.plot(vx, 'b-', linewidth=1, alpha=0.7, label='vx')
    ax.plot(vy, 'r-', linewidth=1, alpha=0.7, label='vy')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Velocity')
    ax.set_title('Velocities')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Rewards
    ax = axes[2, 1]
    ax.plot(rewards, 'g-', linewidth=1, alpha=0.7)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Reward')
    ax.set_title('Rewards')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_filename = filename.replace('.pkl', '_plot.png')
    plt.savefig(plot_filename, dpi=150)
    print(f"\nSaved plot to {plot_filename}")
    plt.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python visualize_episode.py <episode_file.pkl>")
        sys.exit(1)

    visualize_episode(sys.argv[1])
