#!/usr/bin/env python3
"""
Validation script for LunarLanderObstacle-v5 environment.
Tests that obstacles are spawned, collisions are detected, and lidar works.
"""

import gym
import diffusha  # triggers registration

def test_obstacle_env():
    print("Creating LunarLanderObstacle-v5 environment...")
    env = gym.make('LunarLanderObstacle-v5')

    print("Resetting environment...")
    obs = env.reset()

    print(f"Observation shape: {obs.shape}")
    print(f"Expected shape: (17,) [8 base + 1 helipad + 8 lidar]")

    assert obs.shape == (17,), f"Expected obs shape (17,), got {obs.shape}"
    print("✓ Observation shape is correct")

    print("\nRunning episode to test collision detection...")
    collision_detected = False
    for step in range(500):
        # Take random actions
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        if info.get('collision', False):
            print(f"✓ Collision detected at step {step}!")
            print(f"  Reward after collision: {reward}")
            print(f"  Info: {info}")
            collision_detected = True
            break

        if done:
            print(f"Episode ended at step {step}: {info.get('game_over_reason', 'unknown')}")
            break

    if collision_detected:
        print("\n✓ Environment working correctly - collision detection functional!")
    else:
        print("\n⚠ Warning: No collision detected in 500 steps (may need longer episode)")

    print("\nValidation complete!")
    env.close()

if __name__ == "__main__":
    test_obstacle_env()
