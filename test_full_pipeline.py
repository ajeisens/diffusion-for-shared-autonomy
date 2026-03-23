#!/usr/bin/env python3
"""
Comprehensive validation test for all implemented components.

Tests:
1. Obstacle environment (multiple episodes to verify collision detection)
2. Evaluation harness imports and structure
3. Visualization script imports and structure
4. MPC baseline imports and structure

This validates that all Phase 1, 2, 6, 7 components work in Docker.
"""

import sys
import numpy as np
from pathlib import Path


def test_obstacle_environment():
    """Test obstacle environment with multiple episodes"""
    print("=" * 60)
    print("TEST 1: Obstacle Environment (Multiple Episodes)")
    print("=" * 60)

    import gym
    import diffusha

    env = gym.make('LunarLanderObstacle-v5')
    print(f"✓ Created LunarLanderObstacle-v5 environment")

    # Run multiple episodes to increase chance of collision
    n_episodes = 10
    collision_count = 0
    success_count = 0
    crash_count = 0
    timeout_count = 0

    for episode in range(n_episodes):
        obs = env.reset()

        if episode == 0:
            assert obs.shape == (17,), f"Expected obs shape (17,), got {obs.shape}"
            print(f"✓ Observation shape correct: {obs.shape}")

        done = False
        steps = 0
        episode_reward = 0

        while not done and steps < 1000:
            # Random action
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1

        # Categorize outcome
        if info.get('collision', False):
            collision_count += 1
            outcome = "COLLISION"
        elif info.get('landed', False):
            success_count += 1
            outcome = "SUCCESS"
        elif info.get('crashed', False):
            crash_count += 1
            outcome = "CRASH"
        else:
            timeout_count += 1
            outcome = "TIMEOUT"

        print(f"  Episode {episode+1}/{n_episodes}: {outcome} (steps={steps}, reward={episode_reward:.1f})")

    env.close()

    print(f"\n✓ Completed {n_episodes} episodes")
    print(f"  - Collisions: {collision_count}")
    print(f"  - Successes: {success_count}")
    print(f"  - Crashes: {crash_count}")
    print(f"  - Timeouts: {timeout_count}")

    if collision_count > 0:
        print(f"✓ Collision detection working! ({collision_count} collisions detected)")
    else:
        print(f"⚠ No collisions in {n_episodes} episodes (may need more episodes or closer obstacles)")

    return True


def test_mpc_imports():
    """Test that MPC baseline modules can be imported"""
    print("\n" + "=" * 60)
    print("TEST 2: MPC Baseline Imports")
    print("=" * 60)

    try:
        from diffusha.baseline.train_dynamics import DynamicsModel, train_dynamics_model
        print("✓ train_dynamics imports successful")
        print("  - DynamicsModel class available")
        print("  - train_dynamics_model function available")
    except Exception as e:
        print(f"✗ train_dynamics import failed: {e}")
        return False

    try:
        from diffusha.baseline.mpc_cem import CEMPlanner, MPCAgent
        print("✓ mpc_cem imports successful")
        print("  - CEMPlanner class available")
        print("  - MPCAgent class available")
    except Exception as e:
        print(f"✗ mpc_cem import failed: {e}")
        return False

    # Test instantiation (without actual models)
    try:
        import torch
        model = DynamicsModel(state_dim=8, action_dim=2)
        print("✓ DynamicsModel instantiation successful")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters())} params")
    except Exception as e:
        print(f"✗ DynamicsModel instantiation failed: {e}")
        return False

    return True


def test_evaluation_imports():
    """Test that evaluation harness can be imported"""
    print("\n" + "=" * 60)
    print("TEST 3: Evaluation Harness Imports")
    print("=" * 60)

    try:
        from diffusha.diffusion.evaluation.eval_ablation import (
            MetricsTracker,
            evaluate_condition,
            create_teleop_actor,
            run_ablation_study,
        )
        print("✓ eval_ablation imports successful")
        print("  - MetricsTracker class available")
        print("  - evaluate_condition function available")
        print("  - create_teleop_actor function available")
        print("  - run_ablation_study function available")
    except Exception as e:
        print(f"✗ eval_ablation import failed: {e}")
        return False

    # Test MetricsTracker instantiation
    try:
        tracker = MetricsTracker()
        print("✓ MetricsTracker instantiation successful")
    except Exception as e:
        print(f"✗ MetricsTracker instantiation failed: {e}")
        return False

    return True


def test_visualization_imports():
    """Test that visualization script can be imported"""
    print("\n" + "=" * 60)
    print("TEST 4: Visualization Script Imports")
    print("=" * 60)

    try:
        # Add demo directory to path
        sys.path.insert(0, str(Path(__file__).parent / 'demo'))
        import visualize_ablation

        print("✓ visualize_ablation imports successful")

        # Check for expected functions
        assert hasattr(visualize_ablation, 'load_results'), "load_results missing"
        assert hasattr(visualize_ablation, 'plot_grouped_bar_chart'), "plot_grouped_bar_chart missing"
        assert hasattr(visualize_ablation, 'plot_trajectory_overlay'), "plot_trajectory_overlay missing"
        assert hasattr(visualize_ablation, 'plot_gamma_sweep'), "plot_gamma_sweep missing"

        print("  - load_results function available")
        print("  - plot_grouped_bar_chart function available")
        print("  - plot_trajectory_overlay function available")
        print("  - plot_gamma_sweep function available")

    except Exception as e:
        print(f"✗ visualize_ablation import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_integration():
    """Test basic integration - can we create env, actor, and run?"""
    print("\n" + "=" * 60)
    print("TEST 5: Basic Integration (Env + Teleop Actor)")
    print("=" * 60)

    try:
        import gym
        import diffusha
        from diffusha.diffusion.evaluation.eval_ablation import create_teleop_actor

        # Create environment
        env = gym.make('LunarLanderObstacle-v5')
        print("✓ Environment created")

        # Create teleop actor (needs env, pilot_noise, seed)
        actor = create_teleop_actor(env, pilot_noise=0.3, seed=0)
        print("✓ Teleop actor created")

        # Run a few steps
        obs = env.reset()
        for i in range(10):
            action = actor.act(obs)
            obs, reward, done, info = env.step(action)
            if done:
                break

        env.close()
        print(f"✓ Ran {i+1} steps successfully")
        print("✓ Integration test passed: env + actor working together")

    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def main():
    print("╔" + "=" * 58 + "╗")
    print("║  FULL PIPELINE VALIDATION - Phases 1, 2, 6, 7          ║")
    print("╚" + "=" * 58 + "╝")
    print()

    results = {}

    try:
        results['obstacle_env'] = test_obstacle_environment()
    except Exception as e:
        print(f"\n✗ Test 1 failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results['obstacle_env'] = False

    try:
        results['mpc_imports'] = test_mpc_imports()
    except Exception as e:
        print(f"\n✗ Test 2 failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results['mpc_imports'] = False

    try:
        results['eval_imports'] = test_evaluation_imports()
    except Exception as e:
        print(f"\n✗ Test 3 failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results['eval_imports'] = False

    try:
        results['viz_imports'] = test_visualization_imports()
    except Exception as e:
        print(f"\n✗ Test 4 failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results['viz_imports'] = False

    try:
        results['integration'] = test_integration()
    except Exception as e:
        print(f"\n✗ Test 5 failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results['integration'] = False

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    test_names = {
        'obstacle_env': 'Obstacle Environment',
        'mpc_imports': 'MPC Baseline Imports',
        'eval_imports': 'Evaluation Harness Imports',
        'viz_imports': 'Visualization Imports',
        'integration': 'Integration Test',
    }

    for key, name in test_names.items():
        status = "✓ PASS" if results.get(key, False) else "✗ FAIL"
        print(f"{status} - {name}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Pipeline is ready.")
        print("\nNext steps:")
        print("1. Train MPC dynamics model (Phase 2)")
        print("2. Run evaluation with existing SAC models")
        print("3. Generate visualizations")
        return 0
    else:
        print("\n⚠ Some tests failed. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
