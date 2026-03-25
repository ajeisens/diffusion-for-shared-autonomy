#!/usr/bin/env python3
"""
Headless batch episode collector for Lunar Lander KTO environment.

Runs heuristic or KTO controllers without a display to generate training data
in the same format as play.py's EpisodeRecorder.

Usage:
    # Collect 500 heuristic episodes
    python demo/interactive/collect_episodes.py --mode heuristic --n_episodes 500

    # Collect 500 KTO episodes (requires Drake)
    python demo/interactive/collect_episodes.py --mode kto --n_episodes 500

    # Heuristic with lidar occlusion (failure injection)
    python demo/interactive/collect_episodes.py --mode heuristic --n_episodes 500 --failure_level 2
"""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))  # for episode_recorder

import numpy as np

from diffusha.envs.lunar_lander_kto import LunarLanderKTO
from episode_recorder import EpisodeRecorder


# Occlusion rates per failure level (mirrors play.py)
OCCLUSION_RATES = {0: 0.0, 1: 0.10, 2: 0.30, 3: 0.50}


def heuristic(env, s, failure_level: int = 0):
    """
    Physics-based heuristic PID controller with optional lidar occlusion.

    Gains derived from KTO physics constants (world coordinates):
      INERTIA=0.2, SIDE_MAX=5N, SIDE_ARM=1m, THRUST_MAX=20N, MASS=1kg, G=10m/s²

    Args:
        env: LunarLanderKTO instance
        s: 15-dim observation [x, y, theta, vx, vy, omega, lidar×8, pad_x]
        failure_level: 0-3 lidar occlusion intensity (0 = perfect)

    Returns:
        action: [main_thrust, side_thrust] in [0,1] × [-1,1]
        occluded: set of ray indices that were occluded
    """
    obs = s.copy()
    occluded = set()

    if failure_level > 0:
        rate = OCCLUSION_RATES[failure_level]
        for i in range(6, 14):
            if np.random.random() < rate:
                obs[i] = 1.0
                occluded.add(i - 6)

    x, y, theta, vx, vy, omega = obs[:6]

    # KTO physics constants (world coordinates)
    _G = 10.0; _MASS = 1.0; _I = 0.2
    _FMAX = 20.0; _SMAX = 5.0; _SARM = 1.0; _PAD_Y = 3.33

    pad_x = obs[14]  # = 10.0 world units
    dx = x - pad_x   # lateral offset to pad, world units

    # ---- Angle control ----
    # Tilt toward pad + damp lateral velocity (small gains: dx in world units)
    theta_targ = np.clip(dx * 0.04 + vx * 0.02, -0.4, 0.4)

    # Physically-derived PD gains: omega_n=4 rad/s, critically damped
    # alpha = Fs*arm/I  →  a[1] = alpha_des * I/(SMAX*arm)
    alpha_des = 16.0 * (theta_targ - theta) - 8.0 * omega
    a1 = float(np.clip(alpha_des * _I / (_SMAX * _SARM), -1.0, 1.0))

    # ---- Vertical control ----
    # Proportional descent profile: faster high up, slow near pad
    vy_targ = float(np.clip((_PAD_Y - y) * 0.3, -3.0, -0.2))

    # Gravity compensation + vy tracking
    ay_des = 3.0 * (vy_targ - vy)
    Fm_des = _MASS * (ay_des + _G) / max(abs(np.cos(theta)), 0.5)
    a0 = float(np.clip(Fm_des / _FMAX, 0.0, 1.0))

    # Final approach: straighten up for touchdown
    if y < _PAD_Y + 2.0:
        alpha_des = 16.0 * (0.0 - theta) - 8.0 * omega
        a1 = float(np.clip(alpha_des * _I / (_SMAX * _SARM), -1.0, 1.0))

    return np.array([a0, a1]), occluded


def collect_heuristic_episodes(
    n_episodes: int,
    save_dir: str,
    failure_level: int = 0,
    seed: int = 0,
    verbose: bool = True,
) -> None:
    """Collect heuristic-controlled episodes headlessly."""
    env = LunarLanderKTO()
    recorder = EpisodeRecorder(save_dir=save_dir, env_name='LunarLanderKTO-v1')

    np.random.seed(seed)

    successes = 0
    collisions = 0
    t0 = time.time()

    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        recorder.start_episode(mode='heuristic', initial_obs=obs)

        done = False
        prev_obs = obs

        while not done:
            action, _ = heuristic(env, obs, failure_level=failure_level)
            next_obs, reward, done, info = env.step(action)
            info['control_mode'] = 'heuristic'
            recorder.record_step(prev_obs, action, reward, done, info)
            prev_obs = obs
            obs = next_obs

        meta = recorder.end_episode()
        if meta:
            if meta['success']:
                successes += 1
            if meta['collision']:
                collisions += 1

        if verbose and (ep + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (ep + 1) / elapsed
            remaining = (n_episodes - ep - 1) / rate
            print(
                f"  [{ep+1}/{n_episodes}] "
                f"success={successes}/{ep+1} ({100*successes/(ep+1):.1f}%)  "
                f"collisions={collisions}  "
                f"{rate:.1f} ep/s  ETA {remaining:.0f}s"
            )

    elapsed = time.time() - t0
    print(f"\nDone. {n_episodes} episodes in {elapsed:.1f}s "
          f"({n_episodes/elapsed:.1f} ep/s)")
    print(f"Success rate: {successes}/{n_episodes} ({100*successes/n_episodes:.1f}%)")
    print(f"Collision rate: {collisions}/{n_episodes} ({100*collisions/n_episodes:.1f}%)")


def collect_kto_episodes(
    n_episodes: int,
    save_dir: str,
    seed: int = 0,
    plan_timeout: float = 60.0,
    verbose: bool = True,
) -> None:
    """
    Collect KTO-planned episodes headlessly.

    Requires Drake to be installed (Python 3.10-3.12):
        pip install drake>=1.30.0

    The planner runs synchronously per episode: plan → execute → record.
    """
    try:
        from diffusha.planning.async_planner import AsyncKTOPlanner
        from diffusha.planning.physics_kto import MASS, GRAVITY
    except ImportError as e:
        print(f"ERROR: Drake not available — cannot collect KTO episodes.\n  {e}")
        print("Install Drake in a Python 3.10-3.12 environment: pip install drake>=1.30.0")
        sys.exit(1)

    THRUST_MAX = 2.0 * MASS * GRAVITY
    SIDE_MAX = 0.5 * MASS * GRAVITY

    env = LunarLanderKTO()
    recorder = EpisodeRecorder(save_dir=save_dir, env_name='LunarLanderKTO-v1')
    planner = AsyncKTOPlanner(max_warmstart_iters=0, max_obstacle_iters=12)

    np.random.seed(seed)

    successes = 0
    collisions = 0
    plan_failures = 0
    t0 = time.time()

    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        recorder.start_episode(mode='kto', initial_obs=obs)

        # Plan trajectory from current state
        start = np.array([obs[0], obs[1], obs[2]])
        goal = np.array([env.pad_x, env.pad_y, 0.0])
        obstacles = env.obstacles if hasattr(env, 'obstacles') else []

        planner.start_planning(
            start=start,
            goal=goal,
            obstacles=obstacles,
            num_control_points=20,
            num_dynamics_samples=40,
        )
        planner.wait(timeout=plan_timeout)

        try:
            plan_times, plan, constraint_xy = planner.get_result()
            solve_time = planner.get_solve_time()
            plan_ok = True
        except Exception as e:
            if verbose:
                print(f"  Episode {ep+1}: KTO solve failed ({e}), falling back to heuristic")
            plan_ok = False
            plan_failures += 1

        done = False
        sim_time = 0.0
        prev_obs = obs
        kto_metadata = None

        while not done:
            if plan_ok and sim_time < plan_times[-1]:
                # Interpolate thrust from optimized trajectory
                idx = np.searchsorted(plan_times, sim_time, side='right') - 1
                idx = int(np.clip(idx, 0, len(plan_times) - 1))
                Fm_norm = float(np.clip(plan['Fm'][idx] / THRUST_MAX, 0.0, 1.0))
                Fs_norm = float(np.clip(plan['Fs'][idx] / SIDE_MAX, -1.0, 1.0))
                action = np.array([Fm_norm, Fs_norm], dtype=np.float32)
            else:
                # Fallback heuristic (plan expired or failed)
                action, _ = heuristic(env, obs)
                action = action.astype(np.float32)

            next_obs, reward, done, info = env.step(action)
            info['control_mode'] = 'kto'

            if done and plan_ok:
                kto_metadata = {
                    'solve_time': solve_time,
                    'n_obstacles': len(obstacles),
                    'converged': True,
                    'planned_xy': [
                        (float(plan['x'][i]), float(plan['y'][i]))
                        for i in range(0, len(plan['x']), 5)
                    ],
                    'constraint_xy': (
                        constraint_xy.tolist()
                        if constraint_xy is not None else []
                    ),
                    'plan_duration': float(plan_times[-1]),
                    'manual_override_at': None,
                }
                info['kto_metadata'] = kto_metadata

            recorder.record_step(prev_obs, action, reward, done, info)
            prev_obs = obs
            obs = next_obs
            sim_time += env.dt if hasattr(env, 'dt') else (1.0 / 50)

        meta = recorder.end_episode()
        if meta:
            if meta['success']:
                successes += 1
            if meta['collision']:
                collisions += 1

        if verbose and (ep + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (ep + 1) / elapsed
            remaining = (n_episodes - ep - 1) / rate
            print(
                f"  [{ep+1}/{n_episodes}] "
                f"success={successes}/{ep+1} ({100*successes/(ep+1):.1f}%)  "
                f"plan_failures={plan_failures}  "
                f"{rate:.1f} ep/s  ETA {remaining:.0f}s"
            )

    elapsed = time.time() - t0
    print(f"\nDone. {n_episodes} episodes in {elapsed:.1f}s "
          f"({n_episodes/elapsed:.1f} ep/s)")
    print(f"Success rate:   {successes}/{n_episodes} ({100*successes/n_episodes:.1f}%)")
    print(f"Collision rate: {collisions}/{n_episodes} ({100*collisions/n_episodes:.1f}%)")
    print(f"Plan failures:  {plan_failures}/{n_episodes}")


def main():
    parser = argparse.ArgumentParser(
        description="Headless episode collector for LunarLanderKTO"
    )
    parser.add_argument(
        "--mode",
        choices=["heuristic", "kto"],
        required=True,
        help="Control mode to use for data collection",
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=500,
        help="Number of episodes to collect (default: 500)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./recorded_episodes",
        help="Directory to save episode files (default: ./recorded_episodes)",
    )
    parser.add_argument(
        "--failure_level",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Lidar occlusion level for heuristic mode: "
             "0=none, 1=10%%, 2=30%%, 3=50%% (default: 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed base (default: 0)",
    )
    parser.add_argument(
        "--plan_timeout",
        type=float,
        default=60.0,
        help="KTO solver timeout per episode in seconds (default: 60)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-episode progress output",
    )
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Collecting {args.n_episodes} episodes in {args.mode.upper()} mode")
    print(f"Save directory: {save_dir.resolve()}")
    if args.mode == "heuristic":
        print(f"Failure level:  {args.failure_level} "
              f"({OCCLUSION_RATES[args.failure_level]*100:.0f}% lidar occlusion)")
    print("=" * 60)

    if args.mode == "heuristic":
        collect_heuristic_episodes(
            n_episodes=args.n_episodes,
            save_dir=str(save_dir),
            failure_level=args.failure_level,
            seed=args.seed,
            verbose=not args.quiet,
        )
    elif args.mode == "kto":
        collect_kto_episodes(
            n_episodes=args.n_episodes,
            save_dir=str(save_dir),
            seed=args.seed,
            plan_timeout=args.plan_timeout,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
