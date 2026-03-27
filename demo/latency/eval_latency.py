"""Latency mismatch sweep evaluator for the KTO lunar lander.

Runs the diffusion policy under a grid of latency configurations and reports
how performance degrades as observation delays increase and become mismatched.

Usage:
    # Single config (verbose, live per-step output)
    python -m demo.latency.eval_latency \\
        --checkpoint models/dummy/587ipjs8/step_00099999.pt \\
        --lander-delay 2 --lidar-delay 8 --inference-delay 0.1 \\
        -n 20 --verbose

    # Full sweep over the built-in demo grid
    python -m demo.latency.eval_latency \\
        --checkpoint models/dummy/587ipjs8/step_00099999.pt \\
        --sweep -n 50

    # Custom sweep (comma-separated values for each axis)
    python -m demo.latency.eval_latency \\
        --checkpoint models/dummy/587ipjs8/step_00099999.pt \\
        --lander-delays 0,2 --lidar-delays 0,4,8 --inference-delays 0,0.05,0.1 \\
        --exec-horizons 1,4 -n 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

# ── Path setup so we can import from repo root ────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from diffusha.envs.lunar_lander_kto import LunarLanderKTO
from diffusha.diffusion.evaluation.eval_kto import load_model, sample_action
from demo.latency.latency_env import LatencyConfig, LatencyMismatchWrapper, InferenceDelayedPolicy
from demo.latency.latency_diagnostics import LatencyLogger


class _DiffusionPolicyWrapper:
    """Thin wrapper around eval_kto.py's load_model/sample_action/sample_chunk.

    Provides sample_action(obs6) and sample_chunk(obs6, exec_horizon) using
    the same model loading and action decoding as the authoritative eval_kto.py.
    """
    def __init__(self, checkpoint: str, horizon: int = 16, quality_cond: bool = False):
        self.diffusion = load_model(checkpoint, quality_cond=quality_cond, horizon=horizon)
        self.horizon = horizon
        self.quality_cond = quality_cond
        # eval_kto functions expect the full 15-dim obs; pad unused dims with zeros
        self._obs_buf = np.zeros(15, dtype=np.float32)

    def sample_action(self, obs6: np.ndarray) -> np.ndarray:
        """Single action (exec_horizon=1 path)."""
        self._obs_buf[:6] = obs6
        return sample_action(
            self.diffusion,
            self._obs_buf,
            quality_cond=self.quality_cond,
            horizon=self.horizon,
        )  # already clipped to [0,1] × [-1,1] by eval_kto.py

    def sample_chunk(self, obs6: np.ndarray, exec_horizon: int) -> np.ndarray:
        """Return exec_horizon distinct actions from the full horizon-step chunk."""
        from diffusha.diffusion.evaluation.eval_kto import sample_chunk as _sample_chunk
        self._obs_buf[:6] = obs6
        return _sample_chunk(
            self.diffusion,
            self._obs_buf,
            exec_horizon=exec_horizon,
            quality_cond=self.quality_cond,
            horizon=self.horizon,
        )  # (exec_horizon, 2), already clipped


class _HeuristicPolicy:
    """Simple PD controller — hover and descend toward pad.

    Used as a mock policy when --mock is passed (no checkpoint needed).
    Performance is mediocre (it ignores obstacles) but good enough to
    demonstrate that the latency infrastructure degrades task outcomes.
    """

    def sample_action(self, obs6: np.ndarray) -> np.ndarray:
        x, y, theta, vx, vy, omega = obs6
        from diffusha.envs.lunar_lander_kto import PAD_X, PAD_Y, THRUST_MAX, SIDE_MAX

        # Proportional heading correction toward pad
        dx = PAD_X - x
        target_theta = np.clip(dx * 0.3, -0.5, 0.5)
        theta_err = target_theta - theta

        # Descend when above pad height + 1, otherwise maintain
        target_vy = -1.5 if y > PAD_Y + 1.0 else -0.3

        Fm = THRUST_MAX * np.clip(0.5 - vy * 0.3 + (target_vy - vy) * 0.5, 0.0, 1.0)
        Fs = SIDE_MAX * np.clip(theta_err * 2.0 - omega * 0.5, -1.0, 1.0)

        # Normalise to [-1, 1] as expected by the policy wrapper
        fm_norm = float(np.clip(Fm / THRUST_MAX, 0.0, 1.0))
        fs_norm = float(np.clip(Fs / SIDE_MAX, -1.0, 1.0))
        return np.array([fm_norm, fs_norm], dtype=np.float32)


# ---------------------------------------------------------------------------
# Demo grid
# ---------------------------------------------------------------------------

DEMO_GRID: List[LatencyConfig] = [
    # Baseline: no latency at all
    LatencyConfig(lander_delay=0,  lidar_delay=0,  inference_delay_s=0.0,  exec_horizon=1),
    # Proprioception latency only
    LatencyConfig(lander_delay=2,  lidar_delay=0,  inference_delay_s=0.0,  exec_horizon=1),
    # Vision/lidar latency only
    LatencyConfig(lander_delay=0,  lidar_delay=8,  inference_delay_s=0.0,  exec_horizon=1),
    # Both obs delays, no inference delay
    LatencyConfig(lander_delay=2,  lidar_delay=8,  inference_delay_s=0.0,  exec_horizon=1),
    # Full broken (sync): both delays + inference delay (spin-wait, env frozen)
    LatencyConfig(lander_delay=2,  lidar_delay=8,  inference_delay_s=0.05, exec_horizon=1),
    LatencyConfig(lander_delay=2,  lidar_delay=8,  inference_delay_s=0.10, exec_horizon=1),
    # Larger exec_horizon: reduces inference overhead but obs staleness grows
    LatencyConfig(lander_delay=2,  lidar_delay=8,  inference_delay_s=0.10, exec_horizon=4),
    LatencyConfig(lander_delay=2,  lidar_delay=8,  inference_delay_s=0.10, exec_horizon=8),
]

# Grid for async-inference mode (env advances during inference; 0.1s = 5 steps at 50Hz)
ASYNC_DEMO_GRID: List[LatencyConfig] = [
    # Baseline: no latency at all
    LatencyConfig(lander_delay=0,  lidar_delay=0,  inference_delay_s=0.0,  exec_horizon=1),
    # Both obs delays, no inference delay
    LatencyConfig(lander_delay=2,  lidar_delay=8,  inference_delay_s=0.0,  exec_horizon=1),
    # Async inference: env advances 3 steps (60ms) during inference
    LatencyConfig(lander_delay=2,  lidar_delay=8,  inference_delay_s=0.06, exec_horizon=1),
    # Async inference: env advances 5 steps (100ms) during inference
    LatencyConfig(lander_delay=2,  lidar_delay=8,  inference_delay_s=0.10, exec_horizon=1),
    # Async + exec_horizon=4: obs age = 2+5=7 steps (lander), executes 4 before re-inferring
    LatencyConfig(lander_delay=2,  lidar_delay=8,  inference_delay_s=0.10, exec_horizon=4),
    # Async + exec_horizon=8: obs age = 2+5=7 steps, executes 8 before re-inferring
    LatencyConfig(lander_delay=2,  lidar_delay=8,  inference_delay_s=0.10, exec_horizon=8),
]


# ---------------------------------------------------------------------------
# Core episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: LatencyMismatchWrapper,
    policy: InferenceDelayedPolicy,
    logger: LatencyLogger,
    episode_idx: int,
    seed: int,
    verbose: bool = False,
) -> str:
    """Run one episode.  Returns the outcome string."""
    env.env.seed(seed)
    obs = env.reset()
    policy.reset()
    logger.begin_episode()

    done = False
    outcome = "timeout"

    while not done:
        action = policy.act(obs)
        obs, reward, done, info = env.step(action)

        diag = info["latency"]
        # Tag whether this step was an async inference hold (env advanced while
        # inference was running; policy replayed its last action).
        diag.inference_hold = policy.is_in_inference_hold
        if done and info.get("goal"):
            # Patch outcome into the final diagnostic record
            diag.outcome = info["goal"]
            outcome = info["goal"]

        logger.record(diag)

        if verbose:
            logger.print_live(diag)

    return outcome


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_sweep(
    checkpoint: str | None,
    configs: List[LatencyConfig],
    n_episodes: int = 50,
    base_seed: int = 0,
    verbose: bool = False,
    csv_dir: str | None = None,
    mock: bool = False,
    horizon: int = 16,
    async_inference: bool = False,
) -> List[Dict[str, Any]]:
    """Run n_episodes for each config in configs.  Returns aggregate results.

    Args:
        checkpoint: Path to a .pt checkpoint file.  Ignored when mock=True.
        mock: If True, use the built-in heuristic PD controller instead of a
              trained diffusion model.  Useful for validating the latency
              infrastructure without needing a trained checkpoint.
        async_inference: If True, use async inference mode (env advances during
              inference; no spin-wait).  This compounds sensor latency with
              inference latency, modelling real async-GPU deployments.
    """
    rng = np.random.RandomState(base_seed)
    seeds = [int(rng.randint(0, 2**31)) for _ in range(n_episodes)]

    if mock:
        print("\nUsing heuristic mock policy (--mock mode, no checkpoint needed)")
        policy_model = _HeuristicPolicy()
    else:
        print(f"\nLoading policy from: {checkpoint}")
        policy_model = _DiffusionPolicyWrapper(checkpoint, horizon=horizon)

    mode_label = "ASYNC" if async_inference else "SYNC"
    print(f"Inference mode: {mode_label}")

    all_results = []

    for cfg_idx, cfg in enumerate(configs):
        print(f"\n{'='*70}")
        print(f"Config {cfg_idx+1}/{len(configs)}: {cfg}")
        if async_inference and cfg.inference_steps > 0:
            print(
                f"  [async] inference_steps={cfg.inference_steps}  "
                f"effective_lander_age={cfg.lander_delay + cfg.inference_steps}  "
                f"effective_lidar_age={cfg.lidar_delay + cfg.inference_steps}"
            )
        print(f"{'='*70}")

        env = LatencyMismatchWrapper(LunarLanderKTO(), cfg)
        policy = InferenceDelayedPolicy(policy_model, cfg, async_inference=async_inference)
        logger = LatencyLogger()

        t0 = time.monotonic()
        outcomes = []

        for ep_i, seed in enumerate(seeds):
            outcome = run_episode(env, policy, logger, ep_i, seed, verbose=verbose)
            outcomes.append(outcome)
            if not verbose:
                _print_episode_progress(ep_i, n_episodes, outcome, outcomes, cfg)

        elapsed = time.monotonic() - t0
        agg = logger.aggregate_summary()
        agg["elapsed_s"] = round(elapsed, 1)
        agg["mean_actual_inference_s"] = round(policy.mean_inference_time, 4)
        agg["async_inference"] = async_inference

        all_results.append(agg)

        _print_config_summary(agg, elapsed)

        if csv_dir:
            p = Path(csv_dir)
            p.mkdir(parents=True, exist_ok=True)
            mode_tag = "async" if async_inference else "sync"
            tag = (
                f"{mode_tag}_ld{cfg.lander_delay}_li{cfg.lidar_delay}"
                f"_inf{int(cfg.inference_delay_s*1000)}ms_eh{cfg.exec_horizon}"
            )
            logger.to_csv(p / f"steps_{tag}.csv")
            logger.summary_to_csv(p / f"episodes_{tag}.csv")

    return all_results


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _print_episode_progress(ep_i, n, outcome, outcomes, cfg):
    landed = outcomes.count("landed")
    collisions = outcomes.count("obstacle-collision")
    crashes = outcomes.count("terrain-crash")
    pct = 100.0 * landed / (ep_i + 1)
    print(
        f"  ep {ep_i+1:3d}/{n}  {outcome:22s}  "
        f"landed={pct:5.1f}%  collision={100*collisions/(ep_i+1):5.1f}%  "
        f"crash={100*crashes/(ep_i+1):5.1f}%",
        flush=True,
    )


def _print_config_summary(agg: Dict[str, Any], elapsed: float):
    print(f"\n  --- Summary ---")
    print(f"  landed:             {agg['landed_pct']:5.1f}%")
    print(f"  obstacle collis:    {agg['obstacle_pct']:5.1f}%")
    print(f"  terrain crash:      {agg['crashed_pct']:5.1f}%")
    print(f"  timeout:            {agg['timeout_pct']:5.1f}%")
    print(f"  mean lander err:    {agg['mean_lander_state_error']:.4f}")
    print(f"  mean lidar err:     {agg['mean_lidar_max_error']:.4f}")
    print(f"  lidar overest:      {agg['mean_lidar_proximity_overestimate']:+.4f}")
    print(f"  kino viol (all):    {agg['kino_violation_pct']:5.1f}%")
    print(f"  kino viol (cs%):    {agg['cs_violation_pct']:5.1f}%  ← % of chunk-start steps that violated")
    print(f"  mean cs Δmain/side: {agg['mean_cs_delta_main']:.3f} / {agg['mean_cs_delta_side']:.3f}  ← avg jump at chunk start")
    print(f"  max  cs Δmain/side: {agg['max_cs_delta_main']:.3f} / {agg['max_cs_delta_side']:.3f}  ← worst-case jump")
    print(f"  elapsed:            {elapsed:.1f}s")


def print_sweep_table(results: List[Dict[str, Any]]):
    """Print a compact comparison table across all configs."""
    async_mode = any(r.get("async_inference") for r in results)
    mode_str = "ASYNC" if async_mode else "SYNC"
    W = 135
    print(f"\n{'='*W}")
    print(f"LATENCY SWEEP RESULTS  [{mode_str} inference mode]")
    print(f"{'='*W}")
    # cs_ = chunk-start step metrics (not diluted by hold/within-chunk steps)
    header = (
        f"{'ld':>3} {'li':>3} {'inf_ms':>6} {'eh':>3} | "
        f"{'land%':>6} {'crash%':>7} {'to%':>4} | "
        f"{'land_err':>8} {'lidar_oe':>8} | "
        f"{'cs_viol%':>8} {'csΔmain':>7} {'csΔside':>7} {'maxΔmain':>8} {'maxΔside':>8}"
    )
    if async_mode:
        header = f"{'inf_st':>6} " + header
    print(header)
    print("-" * W)
    for r in results:
        inf_steps = round(r["inference_delay_s"] / (1.0/50))
        row = (
            f"{r['lander_delay']:>3} {r['lidar_delay']:>3} "
            f"{int(r['inference_delay_s']*1000):>6} {r['exec_horizon']:>3} | "
            f"{r['landed_pct']:>6.1f} {r['crashed_pct']:>7.1f} {r['timeout_pct']:>4.1f} | "
            f"{r['mean_lander_state_error']:>8.4f} "
            f"{r['mean_lidar_proximity_overestimate']:>+8.4f} | "
            f"{r['cs_violation_pct']:>8.1f} "
            f"{r['mean_cs_delta_main']:>7.3f} {r['mean_cs_delta_side']:>7.3f} "
            f"{r['max_cs_delta_main']:>8.3f} {r['max_cs_delta_side']:>8.3f}"
        )
        if async_mode:
            row = f"{inf_steps:>6} " + row
        print(row)
    print(
        f"\nColumns: ld=lander_delay(steps) li=lidar_delay(steps) "
        f"inf_ms=inference_delay(ms) eh=exec_horizon\n"
        f"         land%/crash%/to%=outcome rates | "
        f"land_err=lander obs error  lidar_oe=proximity overestimate\n"
        f"         cs_viol%  = % of chunk-start steps where |Δaction| > limit\n"
        f"         csΔmain/csΔside = mean |Δaction| at chunk-start steps (not diluted by hold)\n"
        f"         maxΔmain/maxΔside = mean of per-episode worst-case jump at chunk starts"
        + ("\n         inf_st=env steps advanced during inference" if async_mode else "")
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_configs_from_args(args) -> List[LatencyConfig]:
    """Build config grid from CLI arguments."""
    if args.sweep:
        return DEMO_GRID

    lander_delays    = [int(x)   for x in args.lander_delays.split(",")]
    lidar_delays     = [int(x)   for x in args.lidar_delays.split(",")]
    inference_delays = [float(x) for x in args.inference_delays.split(",")]
    exec_horizons    = [int(x)   for x in args.exec_horizons.split(",")]

    configs = []
    for ld in lander_delays:
        for li in lidar_delays:
            for inf in inference_delays:
                for eh in exec_horizons:
                    configs.append(LatencyConfig(
                        lander_delay=ld,
                        lidar_delay=li,
                        inference_delay_s=inf,
                        exec_horizon=eh,
                    ))
    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate latency mismatch on KTO lunar lander",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to a trained .pt checkpoint file (not needed with --mock)",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use a built-in heuristic PD controller instead of a trained model. "
             "Useful for testing the latency infrastructure without a checkpoint.",
    )
    parser.add_argument("-n", type=int, default=50, help="Episodes per config (default: 50)")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed (default: 0)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-step diagnostics")
    parser.add_argument("--csv-dir", default=None,
                        help="Directory to write per-step and per-episode CSV files")
    parser.add_argument("--out", default=None,
                        help="JSON file to write aggregate results (default: results_latency.json)")

    # Config selection
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sweep", action="store_true",
                      help="Run the built-in demo grid (overrides individual delay args)")

    # Single-config shorthand
    parser.add_argument("--lander-delay", type=int, default=2)
    parser.add_argument("--lidar-delay",  type=int, default=8)
    parser.add_argument("--inference-delay", type=float, default=0.1,
                        help="Seconds of simulated GPU inference delay")
    parser.add_argument("--exec-horizon", type=int, default=1)

    # Multi-value sweep
    parser.add_argument("--lander-delays",    default=None,
                        help="Comma-separated lander_delay values for custom sweep")
    parser.add_argument("--lidar-delays",     default=None)
    parser.add_argument("--inference-delays", default=None)
    parser.add_argument("--exec-horizons",    default=None)

    parser.add_argument("--horizon", type=int, default=16,
                        help="Action chunk length the model was trained with (default: 16)")
    parser.add_argument(
        "--async-inference", action="store_true",
        help="Use async inference mode: env physically advances during inference instead of "
             "spin-waiting.  Compounds sensor latency with inference latency.  "
             "--sweep will use ASYNC_DEMO_GRID instead of DEMO_GRID.",
    )

    args = parser.parse_args()

    # Decide on configs
    if args.sweep:
        configs = ASYNC_DEMO_GRID if args.async_inference else DEMO_GRID
    elif any([args.lander_delays, args.lidar_delays, args.inference_delays, args.exec_horizons]):
        args.lander_delays    = args.lander_delays    or str(args.lander_delay)
        args.lidar_delays     = args.lidar_delays     or str(args.lidar_delay)
        args.inference_delays = args.inference_delays or str(args.inference_delay)
        args.exec_horizons    = args.exec_horizons    or str(args.exec_horizon)
        configs = build_configs_from_args(args)
    else:
        # Single config
        configs = [LatencyConfig(
            lander_delay=args.lander_delay,
            lidar_delay=args.lidar_delay,
            inference_delay_s=args.inference_delay,
            exec_horizon=args.exec_horizon,
        )]

    if not args.mock and not args.checkpoint:
        parser.error("--checkpoint is required unless --mock is used")

    results = run_sweep(
        checkpoint=args.checkpoint,
        configs=configs,
        n_episodes=args.n,
        base_seed=args.seed,
        verbose=args.verbose,
        csv_dir=args.csv_dir,
        mock=args.mock,
        horizon=args.horizon,
        async_inference=args.async_inference,
    )

    print_sweep_table(results)

    out_path = args.out or "results_latency.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
