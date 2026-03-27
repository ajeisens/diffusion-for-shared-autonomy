"""Per-step latency diagnostics: dataclass and episode logger.

LatencyDiagnostics is attached to env.step() info dicts under "latency".
LatencyLogger accumulates per-step records and computes episode summaries.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass
class LatencyDiagnostics:
    """Snapshot of latency state at one env step."""

    # Step identity
    episode: int
    step: int
    wall_time: float           # time.perf_counter() at action execution

    # Config (for convenience; constant within an episode)
    lander_delay: int          # steps of lander-state delay
    lidar_delay: int           # steps of lidar delay
    inference_delay_s: float   # simulated GPU inference time (s)
    exec_horizon: int          # chunk size before re-inference

    # Which step within the current chunk (0 = fresh inference just ran)
    chunk_step: int

    # Observation staleness (in env steps)
    lander_obs_age: int        # always == lander_delay
    lidar_obs_age: int         # always == lidar_delay

    # Mismatch scores: >1.0 means the obs is older than one full action chunk.
    # This is the critical threshold where latency compensation becomes necessary.
    lander_mismatch_score: float   # lander_delay / exec_horizon
    lidar_mismatch_score: float    # lidar_delay  / exec_horizon

    # Ground-truth vs delivered observation error
    lander_state_error: float      # L2(delayed_lander, true_lander)
    lidar_max_error: float         # max|delayed_lidar - true_lidar| across 8 rays
    lidar_min_reading: float       # min(delayed_lidar) - what the policy "sees"
    lidar_min_reading_true: float  # min(true_lidar)    - what's actually there

    # Task state (from true env state)
    dist_to_pad: float
    speed: float               # sqrt(vx^2 + vy^2)
    angle_abs_deg: float       # |theta| in degrees

    # Async inference state
    # True during the steps where the env advances while waiting for inference.
    # During these steps the policy holds its last action and the obs age grows.
    inference_hold: bool = False

    # Kinodynamic constraint tracking (action continuity)
    # delta_* = |action_t - action_{t-1}|; 0.0 at the first step of each episode.
    # kino_violation = True when any delta exceeds the configured rate limit.
    # On a real arm these correspond to joint velocity/acceleration limit violations
    # produced by discontinuous action chunks planned from stale observations.
    action_main: float = 0.0
    action_side: float = 0.0
    delta_main: float = 0.0
    delta_side: float = 0.0
    kino_violation: bool = False

    # Outcome (only set at terminal step, else None)
    outcome: Optional[str] = None


class LatencyLogger:
    """Accumulates per-step LatencyDiagnostics and computes episode summaries."""

    def __init__(self):
        self._records: List[LatencyDiagnostics] = []
        self._episode_starts: List[int] = []   # record indices where each ep begins

    def begin_episode(self):
        """Call at the start of each episode."""
        self._episode_starts.append(len(self._records))

    def record(self, d: LatencyDiagnostics):
        self._records.append(d)

    def print_live(self, d: LatencyDiagnostics):
        """Print a compact one-liner for the current step."""
        flags = ""
        if d.lander_mismatch_score > 1.0 or d.lidar_mismatch_score > 1.0:
            flags += " MISMATCH"
        if d.inference_hold:
            flags += " HOLD"
        if d.kino_violation:
            flags += f" *** KINO_VIOL(Δm={d.delta_main:.2f} Δs={d.delta_side:.2f})"
        print(
            f"ep={d.episode:3d} t={d.step:4d} | "
            f"act=[{d.action_main:.2f},{d.action_side:+.2f}] "
            f"Δ=[{d.delta_main:.2f},{d.delta_side:.2f}] | "
            f"lander_err={d.lander_state_error:.3f} lidar_err={d.lidar_max_error:.3f} | "
            f"dist_pad={d.dist_to_pad:.2f}{flags}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Episode summaries
    # ------------------------------------------------------------------

    def _episode_records(self, ep_idx: int) -> List[LatencyDiagnostics]:
        start = self._episode_starts[ep_idx]
        end = (
            self._episode_starts[ep_idx + 1]
            if ep_idx + 1 < len(self._episode_starts)
            else len(self._records)
        )
        return self._records[start:end]

    def episode_summary(self, ep_idx: int) -> Dict[str, Any]:
        records = self._episode_records(ep_idx)
        if not records:
            return {}

        last = records[-1]
        n = len(records)

        lander_errors = [r.lander_state_error for r in records]
        lidar_errors  = [r.lidar_max_error    for r in records]
        lidar_mins    = [r.lidar_min_reading   for r in records]
        lidar_mins_t  = [r.lidar_min_reading_true for r in records]
        mismatch_steps = sum(
            1 for r in records
            if r.lander_mismatch_score > 1.0 or r.lidar_mismatch_score > 1.0
        )

        # Kinodynamic stats
        kino_violations = sum(1 for r in records if r.kino_violation)
        delta_mains = [r.delta_main for r in records]
        delta_sides = [r.delta_side for r in records]

        # Identify chunk-start steps: the single step where a fresh inference result
        # first executes.  This is where the dangerous discontinuity lives.
        #
        # In async mode: the step where inference_hold transitions True→False.
        # In sync mode (no hold): the step where chunk_step==0 (every step when eh=1,
        # every eh-th step when eh>1).  chunk_step is always 0 when eh=1 so every
        # sync step counts — correct, because sync re-infers every step at eh=1.
        def _is_chunk_start(i: int) -> bool:
            r = records[i]
            if r.inference_hold:
                return False
            if i == 0:
                return True
            prev = records[i - 1]
            if prev.inference_hold:
                return True       # async: hold just ended, new plan starts
            return r.chunk_step == 0  # sync or zero-inference-delay async

        chunk_start_indices = [i for i in range(n) if _is_chunk_start(i)]
        chunk_start_records = [records[i] for i in chunk_start_indices]

        cs_delta_mains = [r.delta_main for r in chunk_start_records] or [0.0]
        cs_delta_sides = [r.delta_side for r in chunk_start_records] or [0.0]
        n_cs = len(chunk_start_records)

        # Chunk-boundary steps: where inference_hold transitions True→False
        # (the first step of each new chunk, when the discontinuity occurs)
        boundary_violations = sum(
            1 for r in chunk_start_records if r.kino_violation
        )

        return {
            "episode": last.episode,
            "outcome": last.outcome,
            "steps": n,
            # Config
            "lander_delay": last.lander_delay,
            "lidar_delay":  last.lidar_delay,
            "inference_delay_s": last.inference_delay_s,
            "exec_horizon": last.exec_horizon,
            # Mismatch scores (constant within episode, but useful in summary)
            "lander_mismatch_score": last.lander_mismatch_score,
            "lidar_mismatch_score":  last.lidar_mismatch_score,
            "mismatch_steps_pct": 100.0 * mismatch_steps / n,
            # Observation error stats
            "mean_lander_state_error": sum(lander_errors) / n,
            "max_lander_state_error":  max(lander_errors),
            "mean_lidar_max_error":    sum(lidar_errors)  / n,
            "max_lidar_max_error":     max(lidar_errors),
            # Lidar proximity: how close was the policy "thinking" it was vs reality
            "mean_lidar_min_reading":       sum(lidar_mins)   / n,
            "mean_lidar_min_reading_true":  sum(lidar_mins_t) / n,
            # Proximity mismatch: positive = policy overestimated clearance (dangerous)
            "mean_lidar_proximity_overestimate": sum(
                r.lidar_min_reading - r.lidar_min_reading_true for r in records
            ) / n,
            # Kinodynamic constraint violations
            # kino_violation_pct: % of all steps where |Δaction| exceeded the limit
            # cs_violation_pct: % of chunk-start steps that violated the limit
            #   (the primary failure mode — stale-obs plan jumps from current action)
            # mean/max_cs_delta_*: action discontinuity specifically at chunk starts,
            #   not diluted by hold steps — the true severity metric
            "kino_violation_pct":      100.0 * kino_violations / n,
            "cs_violation_pct":        100.0 * boundary_violations / n_cs if n_cs else 0.0,
            "mean_delta_main":         sum(delta_mains) / n,
            "mean_delta_side":         sum(delta_sides) / n,
            "mean_cs_delta_main":      sum(cs_delta_mains) / n_cs,
            "mean_cs_delta_side":      sum(cs_delta_sides) / n_cs,
            "max_cs_delta_main":       max(cs_delta_mains),
            "max_cs_delta_side":       max(cs_delta_sides),
        }

    def all_episode_summaries(self) -> List[Dict[str, Any]]:
        return [self.episode_summary(i) for i in range(len(self._episode_starts))]

    def aggregate_summary(self) -> Dict[str, Any]:
        """Aggregate across all episodes: outcome rates + mean error stats."""
        summaries = self.all_episode_summaries()
        if not summaries:
            return {}

        n_ep = len(summaries)
        outcomes = [s["outcome"] for s in summaries]

        def mean(key):
            vals = [s[key] for s in summaries if s.get(key) is not None]
            return sum(vals) / len(vals) if vals else float("nan")

        # Use config from last episode (constant across a sweep run)
        last = summaries[-1]

        return {
            "n_episodes": n_ep,
            "lander_delay": last["lander_delay"],
            "lidar_delay":  last["lidar_delay"],
            "inference_delay_s": last["inference_delay_s"],
            "exec_horizon": last["exec_horizon"],
            "lander_mismatch_score": last["lander_mismatch_score"],
            "lidar_mismatch_score":  last["lidar_mismatch_score"],
            # Outcome rates
            "landed_pct":    100.0 * outcomes.count("landed")             / n_ep,
            "crashed_pct":   100.0 * outcomes.count("terrain-crash")      / n_ep,
            "obstacle_pct":  100.0 * outcomes.count("obstacle-collision")  / n_ep,
            "timeout_pct":   100.0 * outcomes.count("timeout")            / n_ep,
            "oob_pct":       100.0 * outcomes.count("out-of-bounds")      / n_ep,
            # Error stats (mean across episodes)
            "mean_lander_state_error":           mean("mean_lander_state_error"),
            "mean_lidar_max_error":              mean("mean_lidar_max_error"),
            "mean_lidar_proximity_overestimate": mean("mean_lidar_proximity_overestimate"),
            # Kinodynamic constraint violations (mean across episodes)
            "kino_violation_pct":   mean("kino_violation_pct"),
            "cs_violation_pct":     mean("cs_violation_pct"),
            "mean_delta_main":      mean("mean_delta_main"),
            "mean_delta_side":      mean("mean_delta_side"),
            "mean_cs_delta_main":   mean("mean_cs_delta_main"),
            "mean_cs_delta_side":   mean("mean_cs_delta_side"),
            "max_cs_delta_main":    mean("max_cs_delta_main"),  # mean of per-ep max
            "max_cs_delta_side":    mean("max_cs_delta_side"),
        }

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def to_csv(self, path: str | Path):
        """Write all per-step records to a CSV file."""
        path = Path(path)
        records = self._records
        if not records:
            return

        fieldnames = list(asdict(records[0]).keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(asdict(r))

    def summary_to_csv(self, path: str | Path):
        """Write per-episode summaries to a CSV file."""
        path = Path(path)
        summaries = self.all_episode_summaries()
        if not summaries:
            return

        fieldnames = list(summaries[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in summaries:
                writer.writerow(s)
