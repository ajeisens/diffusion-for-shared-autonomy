"""Latency mismatch wrapper for LunarLanderKTO.

Two core components:

1. LatencyMismatchWrapper(gym.Wrapper)
   Wraps LunarLanderKTO and injects independent per-modality observation delays.

   Observation layout (15-dim, matches lunar_lander_kto.py):
       obs[0:6]   = lander state  (x, y, theta, vx, vy, omega)  — delayed by lander_delay steps
       obs[6:14]  = lidar readings (8 rays, normalized 0-1)       — delayed by lidar_delay steps
       obs[14]    = pad_x (landing pad position)                  — always current (env constant)

   The underlying physics always advances with the true state.  Only what the
   policy receives is delayed.  The true (undelayed) obs is stored in
   info["true_obs"] for diagnostic comparison.

2. InferenceDelayedPolicy
   Wraps any callable policy and adds simulated GPU inference latency.
   At chunk boundaries a spin-wait fires for `inference_delay_s` wall-clock
   seconds before the underlying model is called.  Between boundaries the
   cached action is returned immediately, matching real async-GPU behavior.

Extension point: subclass LatencyMismatchWrapper and override _merge_obs() to
implement compensation strategies (prediction, interpolation, etc.) without
changing any other code.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

import numpy as np
import gym

from demo.latency.latency_diagnostics import LatencyDiagnostics


# Env timestep (must match LunarLanderKTO.DT)
_DT = 1.0 / 50

# Observation slice indices (must match LunarLanderKTO._get_obs)
LANDER_SLICE = slice(0, 6)    # [x, y, theta, vx, vy, omega]
LIDAR_SLICE  = slice(6, 14)   # 8 lidar rays
PAD_IDX      = 14             # pad_x (scalar)

OBS_DIM = 15


@dataclass
class LatencyConfig:
    """All configurable latency parameters in one place.

    Analog mapping to real 6DOF robot:
        lander_delay  →  proprioception (joint encoders, IMU): ~1-5 ms, 1-2 steps at 50 Hz
        lidar_delay   →  camera / vision pipeline: ~30-100 ms, 2-8 steps at 50 Hz
        inference_delay_s → GPU diffusion inference: 50-200 ms depending on hardware
        exec_horizon  →  how many cached actions to execute before re-inferring
    """
    lander_delay: int = 2           # steps; proprioception latency
    lidar_delay: int = 8            # steps; camera/vision latency
    inference_delay_s: float = 0.10 # seconds; simulated GPU inference time
    exec_horizon: int = 1           # chunk steps before re-inference

    # Kinodynamic limits (action rate of change per env step).
    # Analog to joint velocity/acceleration limits on a real arm.
    # A violation flags when |action_t - action_{t-1}| exceeds the limit on either axis.
    # main_thrust ∈ [0,1], side_thrust ∈ [-1,1]; limits expressed in same units per step.
    max_thrust_rate: float = 0.4    # max |Δmain_thrust| per step
    max_side_rate: float = 0.4      # max |Δside_thrust| per step

    @property
    def inference_steps(self) -> int:
        """Number of env steps consumed by inference in async mode.
        In async mode the env physically advances for this many steps while the
        GPU runs, compounding on top of the sensor observation delays."""
        return round(self.inference_delay_s / _DT)

    @property
    def lander_mismatch_score(self) -> float:
        """Normalised staleness of lander obs relative to chunk size.
        > 1.0 means the obs is older than one full action chunk: critical threshold."""
        return self.lander_delay / max(self.exec_horizon, 1)

    @property
    def lidar_mismatch_score(self) -> float:
        """Same for lidar."""
        return self.lidar_delay / max(self.exec_horizon, 1)

    def __str__(self) -> str:
        return (
            f"lander_delay={self.lander_delay} lidar_delay={self.lidar_delay} "
            f"inf={self.inference_delay_s:.3f}s ({self.inference_steps} steps) "
            f"exec_horizon={self.exec_horizon} "
            f"(mismatch: lander={self.lander_mismatch_score:.2f} "
            f"lidar={self.lidar_mismatch_score:.2f})"
        )


class LatencyMismatchWrapper(gym.Wrapper):
    """Inject independent per-modality observation delays into LunarLanderKTO.

    Usage::

        from diffusha.envs.lunar_lander_kto import LunarLanderKTO
        from demo.latency.latency_env import LatencyMismatchWrapper, LatencyConfig

        cfg = LatencyConfig(lander_delay=2, lidar_delay=8)
        env = LatencyMismatchWrapper(LunarLanderKTO(), cfg)

        obs = env.reset()                       # 15-dim delayed obs
        obs, r, done, info = env.step(action)   # info["latency"] = LatencyDiagnostics
                                                # info["true_obs"] = true undelayed obs
    """

    def __init__(self, env: gym.Env, config: LatencyConfig):
        super().__init__(env)
        self.config = config

        # Per-modality FIFO queues — populated in reset()
        self._lander_queue: deque = deque()
        self._lidar_queue:  deque = deque()

        # Ground-truth storage (updated each step before queueing)
        self._true_obs: Optional[np.ndarray] = None

        # Episode and step counters
        self._episode_idx: int = -1
        self._step_idx: int = 0

        # Chunk tracking (for mismatch diagnostics)
        self._chunk_step: int = 0

        # Kinodynamic tracking: last action sent to the underlying env
        self._prev_action: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # gym.Env interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> np.ndarray:
        true_obs = self.env.reset(**kwargs)
        self._true_obs = true_obs
        self._episode_idx += 1
        self._step_idx = 0
        self._chunk_step = 0
        self._prev_action = None
        self._build_queues(true_obs)
        return self._delayed_obs(true_obs)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        # Advance true physics
        true_obs, reward, done, info = self.env.step(action)
        self._true_obs = true_obs
        self._step_idx += 1
        self._chunk_step = (self._chunk_step + 1) % max(self.config.exec_horizon, 1)

        # Push true slices into queues; pop delayed slices
        delayed_obs = self._delayed_obs(true_obs)

        # Compute per-step diagnostics (includes kinodynamic check against prev action)
        diag = self._compute_diagnostics(true_obs, delayed_obs, action, info.get("goal"))

        # Update prev_action after diagnostics are computed
        self._prev_action = action.copy()

        info["latency"] = diag
        info["true_obs"] = true_obs.copy()

        return delayed_obs, reward, done, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_queues(self, initial_obs: np.ndarray):
        """Pre-fill both queues so step 1 has properly initialised delay."""
        lander0 = initial_obs[LANDER_SLICE].copy()
        lidar0  = initial_obs[LIDAR_SLICE].copy()

        # deque of size delay+1: append new → oldest element is delay steps old.
        # With delay=0: maxlen=1, append instantly overwrites → always current.
        ld = self.config.lander_delay
        li = self.config.lidar_delay

        self._lander_queue = deque(
            [lander0.copy() for _ in range(ld + 1)], maxlen=ld + 1
        )
        self._lidar_queue = deque(
            [lidar0.copy() for _ in range(li + 1)], maxlen=li + 1
        )

    def _delayed_obs(self, true_obs: np.ndarray) -> np.ndarray:
        """Push current slices into queues, return the merged delayed observation."""
        # Push true slices; the oldest element is automatically evicted by maxlen
        self._lander_queue.append(true_obs[LANDER_SLICE].copy())
        self._lidar_queue.append(true_obs[LIDAR_SLICE].copy())

        delayed_lander = self._lander_queue[0]   # oldest = lander_delay steps old
        delayed_lidar  = self._lidar_queue[0]    # oldest = lidar_delay steps old
        current_pad_x  = true_obs[PAD_IDX]       # pad_x is an env constant: no delay

        return self._merge_obs(delayed_lander, delayed_lidar, current_pad_x)

    def _merge_obs(
        self,
        delayed_lander: np.ndarray,
        delayed_lidar: np.ndarray,
        pad_x: float,
    ) -> np.ndarray:
        """Assemble the observation delivered to the policy.

        Override this in a subclass to implement compensation strategies
        (e.g., forward-integrate lander state, interpolate lidar, etc.)
        without changing any other code.
        """
        return np.concatenate([
            delayed_lander,
            delayed_lidar,
            [pad_x],
        ]).astype(np.float32)

    def _compute_diagnostics(
        self,
        true_obs: np.ndarray,
        delayed_obs: np.ndarray,
        action: np.ndarray,
        outcome: Optional[str],
    ) -> LatencyDiagnostics:
        cfg = self.config

        true_lander  = true_obs[LANDER_SLICE]
        true_lidar   = true_obs[LIDAR_SLICE]
        del_lander   = delayed_obs[LANDER_SLICE]
        del_lidar    = delayed_obs[LIDAR_SLICE]

        lander_err = float(np.linalg.norm(del_lander - true_lander))
        lidar_err  = float(np.max(np.abs(del_lidar - true_lidar)))

        # Task metrics from true state
        state = self.env.state
        dist_to_pad = abs(state["x"] - self.env.pad_x) + abs(state["y"] - self.env.pad_y)
        speed = (state["vx"] ** 2 + state["vy"] ** 2) ** 0.5
        angle_abs_deg = abs(state["theta"]) * 180.0 / 3.14159265

        # Kinodynamic constraint check: rate of change of action vs configured limits.
        # First step of each episode has no previous action — delta is zero (not a violation).
        if self._prev_action is not None:
            delta_main = float(abs(action[0] - self._prev_action[0]))
            delta_side = float(abs(action[1] - self._prev_action[1]))
        else:
            delta_main = 0.0
            delta_side = 0.0
        kino_violation = (
            delta_main > cfg.max_thrust_rate or
            delta_side > cfg.max_side_rate
        )

        return LatencyDiagnostics(
            episode=self._episode_idx,
            step=self._step_idx,
            wall_time=time.perf_counter(),
            lander_delay=cfg.lander_delay,
            lidar_delay=cfg.lidar_delay,
            inference_delay_s=cfg.inference_delay_s,
            exec_horizon=cfg.exec_horizon,
            chunk_step=self._chunk_step,
            lander_obs_age=cfg.lander_delay,
            lidar_obs_age=cfg.lidar_delay,
            lander_mismatch_score=cfg.lander_mismatch_score,
            lidar_mismatch_score=cfg.lidar_mismatch_score,
            lander_state_error=lander_err,
            lidar_max_error=lidar_err,
            lidar_min_reading=float(np.min(del_lidar)),
            lidar_min_reading_true=float(np.min(true_lidar)),
            dist_to_pad=float(dist_to_pad),
            speed=float(speed),
            angle_abs_deg=float(angle_abs_deg),
            action_main=float(action[0]),
            action_side=float(action[1]),
            delta_main=delta_main,
            delta_side=delta_side,
            kino_violation=kino_violation,
            outcome=outcome,
        )


# ---------------------------------------------------------------------------
# Inference latency wrapper
# ---------------------------------------------------------------------------

class InferenceDelayedPolicy:
    """Wraps a DiffusionPolicy and adds simulated GPU inference latency.

    Two modes are supported, selected by `async_inference`:

    **Sync mode** (default, ``async_inference=False``):
        At chunk boundaries the policy spin-waits for ``config.inference_delay_s``
        wall-clock seconds and then calls the model.  The env does NOT advance
        during the wait.  This is the simpler "broken" baseline.

    **Async mode** (``async_inference=True``):
        At chunk boundaries the policy captures the current (delayed) observation,
        holds the last action for ``config.inference_steps`` env steps while the
        env physically advances, then calls the model on the captured obs.  No
        spin-wait occurs.  The effective obs age at execution time equals
        ``sensor_delay + inference_steps``.  This models real async-GPU pipelines
        where the GPU runs while the robot continues executing its previous plan.

    Args:
        policy: Any object with a sample_action(obs6: np.ndarray) -> np.ndarray
        config: LatencyConfig providing exec_horizon and inference_delay_s
        async_inference: If True, use async mode (env advances during inference).
    """

    def __init__(self, policy, config: LatencyConfig, async_inference: bool = False):
        self.policy = policy
        self.config = config
        self._async_inference = async_inference

        self._chunk_cache: list = []
        self._cache_idx: int = 0
        self._total_inferences: int = 0
        self._total_inference_time: float = 0.0

        # Async-mode state
        self._hold_remaining: int = 0          # steps left in inference hold
        self._pending_obs: Optional[np.ndarray] = None  # obs captured at chunk boundary
        self._last_action: np.ndarray = np.zeros(2, dtype=np.float32)
        self._acted_in_hold: bool = False      # set each act() call

    def reset(self):
        """Call at the start of each episode to clear the chunk cache."""
        self._chunk_cache = []
        self._cache_idx = 0
        self._hold_remaining = 0
        self._pending_obs = None
        self._last_action = np.zeros(2, dtype=np.float32)
        self._acted_in_hold = False

    @property
    def is_in_inference_hold(self) -> bool:
        """True if the most recent act() call returned a held action (async mode only)."""
        return self._acted_in_hold

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Return next action, triggering inference at chunk boundaries.

        In sync mode: inference runs (with spin-wait) before returning.
        In async mode: inference hold starts; env advances for inference_steps
        before inference runs on the captured obs.

        Args:
            obs: Full 15-dim delayed observation from LatencyMismatchWrapper.
                 Only obs[0:6] (delayed lander state) is passed to the policy,
                 matching the training distribution of DiffusionPolicy.

        Returns:
            action: 2-dim action normalised for LunarLanderKTO.step():
                    [main_thrust ∈ [0, 1], side_thrust ∈ [-1, 1]].
        """
        self._acted_in_hold = False

        if self._async_inference:
            if self._hold_remaining > 0:
                # Env is advancing while inference "runs" — replay last action.
                self._hold_remaining -= 1
                if self._hold_remaining == 0:
                    # Hold complete: run inference on the stale captured obs,
                    # then fall through to serve the first action from the new cache.
                    self._run_inference(self._pending_obs)
                else:
                    # Still holding — replay last action.
                    self._acted_in_hold = True
                    return self._last_action.copy()

            if self._cache_idx >= len(self._chunk_cache):
                # Chunk boundary: start async hold (or run immediately if 0 steps).
                inf_steps = self.config.inference_steps
                if inf_steps > 0:
                    self._pending_obs = obs.copy()
                    self._hold_remaining = inf_steps
                    self._acted_in_hold = True
                    return self._last_action.copy()
                else:
                    # Zero inference delay in async mode: run synchronously.
                    self._run_inference(obs)
        else:
            if self._cache_idx >= len(self._chunk_cache):
                # Sync mode: spin-wait then infer.
                self._run_inference(obs)

        action = self._chunk_cache[self._cache_idx]
        self._cache_idx += 1
        self._last_action = action.copy()
        return action

    def _run_inference(self, obs: np.ndarray):
        """Call the underlying policy (with spin-wait in sync mode only)."""
        if not self._async_inference:
            # Sync mode: busy-wait to simulate wall-clock inference time.
            delay_s = self.config.inference_delay_s
            if delay_s > 0.0:
                deadline = time.perf_counter() + delay_s
                while time.perf_counter() < deadline:
                    pass  # busy-wait for accurate timing on Windows

        t0 = time.perf_counter()
        obs6 = obs[:6]  # lander state only — matches training obs
        exec_horizon = self.config.exec_horizon

        if exec_horizon > 1 and hasattr(self.policy, 'sample_chunk'):
            # Use the full planned chunk: (exec_horizon, 2) distinct actions.
            # This matches eval_kto.py's sample_chunk path and is critical for
            # horizon > 1 models — repeating a single action is NOT equivalent.
            chunk = self.policy.sample_chunk(obs6, exec_horizon)  # already clipped
            actions = [chunk[i] for i in range(exec_horizon)]
        else:
            # exec_horizon == 1 or policy doesn't support chunking
            action = self.policy.sample_action(obs6)
            # Clip to env action space: [main_thrust ∈ [0,1], side_thrust ∈ [-1,1]]
            action = np.clip(action, -1.0, 1.0).astype(np.float32)
            action[0] = float(np.clip(action[0], 0.0, 1.0))
            actions = [action]

        t1 = time.perf_counter()

        self._total_inferences += 1
        if self._async_inference:
            # In async mode the "delay" was paid in env steps, not wall time.
            # Only count actual model compute time.
            self._total_inference_time += (t1 - t0)
        else:
            self._total_inference_time += (t1 - t0) + self.config.inference_delay_s

        self._chunk_cache = actions
        self._cache_idx = 0

    @property
    def mean_inference_time(self) -> float:
        if self._total_inferences == 0:
            return 0.0
        return self._total_inference_time / self._total_inferences
