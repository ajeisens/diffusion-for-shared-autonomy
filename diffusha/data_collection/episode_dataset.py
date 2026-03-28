"""
Dataset loader for episodes recorded by demo/interactive/collect_episodes.py
and demo/interactive/play.py (EpisodeRecorder format).

Provides a drop-in replacement for ExpertTransitionDataset that reads from
the new .pkl episode files instead of the legacy ReplayBuffer chunked format.

Episode file structure (torch.save'd dict):
    {
        'observations': np.ndarray (T, 15)  [x, y, θ, vx, vy, ω, lidar×8, pad_x]
        'actions':      np.ndarray (T, 2)   [main_thrust, side_thrust]
        'rewards':      np.ndarray (T,)
        'dones':        np.ndarray (T,) bool
        'infos':        list[dict]
        'metadata': {
            'mode':      str   ('teleop', 'heuristic', 'kto')
            'success':   bool
            'collision': bool
            'crashed':   bool
            ...
        }
    }

Sampling convention (matches ExpertTransitionDataset):
    Each sample is a 1-D numpy array: [copilot_obs | (quality) | action]
    - copilot_obs = obs[:copilot_obs_dim]  (default: first 6 dims)
    - quality     = episode-level binary label (1 = no collision, 0 = collision)
                    only included when include_quality_label=True
    - action      = actions[t]
"""

import json
import random
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset


class EpisodeDataset(IterableDataset):
    """
    Streams (copilot_obs, action) transitions from recorded episode files.

    Optionally includes a binary episode-level quality label suitable for
    classifier-free guidance training (π0.6 / RECAP approach).

    Args:
        data_dir: Path to directory containing episode .pkl files and metadata.json.
        modes: Control modes to include, e.g. ['teleop'], ['heuristic', 'kto'].
               None means all modes.
        copilot_obs_dim: Number of leading observation dims used as conditioning.
                         Default 6: [x, y, θ, vx, vy, ω] (strips lidar + pad_x).
        include_quality_label: If True, insert a scalar quality label between
                               copilot_obs and action. Label is 1.0 if the episode
                               had no collision, 0.0 if it did.
        cfg_dropout_prob: During iteration, zero out the quality label with this
                          probability for classifier-free guidance training.
                          Only used when include_quality_label=True.
        filter_successful_only: If True, only include episodes where success=True.
        seed: Random seed for reproducibility.
        lander_delay_range: (min_delay, max_delay) in env steps.  At each sampled
                            transition a delay is drawn uniformly from this range and
                            applied to the copilot observation: the model receives
                            obs[t - delay] instead of obs[t], while the action chunk
                            remains anchored at t.  This trains the policy to produce
                            correct actions from stale observations, mimicking the
                            sensor latency present at deployment on a real robot.
                            Default (0, 0) disables augmentation (original behaviour).
    """

    def __init__(
        self,
        data_dir: str,
        modes: Optional[List[str]] = None,
        copilot_obs_dim: int = 6,
        include_quality_label: bool = False,
        cfg_dropout_prob: float = 0.1,
        filter_successful_only: bool = False,
        seed: Optional[int] = None,
        horizon: int = 1,
        lander_delay_range: Tuple[int, int] = (0, 0),
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.modes: Optional[Set[str]] = set(modes) if modes is not None else None
        self.copilot_obs_dim = copilot_obs_dim
        self.include_quality_label = include_quality_label
        self.cfg_dropout_prob = cfg_dropout_prob
        self.filter_successful_only = filter_successful_only
        self.seed = seed
        self.horizon = horizon
        self.lander_delay_range = lander_delay_range

        # Load episode index from metadata.json
        self._episodes = self._load_index()
        if not self._episodes:
            raise ValueError(
                f"No episodes found in {self.data_dir} "
                f"(modes={modes}, filter_successful_only={filter_successful_only}). "
                f"Run collect_episodes.py or play.py to generate data."
            )

        delay_str = (
            f"delay={lander_delay_range[0]}"
            if lander_delay_range[0] == lander_delay_range[1]
            else f"delay~U[{lander_delay_range[0]},{lander_delay_range[1]}]"
        )
        print(
            f"EpisodeDataset: {len(self._episodes)} episodes from {self.data_dir} "
            f"(modes={modes or 'all'}, quality_label={include_quality_label}, {delay_str})"
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def sample_dim(self) -> int:
        """Dimension of each yielded sample."""
        return self.copilot_obs_dim + (1 if self.include_quality_label else 0) + 2 * self.horizon

    def episode_count(self) -> int:
        return len(self._episodes)

    def success_rate(self) -> float:
        n = len(self._episodes)
        return sum(e["success"] for e in self._episodes) / n if n else 0.0

    def collision_rate(self) -> float:
        n = len(self._episodes)
        return sum(e["collision"] for e in self._episodes) / n if n else 0.0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_index(self) -> List[dict]:
        """Read metadata.json and return filtered episode list."""
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            # Fall back to scanning directory if metadata.json absent
            return self._scan_directory()

        with open(metadata_path, "r") as f:
            data = json.load(f)

        episodes = data.get("episodes", [])
        return self._filter(episodes)

    def _scan_directory(self) -> List[dict]:
        """Fallback: infer metadata by loading each .pkl header."""
        episodes = []
        for pkl_path in sorted(self.data_dir.glob("episode_*.pkl")):
            try:
                ep = torch.load(pkl_path, map_location="cpu", weights_only=False)
                meta = ep.get("metadata", {})
                mode = meta.get("mode", pkl_path.stem.rsplit("_", 1)[-1])
                entry = {
                    "filename": pkl_path.name,
                    "mode": mode,
                    "success": bool(meta.get("success", False)),
                    "collision": bool(meta.get("collision", False)),
                    "crashed": bool(meta.get("crashed", False)),
                    "return": float(meta.get("episode_return", 0.0)),
                    "length": int(meta.get("episode_length", 0)),
                }
                episodes.append(entry)
            except Exception as e:
                print(f"Warning: could not read {pkl_path.name}: {e}")
        return self._filter(episodes)

    def _filter(self, episodes: List[dict]) -> List[dict]:
        out = []
        for ep in episodes:
            if self.modes is not None and ep.get("mode") not in self.modes:
                continue
            if self.filter_successful_only and not ep.get("success", False):
                continue
            out.append(ep)
        return out

    def _load_episode(self, entry: dict):
        """Load a single episode .pkl and return (observations, actions, quality_label)."""
        path = self.data_dir / entry["filename"]
        ep = torch.load(path, map_location="cpu", weights_only=False)

        observations = np.asarray(ep["observations"], dtype=np.float32)  # (T, 15)
        actions = np.asarray(ep["actions"], dtype=np.float32)              # (T, 2)

        # Episode-level quality label: 1 = clean (no collision), 0 = had collision
        quality_label = 0.0 if entry.get("collision", False) else 1.0

        return observations, actions, quality_label

    # ------------------------------------------------------------------
    # IterableDataset interface
    # ------------------------------------------------------------------

    def __iter__(self):
        rng = random.Random(self.seed)

        while True:
            # Sample a random episode from the index
            entry = rng.choice(self._episodes)

            try:
                observations, actions, quality_label = self._load_episode(entry)
            except Exception as e:
                print(f"Warning: failed to load {entry['filename']}: {e}")
                continue

            T = len(actions)
            if T == 0:
                continue

            # Sample a random transition from the episode
            t = rng.randint(0, T - 1)

            # Latency augmentation: deliver a stale observation to the policy
            # while keeping the action chunk anchored at the true time t.
            # delay=0 is the original behaviour.
            delay = rng.randint(self.lander_delay_range[0], self.lander_delay_range[1])
            t_obs = max(0, t - delay)
            copilot_obs = observations[t_obs, : self.copilot_obs_dim]  # (6,)

            if self.horizon == 1:
                action = actions[t]  # (2,)
            else:
                # Build action window of length horizon; zero-pad if near episode end
                window = actions[t : t + self.horizon]  # (<=horizon, act_dim)
                act_dim = actions.shape[1]
                if len(window) < self.horizon:
                    pad = np.zeros(
                        (self.horizon - len(window), act_dim), dtype=np.float32
                    )
                    window = np.concatenate([window, pad], axis=0)  # (horizon, act_dim)
                action = window.flatten()  # (act_dim * horizon,)

            if self.include_quality_label:
                # CFG dropout: randomly zero out quality label during training
                q = quality_label
                if rng.random() < self.cfg_dropout_prob:
                    q = 0.0  # treat as unconditional
                quality = np.array([q], dtype=np.float32)
                sample = np.concatenate([copilot_obs, quality, action])
            else:
                sample = np.concatenate([copilot_obs, action])

            yield sample
