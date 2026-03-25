#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
import torch

from .base import Actor
from typing import Callable
from functools import partial
device='cuda'

# Magic to avoid circular import due to type hint annotation
# https://adamj.eu/tech/2021/05/13/python-type-hints-how-to-fix-circular-imports/
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from diffusha.diffusion.ddpm import DiffusionModel


class DiffusionAssistedActor(Actor):
    def __init__(self, obs_space, act_space, diffusion: DiffusionModel, behavioral_actor: Actor = None, fwd_diff_ratio: float = 0.45, horizon: int = 1, fwd_diff_steps: int | None = None, exec_horizon: int = 1) -> None:
        super().__init__(obs_space, act_space)
        self.diffusion = diffusion
        self.behavioral_actor = behavioral_actor
        self.fwd_diff_ratio = fwd_diff_ratio

        # self.obs_size = obs_space.low.size
        self.act_size = act_space.low.size

        assert 0 <= fwd_diff_ratio <= 1
        if fwd_diff_steps is not None:
            self._k = fwd_diff_steps
        else:
            self._k = int((self.diffusion.num_diffusion_steps - 1) * self.fwd_diff_ratio)
        self.horizon = horizon
        assert 1 <= exec_horizon <= horizon, (
            f"exec_horizon ({exec_horizon}) must be between 1 and horizon ({horizon})"
        )
        self.exec_horizon = exec_horizon
        print(f'forward diffusion steps for action: {self._k} / {self.diffusion.num_diffusion_steps}')

        # Per-single-env receding horizon state
        self._single_chunk_cache: np.ndarray | None = None  # (exec_horizon, act_dim)
        self._single_step: int = 0

        # Per-batch-env receding horizon state (populated lazily on first batch_act call)
        self._batch_chunk_cache: np.ndarray | None = None   # (B, exec_horizon, act_dim)
        self._batch_step: np.ndarray | None = None          # (B,) int32

    def _diffusion_cond_sample_chunked(self, obs, user_act, run_in_batch=False):
        """Chunked assisted sampling for horizon > 1.

        Position 0: user_act forward-diffused self._k steps.
        Positions 1..horizon-1: pure Gaussian noise N(0, I).
        Then run self._k reverse denoising steps.
        Returns the first exec_horizon action slots as (exec_horizon, act_dim)
        or (B, exec_horizon, act_dim) in batch mode.
        """
        if user_act is None:
            if not run_in_batch:
                user_act = torch.randn(self.act_size)
            else:
                user_act = torch.randn(obs.shape[0], self.act_size)

        if not run_in_batch:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            user_act_tensor = torch.as_tensor(user_act, dtype=torch.float32).unsqueeze(0)
        else:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            user_act_tensor = torch.as_tensor(user_act, dtype=torch.float32)

        B = obs_tensor.shape[0]
        obs_dim = obs_tensor.shape[1]
        act_dim = self.act_size

        # Forward-diffuse [obs | user_act] for self._k steps; keep only the action part
        state_for_diffuse = torch.cat([obs_tensor, user_act_tensor], dim=1)
        k_tensor = torch.as_tensor([self._k]).to(self.diffusion.device)
        state_for_diffuse = state_for_diffuse.float().to(self.diffusion.device)
        x_k, _ = self.diffusion.diffuse(state_for_diffuse, k_tensor)
        user_act_noised = x_k[:, obs_dim:]  # (B, act_dim)

        # Pure noise for the remaining horizon-1 action slots
        rest_noise = torch.randn(B, act_dim * (self.horizon - 1), device=self.diffusion.device)

        # Assemble x_init: [obs | user_act_noised | rest_noise]
        obs_dev = obs_tensor.float().to(self.diffusion.device)
        x_init = torch.cat([obs_dev, user_act_noised, rest_noise], dim=1)

        out, _ = self.diffusion.p_sample_loop(
            shape=x_init.shape,
            start_x=x_init,
            cond=obs_dev,
            naive_cond=True,
            start_t=self._k,
        )

        # Reshape full action block into (B, horizon, act_dim) and return first exec_horizon steps
        act_block = out[:, obs_dim : obs_dim + act_dim * self.horizon]          # (B, act_dim*horizon)
        act_chunk = act_block.reshape(B, self.horizon, act_dim)[:, :self.exec_horizon, :]  # (B, exec_horizon, act_dim)

        if not run_in_batch:
            return act_chunk.squeeze(0).detach().cpu().numpy()  # (exec_horizon, act_dim)
        else:
            return act_chunk.detach().cpu().numpy()             # (B, exec_horizon, act_dim)

    def _diffusion_cond_sample(self, obs, user_act, run_in_batch=False):
        """Conditional sampling"""

        if self.horizon > 1:
            return self._diffusion_cond_sample_chunked(obs, user_act, run_in_batch)

        if user_act is None:
            user_act = torch.randn((self.act_size,))

        # HACK
        if not run_in_batch:
            obs_size = obs.size
        else:
            obs_size = obs.shape[1]

        # import pdb; pdb.set_trace()

        # Concat obs and user action
        if torch.is_tensor(obs):
            state = torch.cat((obs, user_act), axis=1)
        else:
            # Luzhe: TEMP!
            # This if else condition is specific for play.py I am not sure whether this would cause a problem for eval
            state = torch.as_tensor(np.concatenate((obs, user_act), axis=0))

        # NOTE: Currently only support hard conditioning (replacing a part of the input / output)

        # Forward diffuse user_act for k steps
        if not run_in_batch:
            x_k, e = self.diffusion.diffuse(state.unsqueeze(0), torch.as_tensor([self._k]))
        else:
            x_k, e = self.diffusion.diffuse(state, torch.as_tensor([self._k]))

        # Reverse diffuse Tensor([*crisp_obs, *noisy_user_act]) for (diffusion.num_diffusion_steps - k) steps
        obs = torch.as_tensor(obs, dtype=torch.float32)
        x_k[:, :obs_size] = obs  # Add condition
        x_i = x_k
        for i in reversed(range(self._k)):
            x_i = self.diffusion.p_sample(x_i, i)
            x_i[:, :obs_size] = obs  # Add condition

        if not run_in_batch:
            out = x_i.squeeze()  # Remove batch dim
            return out[obs_size:].cpu().numpy()
        else:
            out = x_i
            return out[..., obs_size:].cpu().numpy()


    def act(self, obs: np.ndarray, report_diff: bool = False, return_original: bool = False):
        if isinstance(obs, dict):
            obs_pilot = obs['pilot']
            obs_copilot = obs['copilot']
        else:
            obs_pilot = obs_copilot = obs

        # Get user input every step so the latest intent conditions re-inferences
        user_act = self.behavioral_actor.act(obs_pilot)

        if self.fwd_diff_ratio != 0:
            # Re-infer at chunk boundary; otherwise pop from cache
            if self._single_chunk_cache is None or self._single_step == 0:
                # Returns (exec_horizon, act_dim)
                self._single_chunk_cache = self._diffusion_cond_sample_chunked(
                    obs_copilot, user_act, run_in_batch=False
                )
                self._single_step = 0

            action = self._single_chunk_cache[self._single_step]        # (act_dim,)
            self._single_step = (self._single_step + 1) % self.exec_horizon
        else:
            action = user_act

        if return_original:
            return action, user_act

        if report_diff:
            diff = np.linalg.norm(user_act - action)
            return action, diff
        else:
            return action

    def batch_act(self, obss: np.ndarray, report_diff: bool = False, return_original: bool = False):
        obs_copilots = []
        obs_pilots = []
        for obs in obss:
            if isinstance(obs, dict):
                obs_pilot = obs['pilot']
                obs_copilot = obs['copilot']
            else:
                obs_pilot = obs_copilot = obs
            obs_copilots.append(obs_copilot)
            obs_pilots.append(obs_pilot)

        # Collect fresh user intent every step (used at re-inference boundaries)
        user_actions = self.behavioral_actor.batch_act(obs_pilots)

        if self.fwd_diff_ratio != 0:
            B = len(obss)
            act_dim = self.act_size

            # Lazy init or reset when batch size changes
            if self._batch_chunk_cache is None or self._batch_chunk_cache.shape[0] != B:
                self._batch_chunk_cache = np.zeros((B, self.exec_horizon, act_dim), dtype=np.float32)
                self._batch_step = np.zeros(B, dtype=np.int32)

            # Environments that have reached the end of their current chunk
            reinfer_mask = (self._batch_step == 0)  # (B,) bool

            if reinfer_mask.any():
                reinfer_idx = np.where(reinfer_mask)[0]
                obs_sub = torch.as_tensor(
                    np.array([obs_copilots[i] for i in reinfer_idx]), dtype=torch.float32
                )
                ua_sub = torch.as_tensor(
                    np.array([np.array(user_actions[i]) for i in reinfer_idx]), dtype=torch.float32
                )
                # Returns (n_reinfer, exec_horizon, act_dim)
                new_chunks = self._diffusion_cond_sample_chunked(obs_sub, ua_sub, run_in_batch=True)
                self._batch_chunk_cache[reinfer_idx] = new_chunks

            # Pop current action for each env and advance counters
            arange = np.arange(B)
            actions = self._batch_chunk_cache[arange, self._batch_step]    # (B, act_dim)
            self._batch_step = (self._batch_step + 1) % self.exec_horizon

        else:
            actions = np.array(user_actions)

        if return_original:
            return actions, user_actions

        if report_diff:
            diffs = np.linalg.norm(np.array(user_actions) - np.array(actions), axis=1)
            return actions, diffs
        else:
            return actions

    def reset_chunk(self, env_indices=None) -> None:
        """Reset the receding-horizon chunk cache.

        Call this from an eval loop when an episode ends so the next call to
        act() / batch_act() triggers a fresh diffusion inference rather than
        consuming a stale cached action from the previous episode.

        Args:
            env_indices: int, list of ints, or None.
                         None resets everything (single-env and all batch slots).
                         An index list resets only those batch-env slots.
        """
        if env_indices is None:
            self._single_chunk_cache = None
            self._single_step = 0
            self._batch_chunk_cache = None
            self._batch_step = None
        else:
            if self._batch_step is not None:
                indices = np.atleast_1d(np.asarray(env_indices, dtype=np.int32))
                self._batch_step[indices] = 0
                if self._batch_chunk_cache is not None:
                    self._batch_chunk_cache[indices] = 0.0

    def act_without_env(self, obs: np.ndarray, act: np.ndarray, report_diff: bool = False):
        if isinstance(obs, dict):
            obs_pilot = obs['pilot']
            obs_copilot = obs['copilot']
        else:
            obs_pilot = obs_copilot = obs
        # Get user input
        user_act = act

        if self.fwd_diff_ratio != 0:
            action = self._diffusion_cond_sample(obs_copilot, user_act)
        else:
            action = user_act

        if report_diff:
            diff = np.linalg.norm(user_act - action)
            return action, diff
        else:
            return action




