#!/usr/bin/env python3
from typing import Optional
import os
from pathlib import Path
import random
import numpy as np
import torch
from torch.optim import optimizer
from torch.utils.data import IterableDataset, DataLoader
from diffusha.data_collection.episode_dataset import EpisodeDataset
from diffusha.config.default_args import Args
import wandb

from diffusha.diffusion.ddpm import (
    DiffusionCore,
    DiffusionModel,
    # PartiallySmallNoiseDiffusionCore,
    Trainer,
)


class ExpertTransitionDataset(IterableDataset):
    def __init__(
        self, directory, state_dim, action_dim, new_state_dim: int = 0
    ) -> None:
        super().__init__()
        self.state_action_dim = state_dim + action_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        from diffusha.data_collection.generate_data import ReplayBuffer
        self.replay_buffer = ReplayBuffer(directory, state_dim, action_dim)
        self.new_state_dim = new_state_dim

    def __iter__(self):
        while True:
            sample = self.replay_buffer.sample()[: self.state_action_dim]
            if self.new_state_dim > 0:
                state = sample[: self.new_state_dim]
                act = sample[self.state_dim : self.state_dim + self.action_dim]
                sample = np.concatenate((state, act), axis=-1)
            yield sample


class MultiExpertTransitionDataset(IterableDataset):
    def __init__(
        self, directories, state_dim, action_dim, new_state_dim: int = 0
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        from diffusha.data_collection.generate_data import ReplayBuffer
        self.replay_buffers = [
            ReplayBuffer(directory, state_dim, action_dim) for directory in directories
        ]
        self.new_state_dim = new_state_dim

    def __iter__(self):
        while True:
            replay_buffer = random.choice(self.replay_buffers)
            sample = replay_buffer.sample()[: self.state_dim + self.action_dim]
            if self.new_state_dim > 0:
                state = sample[: self.new_state_dim]
                act = sample[self.state_dim : self.state_dim + self.action_dim]
                sample = np.concatenate((state, act), axis=-1)
            yield sample


def get_datadir(env_name, randp):
    if "LunarLander" in env_name:
        lvl = env_name.split("-")[-1]
        data_dir = f"{Args.lunar_data_dir}/{lvl}/randp_{randp:.1f}"
    elif "simple-two-goals" in env_name:
        # Use Simple Multi Goal experts
        data_dir = [
            f"{Args.pointmaze_data_dir}/{env_name}-{goal}-initpos0.4initloc"
            for goal in ["left", "right"]
        ]
    elif "BlockPushMultimodal" in env_name:
        data_dir = [
            f"{Args.blockpush_data_dir}/{target}/randp_{randp:.1f}"
            for target in ["target", "target-flipped"]
        ]
        assert (
            randp != 0.6
        ), 'NOTE that randp_0.6 and target=="target" has only 1,720k transitions rather than 3,000k!'
    else:
        raise ValueError(f"data_dir is unknown for env: {env_name}")

    return data_dir


def main_episodes():
    """Training entry point for KTO LunarLander episode data.

    Used when Args.data_source == 'episodes'. Loads from EpisodeRecorder .pkl
    files instead of the legacy ReplayBuffer format. Skips SAC-based evaluation
    during training (use eval_kto.py for post-training evaluation instead).

    KTO environment dimensions:
        copilot_obs_size = 6   [x, y, θ, vx, vy, ω]
        act_size         = 2   [main_thrust, side_thrust]

    With quality conditioning (Args.quality_cond = True):
        input_size = 9,  cond_dim = 7  (obs + quality label kept noise-free)
    Without:
        input_size = 8,  cond_dim = 6
    """
    # KTO environment has fixed dimensions — no need to instantiate an env.
    copilot_obs_size = 6
    act_size = 2

    # Optional quality label dim for CFG training
    quality_dim = 1 if Args.quality_cond else 0
    input_size = copilot_obs_size + quality_dim + act_size
    cond_dim = copilot_obs_size + quality_dim

    dataset = EpisodeDataset(
        data_dir=Args.episode_data_dir,
        modes=list(Args.episode_modes) if Args.episode_modes is not None else None,
        copilot_obs_dim=copilot_obs_size,
        include_quality_label=Args.quality_cond,
        cfg_dropout_prob=Args.cfg_dropout_prob,
        seed=Args.seed,
    )
    print(f"Dataset: {dataset.episode_count()} episodes, "
          f"success={dataset.success_rate():.1%}, "
          f"collision={dataset.collision_rate():.1%}")

    loader = iter(DataLoader(dataset, batch_size=Args.batch_size, num_workers=4))

    diffusion = DiffusionModel(
        diffusion_core=DiffusionCore(),
        num_diffusion_steps=Args.num_diffusion_steps,
        input_size=input_size,
        beta_schedule=Args.beta_schedule,
        beta_min=Args.beta_min,
        beta_max=Args.beta_max,
        cond_dim=cond_dim,
    )

    trainer = Trainer(
        diffusion,
        cond_dim,   # obs_size used for logging; matches cond_dim here
        act_size,
        save_every=Args.save_every,
        eval_every=Args.eval_every,
    )

    # No make_eval_env: use eval_kto.py for post-training evaluation.
    trainer.train(
        loader,
        make_eval_env=None,
        num_training_steps=Args.num_training_steps,
        eval_assistance=False,
    )


def main():
    if Args.data_source == 'episodes':
        main_episodes()
        return

    from diffusha.data_collection.env import is_lunarlander, make_env
    make_eval_env = lambda **kwargs: make_env(
        Args.env_name,
        test=True,
        split_obs=("LunarLander" in Args.env_name),
        terminate_at_any_goal=True,
        **kwargs,
    )
    sample_env = make_eval_env()

    # NOTE:
    # - pilot_obs: may contain goal information  (pilot == user)
    # - copilot_obs: does not contain goal information  (copilot == assistant)
    act_size = sample_env.action_space.low.size
    pilot_obs_size = sample_env.observation_space.low.size
    if "LunarLander" in Args.env_name:
        copilot_obs_size = sample_env.copilot_observation_space.low.size
    else:
        copilot_obs_size = pilot_obs_size

    # Optionally read from Args.dataset_envs
    # - Args.randp: the probability of a user surrogate to choose a random action
    if Args.dataset_envs is None:
        data_dir = get_datadir(Args.env_name, Args.randp)
    else:
        data_dir = [get_datadir(env_name, Args.randp) for env_name in Args.dataset_envs]

    if isinstance(data_dir, list):
        dataset = MultiExpertTransitionDataset(
            data_dir, pilot_obs_size, act_size, new_state_dim=copilot_obs_size
        )
    else:
        dataset = ExpertTransitionDataset(
            data_dir, pilot_obs_size, act_size, new_state_dim=copilot_obs_size
        )
    loader = iter(DataLoader(dataset, batch_size=Args.batch_size, num_workers=8))

    diffusion = DiffusionModel(
        # diffusion_core=DiffusionCore(small_noise_dim=copilot_obs_size, obs_noise_level=Args.obs_noise_level, obs_noise_cfg_prob=Args.obs_noise_cfg_prob),
        diffusion_core=DiffusionCore(),
        num_diffusion_steps=Args.num_diffusion_steps,
        input_size=(copilot_obs_size + act_size),
        beta_schedule=Args.beta_schedule,
        beta_min=Args.beta_min,
        beta_max=Args.beta_max,
        cond_dim=copilot_obs_size,
    )

    trainer = Trainer(
        diffusion,
        copilot_obs_size,
        act_size,
        save_every=Args.save_every,
        eval_every=Args.eval_every,
    )

    trainer.train(
        loader,
        make_eval_env=make_eval_env,
        num_training_steps=Args.num_training_steps,
        eval_assistance=True,
    )


if __name__ == "__main__":
    # Apply patch on multiprocessing library
    from diffusha.utils import patch

    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_file", type=str, help="sweep file (.jsonl)")
    parser.add_argument(
        "-l", "--line-number", type=int, help="line number of the sweep-file"
    )
    args = parser.parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        avail_gpus = [0]
        gpu_id = 0 if args.line_number is None else args.line_number % len(avail_gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(avail_gpus[gpu_id])

    # Load sweep config from jsonl — one JSON dict per line
    with open(args.sweep_file) as f:
        runs = [json.loads(line) for line in f if line.strip()]
    kwargs = runs[args.line_number]
    Args._update(kwargs)

    sweep_basename = Path(args.sweep_file).stem

    wandb.login()
    wandb.init(
        # Set the project where this run will be logged
        project="diffusha",
        group=f"training-{sweep_basename}",
        config=vars(Args),
    )
    main()
    wandb.finish()
