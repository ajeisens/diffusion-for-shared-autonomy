#!/usr/bin/env python3
from params_proto import ParamsProto
import os
proj_root = os.environ.get('RMX_CODE_DIR', '')

class Args(ParamsProto):
    env_name = 'LunarLander-v1'
    dataset_envs = None

    # DDPM configuration
    num_diffusion_steps = 50
    beta_min = 1e-4
    beta_max = 0.26
    beta_schedule = 'sigmoid'
    ddpm_model_path = '/data/ddpm'

    num_training_steps = 100_000
    eval_every = 2000
    save_every = 2000

    # Data directories
    lunarlander_data_dir = '/data/replay/lunarlander'
    pointmaze_data_dir = '/data/replay/pointmaze'
    blockpush_data_dir = '/data/replay/blockpush'

    # Stores evaluation results
    results_dir = '/data/results'
    pt_dir = '/data'

    randp = 0.
    seed = 0

    # Used in evaluation
    fwd_diff_ratio = 0.4
    laggy_actor_repeat_prob = 0.8
    noisy_actor_eps = 0.8

    batch_size = 4096

    # Temporary directory that stores SAC model checkpoints
    sac_model_dir = '/data/sac'

    # Episode-based training (KTO lunar lander)
    # Set data_source = 'episodes' to train from EpisodeRecorder .pkl files
    # instead of the legacy ReplayBuffer chunked format.
    data_source: str = 'replay_buffer'   # 'replay_buffer' | 'episodes'
    episode_data_dir: str = './demo/interactive/recorded_episodes'
    episode_modes = None                 # e.g. ['teleop'] or ['heuristic', 'kto']

    # Classifier-free guidance quality conditioning (π0.6 style).
    # When True, expects episode data to include a binary collision label,
    # which is appended between copilot_obs and action dims.
    # cond_dim is automatically extended by 1 to cover the quality label.
    quality_cond: bool = False
    cfg_dropout_prob: float = 0.1       # Probability of zeroing quality label during training
