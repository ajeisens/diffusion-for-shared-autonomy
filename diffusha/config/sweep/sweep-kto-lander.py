#!/usr/bin/env python3
"""
Sweep file for training diffusion models on KTO LunarLander episode data.

Four runs (one per line in the generated .jsonl):
  0 - Teleop BC model
  1 - Heuristic BC model
  2 - KTO BC model
  3 - Collision-conditioned model (teleop data + π0.6 CFG quality label)

Usage:
    # Generate sweep .jsonl
    python diffusha/config/sweep/sweep-kto-lander.py

    # Launch run 0 (teleop) on GPU 0
    python -m diffusha.diffusion.train diffusha/config/sweep/sweep-kto-lander.jsonl -l 0

    # Launch run 3 (collision-conditioned) on GPU 0
    python -m diffusha.diffusion.train diffusha/config/sweep/sweep-kto-lander.jsonl -l 3
"""

import os
import sys
from params_proto.hyper import Sweep
from diffusha.config.default_args import Args

this_file_name = sys.argv[0]

with Sweep(Args) as sweep:
    # All runs use the episode-based data source
    Args.data_source = 'episodes'
    Args.episode_data_dir = './demo/interactive/recorded_episodes'
    Args.num_training_steps = 100_000
    Args.save_every = 5_000
    Args.eval_every = -1      # No env-based eval; use eval_kto.py post-training
    Args.batch_size = 4096
    Args.seed = 0

    with sweep.zip:
        Args.episode_modes = [
            ['teleop'],                # run 0: teleop BC
            ['heuristic'],             # run 1: heuristic BC
            ['kto'],                   # run 2: KTO BC
            ['teleop'],                # run 3: collision-conditioned (teleop data)
        ]
        Args.quality_cond = [
            False,                     # run 0: no quality conditioning
            False,                     # run 1: no quality conditioning
            False,                     # run 2: no quality conditioning
            True,                      # run 3: π0.6 CFG quality conditioning
        ]
        Args.cfg_dropout_prob = [
            0.0,                       # unused
            0.0,                       # unused
            0.0,                       # unused
            0.1,                       # 10% dropout on quality label
        ]

sweep.save(os.path.splitext(this_file_name)[0] + '.jsonl')
