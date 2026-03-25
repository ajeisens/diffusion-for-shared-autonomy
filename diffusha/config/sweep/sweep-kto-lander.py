#!/usr/bin/env python3
"""
Generate sweep .jsonl for training diffusion models on KTO LunarLander episode data.

Four runs (one JSON dict per line):
  0 - Teleop BC model
  1 - Heuristic BC model
  2 - KTO BC model
  3 - Collision-conditioned model (teleop data + CFG quality label)

Usage:
    python diffusha/config/sweep/sweep-kto-lander.py

    # Launch run 0 (teleop) on GPU 0
    python -m diffusha.diffusion.train diffusha/config/sweep/sweep-kto-lander.jsonl -l 0

    # Launch run 3 (collision-conditioned)
    python -m diffusha.diffusion.train diffusha/config/sweep/sweep-kto-lander.jsonl -l 3
"""

import json
import os
import sys

out_path = os.path.splitext(sys.argv[0])[0] + '.jsonl'

base = dict(
    data_source='episodes',
    episode_data_dir='./demo/interactive/recorded_episodes',
    num_training_steps=100_000,
    save_every=5_000,
    eval_every=-1,
    batch_size=4096,
    seed=0,
)

runs = [
    {**base, 'episode_modes': ['teleop'],    'quality_cond': False, 'cfg_dropout_prob': 0.0},  # 0: teleop BC
    {**base, 'episode_modes': ['heuristic'], 'quality_cond': False, 'cfg_dropout_prob': 0.0},  # 1: heuristic BC
    {**base, 'episode_modes': ['kto'],       'quality_cond': False, 'cfg_dropout_prob': 0.0},  # 2: KTO BC
    {**base, 'episode_modes': ['teleop'],    'quality_cond': True,  'cfg_dropout_prob': 0.1},  # 3: collision-conditioned
]

with open(out_path, 'w') as f:
    for run in runs:
        f.write(json.dumps(run) + '\n')

print(f"Wrote {len(runs)} runs to {out_path}")
