#!/usr/bin/env python3
"""
Generate sweep .jsonl for training diffusion models on KTO LunarLander episode data.

Runs (one JSON dict per line):
  0 - Heuristic BC model (no latency)
  1 - KTO BC model       (no latency)
  2 - KTO BC model, latency-augmented: delay ~ U[0, 7] per sample
  3 - KTO BC model, latency-augmented: fixed delay = 7 (hardest case)

Latency augmentation (runs 2–3):
  At each training sample, the copilot observation is replaced by
  obs[t - delay] while the action chunk stays at t.  The model learns
  to produce correct actions from stale observations, matching the
  sensor + inference latency present at deployment.

  delay range [0, 7] covers:
    lander_delay=2 (proprioception) + inference_steps=5 (100ms at 50Hz)
  Run 2 sees the full distribution; run 3 stress-tests the worst case.

Usage:
    python diffusha/config/sweep/sweep-kto-lander.py

    # Launch a specific run
    python -m diffusha.diffusion.train diffusha/config/sweep/sweep-kto-lander.jsonl -l 2
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
    horizon=16,
    fwd_diff_steps=5,
)

runs = [
    # ── Baselines (no latency augmentation) ──────────────────────────────────
    {**base, 'episode_modes': ['heuristic'], 'quality_cond': False,
     'cfg_dropout_prob': 0.0, 'lander_delay_range': [0, 0]},   # 0: heuristic BC

    {**base, 'episode_modes': ['kto'],       'quality_cond': False,
     'cfg_dropout_prob': 0.0, 'lander_delay_range': [0, 0]},   # 1: KTO BC

    # ── Latency-augmented training ────────────────────────────────────────────
    # Range [0, 7]: delay=0 (clean) through delay=7 (lander_delay=2 + inf_steps=5).
    # Produces a robust policy that handles all delay levels seen at deployment.
    {**base, 'episode_modes': ['kto'],       'quality_cond': False,
     'cfg_dropout_prob': 0.0, 'lander_delay_range': [0, 7]},   # 2: KTO + latency U[0,7]

    # Fixed delay=7: stress-test whether the model can fully compensate for the
    # worst-case combined sensor + inference staleness.
    {**base, 'episode_modes': ['kto'],       'quality_cond': False,
     'cfg_dropout_prob': 0.0, 'lander_delay_range': [7, 7]},   # 3: KTO + latency fixed=7
]

with open(out_path, 'w') as f:
    for run in runs:
        f.write(json.dumps(run) + '\n')

print(f"Wrote {len(runs)} runs to {out_path}")
