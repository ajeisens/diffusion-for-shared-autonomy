# Training Diffusion Models for Shared Autonomy

This document explains how to collect data and train diffusion models for the KTO Lunar Lander shared autonomy demo.

## Overview

The pipeline has three stages:

1. **Collect data** — human teleop + headless heuristic/KTO controllers
2. **Train** — diffusion model learns action distributions from recorded episodes
3. **Evaluate** — run the trained model in the interactive demo

---

## Requirements

### Data collection machine
- Python 3.10–3.12 (Drake requires this range)
- Docker + Docker Desktop (for the Drake/KTO container)
- VcXsrv or XQuartz for display forwarding (Windows/macOS)

### Training machine
- Docker with NVIDIA container runtime
- GPU with CUDA capability sm_89+ (RTX 4090) or sm_120+ (RTX 5090)
  - PyTorch 2.7+ required for RTX 5090 (Blackwell)

---

## Stage 1: Data Collection

All episode files are saved as `.pkl` files in `demo/interactive/recorded_episodes/`.

### 1a. Human teleop (interactive)

Start the interactive demo in Docker:

```bash
docker compose -f demo/interactive/docker-compose.yml --profile windows up lunar-lander-kto-windows
```

Controls:
- `1` — teleop mode
- `E` — toggle recording
- Arrow Up — main engine
- Arrow Left/Right — rotate (partial thrust)
- `R` — reset episode

Aim for 200+ episodes. Success rate around 5–20% is normal and expected — failed episodes are useful training data too.

### 1b. Headless heuristic collection

Runs a physics-based PD controller without a display. No Drake required.

```bash
# From repo root
python demo/interactive/collect_episodes.py \
    --mode heuristic \
    --n_episodes 500 \
    --save_dir demo/interactive/recorded_episodes
```

Expect ~35 ep/s, ~17% success rate, ~46% collision rate.

### 1c. Headless KTO collection (Drake required)

Runs Drake trajectory optimization per episode. Requires the Docker container.

```bash
docker compose -f demo/interactive/docker-compose.yml --profile windows run -d \
    --name kto-collect lunar-lander-kto-windows \
    python collect_episodes.py --mode kto --n_episodes 500 \
    --save_dir ./recorded_episodes
```

Expect ~0.7 ep/s. Monitor with `docker logs -f kto-collect`.

> **Note:** KTO executes plans open-loop, so success rate (~13%) is similar to
> the heuristic. The episodes are still useful — they demonstrate planned
> trajectories and are labelled `kto` in metadata.

### Check your dataset

```bash
python -c "
import torch, json
from pathlib import Path
from collections import defaultdict

save_dir = Path('demo/interactive/recorded_episodes')
files = sorted(save_dir.glob('episode_*.pkl'))
print(f'Total .pkl files: {len(files)}')

by_mode = defaultdict(list)
for f in files:
    ep = torch.load(f, weights_only=False)
    m = ep['metadata']
    by_mode[m['mode']].append(m)

for mode, eps in sorted(by_mode.items()):
    n = len(eps)
    succ = sum(e.get('success', False) for e in eps)
    coll = sum(e.get('collision', False) for e in eps)
    print(f'{mode:12s}: {n:4d} ep  success={succ/n*100:.1f}%  collision={coll/n*100:.1f}%')
"
```

A healthy dataset for training has:
- 200+ teleop episodes (human intent demonstrations)
- 500+ heuristic episodes (diverse success/failure coverage)
- Enough collision examples so the collision-conditioned model has signal

---

## Stage 2: Transfer Data to Training Machine

```bash
# From your local machine
scp -r demo/interactive/recorded_episodes/ \
    user@trainbox:/path/to/repo/demo/interactive/recorded_episodes/
```

---

## Stage 3: Training

Training runs in Docker on the GPU machine. The `Dockerfile.train` installs PyTorch with the correct CUDA version — no Drake or pygame needed.

### 3a. Build the training image

```bash
cd /path/to/repo
docker compose -f docker-compose.train.yml build
```

This takes a few minutes on first run (downloads the PyTorch base image).

### 3b. Generate the sweep file

```bash
docker compose -f docker-compose.train.yml run --rm sweep-gen
```

This writes `diffusha/config/sweep/sweep-kto-lander.jsonl` with four training configurations:

| Run | Mode | Quality conditioning |
|-----|------|----------------------|
| 0   | Teleop BC | No |
| 1   | Heuristic BC | No |
| 2   | KTO BC | No |
| 3   | Teleop + CFG quality label | Yes (collision-conditioned) |

### 3c. Run training

**All four runs sequentially (recommended for single GPU):**

```bash
# Run in a tmux session so it survives SSH disconnects
tmux new -s training
docker compose -f docker-compose.train.yml run --rm train
# Detach: Ctrl+B, D  — reattach: tmux attach -t training
```

**Or run a single specific configuration:**

```bash
docker compose -f docker-compose.train.yml run --rm train \
    python -m diffusha.diffusion.train \
    diffusha/config/sweep/sweep-kto-lander.jsonl -l 0
```

### 3d. Monitor training

From another terminal:

```bash
# Find container name
docker ps

# Stream logs
docker logs -f <container_name>

# Check GPU utilization
watch -n 2 nvidia-smi
```

### 3e. Checkpoints

Models are saved every 5,000 steps to:

```
models/diffusha/<run_id>/step_00005000.pt
models/diffusha/<run_id>/step_00010000.pt
...
models/diffusha/<run_id>/step_00100000.pt
```

Training runs 100,000 steps per configuration. On an RTX 5090 expect roughly 1–2 hours per run.

---

## Training Configurations

Default hyperparameters (set in `diffusha/config/sweep/sweep-kto-lander.py`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_training_steps` | 100,000 | Total gradient steps |
| `batch_size` | 4,096 | Transitions per batch |
| `save_every` | 5,000 | Checkpoint frequency |
| `eval_every` | -1 | Disabled (use eval_kto.py post-training) |
| `copilot_obs_dim` | 6 | State dims fed to model: [x, y, θ, vx, vy, ω] |

To change these, edit `diffusha/config/sweep/sweep-kto-lander.py` and re-run `sweep-gen`.

---

## Post-Training Evaluation

```bash
python diffusha/diffusion/evaluation/eval_kto.py \
    --model_path models/diffusha/<run_id>/step_00100000.pt \
    --n_episodes 100
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'pfrl'`**
The training container doesn't include `pfrl` (legacy dependency). Make sure `Args.data_source = 'episodes'` is set in your sweep — the legacy `replay_buffer` path requires pfrl and is not supported in the training container.

**`CUDA kernel errors` / sm_120 not supported**
Your GPU requires PyTorch 2.6+. The `Dockerfile.train` uses `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime`. Rebuild the image after pulling the latest Dockerfile.

**`weights_only` errors loading episodes**
PyTorch 2.6+ changed the default. This is already fixed in `episode_dataset.py` — make sure you have the latest code (`git pull`).

**Models saving to `/data/ddpm/` but not persisting**
The `./models` directory on the host is mounted to `/data/ddpm` in the container. Check that `docker-compose.train.yml` has the volume entry and that the `models/` directory exists (it will be created automatically on first save).

**OOM during training**
Reduce `batch_size` in `diffusha/config/sweep/sweep-kto-lander.py` (try 2048 or 1024), regenerate the sweep file, and restart.
