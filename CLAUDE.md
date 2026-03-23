# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Existing Codebase Architecture

This is the **Diffusion for Shared Autonomy** (DiffuSHA) codebase from Yoneda et al., RSS 2023. The core idea: use diffusion models to assist human operators by blending noisy human actions with learned expert behavior through partial reverse diffusion.

### Module Structure

```
diffusha/
├── actor/                     # Actor abstraction for different control policies
│   ├── base.py               # Base Actor class + variants (Random, Expert, Noisy, Laggy)
│   └── assistive.py          # DiffusionAssistedActor - wraps behavioral actor with diffusion
├── diffusion/                # Diffusion model training and inference
│   ├── ddpm.py              # Core: DiffusionModel, DiffusionCore, Trainer
│   ├── models.py            # ConditionalModel (score network), EMA
│   ├── train.py             # Training entry point (uses sweep files)
│   ├── sha.py               # Shared autonomy utilities
│   └── evaluation/
│       └── eval_assistance.py  # Evaluate diffusion-assisted control
├── data_collection/          # SAC training and demonstration collection
│   ├── train_sac.py         # Train SAC expert policies (PFRL-based)
│   ├── generate_data.py     # Rollout experts to collect demos (ReplayBuffer)
│   ├── env/                 # Environment creation and wrappers
│   │   ├── __init__.py      # make_env() - creates LunarLander/maze/BlockPush envs
│   │   ├── lunar_lander.py  # Custom LunarLander with task variants (v1-v5)
│   │   └── assistance_wrappers.py  # SplitObsWrapper (pilot vs copilot observations)
│   └── config/
│       └── sweep/           # Sweep configs for data collection (params_proto)
├── baseline/                # Baseline methods (e.g., behavioral cloning)
│   └── train_bc.py
├── config/
│   ├── default_args.py      # Global Args class (paths, hyperparams)
│   └── sweep/               # Sweep configs for diffusion training
└── utils/                   # Multiprocessing patches, etc.
```

### Key Concepts

**1. Three-phase pipeline:**
- **Phase 1 - Train SAC expert:** `python -m diffusha.data_collection.train_sac --env-name LunarLander-v1 --steps 3000000`
  - Trains expert policy using Soft Actor-Critic (PFRL library)
  - Saves checkpoints to `$DATA_DIR/sac/<env>/<wandb-id>/`

- **Phase 2 - Collect demonstrations:** `python -m diffusha.data_collection.generate_data -l 0 --sweep-file <sweep.jsonl>`
  - Rolls out expert to collect state-action pairs
  - Saves to `ReplayBuffer` format: `$DATA_DIR/replay/<env>/randp_<noise>/`
  - `randp`: probability expert takes random action (for noisy demos)

- **Phase 3 - Train diffusion model:** `python -m diffusha.diffusion.train --sweep-file <sweep.jsonl> -l 0`
  - Trains denoising diffusion model on (state, action) pairs
  - Saves checkpoints to `$DATA_DIR/ddpm/<env>/`

**2. Partial Diffusion (the core DiffuSHA mechanism):**
- At inference, given human action `a_human` and state `s`:
  1. Forward diffuse `[s, a_human]` for **k steps** (where k = γ * T, γ ∈ [0,1])
  2. Reverse diffuse for k steps with state `s` held constant (hard conditioning)
  3. Return denoised action
- Implementation: `diffusha/actor/assistive.py:DiffusionAssistedActor._diffusion_cond_sample()`
- Key parameter: `fwd_diff_ratio` (γ) controls assistance level
  - γ=0.0: pure teleop (no assistance)
  - γ=0.4: typical shared autonomy setting
  - γ=1.0: full diffusion (ignores human)

**3. Pilot vs Copilot observations (LunarLander only):**
- **Pilot obs:** includes goal information (e.g., target landing pad location)
- **Copilot obs:** excludes goal (must infer from pilot's actions)
- Created by `LunarLanderSplitObsWrapper` when `split_obs=True` in `make_env()`
- Observation becomes dict: `{'pilot': obs_with_goal, 'copilot': obs_without_goal}`
- Diffusion model trains on copilot obs; human provides pilot obs

**4. Sweep files (params_proto):**
- Define hyperparameter grids as `.jsonl` files
- Example: `diffusha/data_collection/config/sweep/sweep_lander-v1.jsonl`
  ```json
  {"env_name": "LunarLander-v1", "valid_return_threshold": 800, "randp": 0.0}
  {"env_name": "LunarLander-v1", "valid_return_threshold": 800, "randp": 0.2}
  ```
- Usage: `python -m <module> --sweep-file <file.jsonl> -l <line_number>`
- Line number selects which config to run (for parallel sweeps)

**5. ReplayBuffer format:**
- Stores transitions as `(state, action, q_val)` in chunked PyTorch files
- Location: `diffusha/data_collection/generate_data.py:ReplayBuffer`
- Files: `<replay_dir>/chunk_<N>.pt`
- Each chunk: `np.ndarray` shape `(chunk_size, state_dim + action_dim + 1)`

**6. Actor hierarchy:**
- `Actor` (base): abstract interface with `act(obs)` and `batch_act(obss)`
- `ExpertActor`: wraps PFRL SAC agent
- `NoisyActor`: wraps actor, randomly replaces actions with noise (eps-greedy)
- `LaggyActor`: wraps actor, repeats previous action with probability p
- `DiffusionAssistedActor`: wraps behavioral actor + applies partial diffusion

### Environment Variants

**LunarLander (custom, in `diffusha/data_collection/env/lunar_lander.py`):**
- `LunarLander-v1`: **Reach task** (hover above target, don't land)
- `LunarLander-v2`: Landing task, fixed helipad
- `LunarLander-v4`: Float task
- `LunarLander-v5`: Landing task, **randomized helipad** each episode
- Observation: 8-dim (pos, vel, angle, angular_vel, leg_contact_flags)

**BlockPush:**
- `BlockPushMultimodal-v1`: Push block to one of two goals
- Uses `user_goal` parameter to specify target
- "Flip-replay" trick: mirror demonstrations to other goal

**PointMaze (via d4rl):**
- `maze2d-simple-two-goals-v0`: Navigate to one of two goals

### Common Commands (Current System)

```bash
# Setup environment
export DATA_DIR=/data
export OUT_DIR=/outdir

# Quick evaluation (uses pretrained models from $DATA_DIR)
python -m diffusha.diffusion.evaluation.eval_assistance \
  --env-name LunarLander-v1 \
  --out-dir $OUT_DIR \
  --save-video

# Train SAC expert
python -m diffusha.data_collection.train_sac \
  --env-name LunarLander-v1 \
  --steps 3000000

# Collect demonstrations
python -m diffusha.data_collection.generate_data \
  -l 0 \
  --sweep-file diffusha/data_collection/config/sweep/sweep_lander-v1.jsonl

# Train diffusion model
python -m diffusha.diffusion.train \
  --sweep-file diffusha/config/sweep/sweep-lunarlander.jsonl \
  -l 0  # line 0 = LunarLander-v1 (reacher)
```

### DiffusionModel Architecture

**Core class: `diffusha/diffusion/ddpm.py:DiffusionModel`**
- Input: `[state, action]` concatenated (e.g., 8+2=10 dim for LunarLander)
- Conditioning: state dimensions are held constant during reverse diffusion
- Score network: `ConditionalModel` (transformer-based, from `models.py`)
- Beta schedule: configurable (sigmoid, linear, cosine)
- Timesteps: typically 50 (configurable via `Args.num_diffusion_steps`)

**Training loop (`Trainer` in `ddpm.py`):**
1. Sample batch from replay buffer DataLoader
2. Sample random timesteps t
3. Forward diffuse: `x_t = √(α̅_t) x_0 + √(1-α̅_t) ε`
4. Predict noise: `ε_θ(x_t, t)`
5. Loss: MSE between predicted and actual noise

**Inference (conditional sampling):**
- See `DiffusionAssistedActor._diffusion_cond_sample()` in `assistive.py`
- Hard conditioning: replace state dims with crisp observation at each reverse step

### Configuration System

**Global args: `diffusha/config/default_args.py:Args`**
- Uses `params_proto` library for declarative configs
- Key fields:
  - `env_name`: e.g., 'LunarLander-v1'
  - `num_diffusion_steps`: 50
  - `beta_schedule`: 'sigmoid'
  - `fwd_diff_ratio`: 0.4 (γ for shared autonomy)
  - Data directories: `lunarlander_data_dir`, `blockpush_data_dir`, etc.
  - Surrogate pilot params: `laggy_actor_repeat_prob`, `noisy_actor_eps`

**Data collection args: `diffusha/data_collection/config/default_args.py:DCArgs`**
- `randp`: random action probability during demo collection
- `valid_return_threshold`: minimum episode return to save trajectory
- `sac_model_dir`: where to load/save SAC checkpoints

### Docker Usage

```bash
# Pull and run (GPU required)
docker run -it --gpus all \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -v $CODE_DIR:/code \
  -v $DATA_DIR:/data \
  -v $OUT_DIR:/outdir \
  --workdir /code \
  ripl/diffusion-for-shared-autonomy bash

# Inside container, Python 3.9 + PyTorch 1.12 + MuJoCo 2.1 + Box2D
```

### Data Directory Structure (Expected)

```
data-dir/
├── experts/                   # Pretrained SAC checkpoints
│   ├── lunarlander/
│   │   ├── v1/               # LunarLander-v1 (reacher) expert
│   │   └── v5/               # LunarLander-v5 (lander) expert
│   └── blockpush/
├── replay/                    # Demonstration datasets
│   ├── lunarlander/
│   │   ├── v1/randp_0.0/     # Expert demos (no noise)
│   │   └── v5/randp_0.0/
│   └── blockpush/
│       ├── target/randp_0.0/
│       └── target-flipped/randp_0.0/
└── ddpm/                      # Trained diffusion checkpoints
    ├── lunarlander_v1_ddpm.pt
    └── lunarlander_v5_ddpm.pt
```

---

## Collision Extension Project

The following sections describe a **planned extension** to add obstacle-aware navigation and compare DiffuSHA against open-loop MPC baselines.

### Extension Overview

This project extends the base DiffuSHA codebase to demonstrate that collision-free motion planning
requires genuine world knowledge — which open-loop MPC lacks — while shared autonomy via diffusion
naturally acquires it. We run everything on **LunarLander-v5** with a custom obstacle extension.

### Extension Scientific Claim

Four conditions are compared on `LunarLanderObstacle-v5` (a new collision-extended environment):

| # | Condition | Key property |
|---|-----------|-------------|
| 1 | **Pure teleop** | Noisy pilot, γ=0, no assistance |
| 2 | **Open-loop MPC** | CEM planner, dynamics trained on obstacle-free demos — deliberately blind to obstacles |
| 3 | **Conditioned diffusion** | Diffusion policy conditioned on binarized intent labels + MPC trajectory proposals |
| 4 | **DiffuSHA (shared autonomy)** | Partial diffusion γ=0.4, full human+expert blend, obstacle-aware from training |

Expected ordering: teleop < MPC < conditioned diffusion < DiffuSHA

---

### Repository Structure After Extension

```
diffusion-for-shared-autonomy/
├── diffusha/
│   ├── envs/                         ← NEW DIRECTORY for extension
│   │   ├── __init__.py               ← register LunarLanderObstacle-v5 here
│   │   └── lunar_lander_obstacle.py  ← NEW: collision-extended environment
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── mpc_cem.py                ← NEW: CEM-based MPC agent
│   │   └── train_dynamics.py         ← NEW: train MPC dynamics model on obs-free demos
│   ├── data_collection/
│   │   ├── train_sac.py              ← existing, rerun on new env
│   │   ├── generate_data.py          ← existing, rerun on new env
│   │   └── config/sweep/
│   │       └── sweep_lander_obstacle.jsonl  ← NEW: sweep config for new env
│   ├── diffusion/
│   │   ├── train.py                  ← existing, modify conditioning input
│   │   ├── dataset.py                ← existing, add binary intent label field
│   │   └── evaluation/
│   │       ├── eval_assistance.py    ← existing
│   │       └── eval_ablation.py      ← NEW: unified 4-condition evaluator
│   └── config/
│       └── sweep/
│           └── sweep_lander_obstacle.jsonl  ← NEW: diffusion training sweep
├── demo/
│   └── visualize_ablation.py         ← NEW: interactive results visualization
├── data-dir/                         ← pretrained models + replay buffers
├── output-dir/                       ← eval results, videos
└── CLAUDE.md                         ← this file
```

---

## Extension Setup

### 1. Clone and download pretrained assets (same as base system)

```bash
git clone https://github.com/ripl/diffusion-for-shared-autonomy.git
cd diffusion-for-shared-autonomy
mkdir output-dir

export CODE_DIR=$(pwd)
export DATA_DIR=$CODE_DIR/data-dir
export OUT_DIR=$CODE_DIR/output-dir

# Download pretrained SAC experts + diffusion checkpoints + replay buffers
wget https://dl.ttic.edu/diffusion-for-shared-autonomy.tar.gz
tar xfvz diffusion-for-shared-autonomy.tar.gz
mv hosted_data/* data-dir/
```

### 2. Run via Docker (recommended)

```bash
docker run -it --gpus all \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -v $CODE_DIR:/code \
  -v $DATA_DIR:/data \
  -v $OUT_DIR:/outdir \
  --workdir /code \
  ripl/diffusion-for-shared-autonomy bash
```

Inside the container, install the new dependencies:

```bash
pip install gymnasium[box2d]   # ensure Box2D is available
pip install torch torchvision  # already in image, but confirm
```

### 3. Disable WandB (optional)

```bash
export WANDB_MODE=disable
```

---

## Implementation tasks

Work through these in order — each phase depends on the previous one.

---

### Phase 1 — Collision environment

**File:** `diffusha/envs/lunar_lander_obstacle.py`

Subclass `gymnasium.envs.box2d.lunar_lander.LunarLander`. Key requirements:

- **Obstacle placement:** In `reset()`, spawn `NUM_OBSTACLES=4` static Box2D bodies at random
  positions in the region x∈[-1.2, 1.2], y∈[0.3, 1.4]. Obstacle shape: axis-aligned box,
  half-extents (0.8, 0.4) in Box2D units. Tag each body with `userData = {'obstacle': True}`.
- **Collision detection:** After each `super().step()`, iterate `self.lander.contacts`. If any
  touching contact involves a body tagged as obstacle, set `terminated=True` and add
  `COLLISION_PENALTY = -100.0` to reward. Set `info['collision'] = True`.
- **Lidar observation augmentation:** Cast `N_LIDAR_RAYS=8` rays from the lander centroid at
  evenly-spaced angles [0, 2π). For each ray, compute normalized distance [0, 1] to nearest
  obstacle body. Append to base observation. Final obs dim: 8 (base) + 8 (lidar) = **16**.
- **Binary intent label:** Do NOT include this in the environment observation directly. The eval
  harness will compute and pass it as conditioning context separately.

Register in `diffusha/envs/__init__.py`:
```python
from gymnasium.envs.registration import register
register(
    id='LunarLanderObstacle-v5',
    entry_point='diffusha.envs.lunar_lander_obstacle:LunarLanderObstacle',
    max_episode_steps=1000,
)
```

Also import this module from `diffusha/__init__.py` so registration runs on import.

**Validation:**
```bash
python -c "
import gymnasium as gym
import diffusha  # triggers registration
env = gym.make('LunarLanderObstacle-v5')
obs, _ = env.reset()
print('obs shape:', obs.shape)   # expect (16,)
for _ in range(50):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    if info.get('collision'):
        print('collision detected — env working correctly')
        break
"
```

---

### Phase 2 — MPC baseline (CEM)

**Files:** `diffusha/baselines/mpc_cem.py`, `diffusha/baselines/train_dynamics.py`

#### 2a. Dynamics model (`train_dynamics.py`)

Train a small MLP on the **existing obstacle-free** replay buffer
(`$DATA_DIR/replay/lunarlander/v5/randp_0.0`). This is intentional — the MPC's failure
to avoid obstacles comes from this training distribution mismatch.

- Input: `[s_t (8-dim), a_t (2-dim)]`, output: `delta_s = s_{t+1} - s_t (8-dim)`
- Architecture: 3-layer MLP, hidden=256, SiLU activations
- Loss: MSE on delta_s
- Train for 100k gradient steps, batch size 512
- Save checkpoint to `$DATA_DIR/mpc_dynamics/lunarlander_v5.pt`

Important: the dynamics model is trained on **8-dim obs only** (no lidar augmentation).
At eval time, the MPC strips the lidar dimensions before querying the model. This is how
it becomes blind to obstacles.

#### 2b. CEM planner (`mpc_cem.py`)

```
class MPCAgent:
    horizon H = 10
    n_samples N = 200
    n_elites K = 20
    cem_iterations = 5
    action_dim = 2, clipped to [-1, 1]
```

At each timestep:
1. Strip lidar from obs: `s = obs[:8]`
2. Run CEM: sample N plans of shape (H, 2), rollout with dynamics model, score with reward fn
3. Reward fn: `r(s) = -distance_to_goal(s)` where goal position comes from the env's current
   landing pad location (accessible via `env.unwrapped.helipad_x1`, `helipad_x2`)
4. Select K elites, refit Gaussian, repeat for `cem_iterations` steps
5. Return `mu[0]` — the first action of the mean plan

```bash
# Train dynamics model
python -m diffusha.baselines.train_dynamics \
  --replay-dir /data/replay/lunarlander/v5/randp_0.0 \
  --out-path /data/mpc_dynamics/lunarlander_v5.pt

# Quick sanity check
python -m diffusha.baselines.mpc_cem --test-env LunarLanderObstacle-v5 --n-episodes 5
```

---

### Phase 3 — Expert SAC for obstacle environment

Retrain SAC from scratch on `LunarLanderObstacle-v5`. The new expert must learn to
navigate around obstacles — this becomes the demonstration dataset for DiffuSHA.

```bash
python -m diffusha.data_collection.train_sac \
  --env-name LunarLanderObstacle-v5 \
  --steps 3000000 \
  --save-dir /data/experts/lunarlander/obstacle_v5
```

Expected training time: ~4-6 hours on a single GPU.

Once trained, verify the expert achieves >80% success rate with collision rate <5%:

```bash
python -m diffusha.data_collection.train_sac \
  --env-name LunarLanderObstacle-v5 --eval-only \
  --checkpoint /data/experts/lunarlander/obstacle_v5/<run_id>/final.pt
```

---

### Phase 4 — Collect obstacle-aware demonstrations

```bash
python -m diffusha.data_collection.generate_data \
  -l 0 \
  --sweep-file diffusha/data_collection/config/sweep/sweep_lander_obstacle.jsonl
```

Create `sweep_lander_obstacle.jsonl`:
```json
{"env_name": "LunarLanderObstacle-v5", "valid_return_threshold": 150, "randp": 0.0}
```

Save demos to: `$DATA_DIR/replay/lunarlander/obstacle_v5/randp_0.0`

Each trajectory should include: `(obs_16dim, action_2dim, reward, done, info)`.
The binary intent label is derived at training time from `obs[0]` (x-position of lander
relative to goal) — see Phase 5.

---

### Phase 5 — Modify diffusion conditioning

**File:** `diffusha/diffusion/train.py` and the score network architecture.

The goal is to add two new conditioning signals to the score network:

**a. Binary intent label** (1 dim)
- Derived from the pilot's current action direction: `label = int(action[0] > 0)` (right=1, left=0)
- Alternatively: `label = int(obs[0] > helipad_center_x)` — lander is right of center → goal is right
- Encode with a learned embedding: `nn.Embedding(2, d_model)` added to the score network input

**b. MPC action proposal** (2 dims)
- At each training step, run one step of MPC on the current state (with frozen dynamics model)
  to get a proposal action
- Concatenate to the conditioning vector
- This teaches the diffusion model to "see" what the open-loop planner would do and correct it

**Modified conditioning vector:**
```
c_orig  = state (16-dim)
c_new   = [state (16-dim) | intent_embedding (d) | mpc_proposal (2-dim)]
```

Only the score network input changes. The diffusion process (forward/reverse, γ parameter)
is identical to the original DiffuSHA.

**Training:**
```bash
python -m diffusha.diffusion.train \
  --sweep-file diffusha/config/sweep/sweep_lander_obstacle.jsonl -l 0
```

Create `sweep_lander_obstacle.jsonl`:
```json
{"env_name": "LunarLanderObstacle-v5", "conditioning": "annotated", "gamma": 0.4}
```

Two model checkpoints will be produced:
- `conditioned_diffusion.pt` — for Condition 3 (intent + MPC proposals)
- `diffusha.pt` — standard DiffuSHA at γ=0.4 (for Condition 4, retrained on obstacle demos)

---

### Phase 6 — Unified evaluation harness

**File:** `diffusha/diffusion/evaluation/eval_ablation.py`

This script runs all four conditions and collects structured metrics.

```python
CONDITIONS = {
    'teleop':          {'mode': 'teleop',     'gamma': 0.0},
    'mpc':             {'mode': 'mpc',        'gamma': None},
    'conditioned':     {'mode': 'conditioned','gamma': 0.4},
    'diffusha':        {'mode': 'diffusha',   'gamma': 0.4},
}

METRICS = [
    'success_rate',      # landed at goal pad without collision
    'collision_rate',    # hit an obstacle
    'crash_rate',        # hit terrain (non-goal)
    'timeout_rate',      # episode truncated at max steps
    'goal_fidelity',     # landed within 0.5 units of pilot's intended zone
    'mean_clearance',    # mean min-distance to nearest obstacle over trajectory
    'episode_return',    # total undiscounted reward
]
```

Run with:
```bash
python -m diffusha.diffusion.evaluation.eval_ablation \
  --env-name LunarLanderObstacle-v5 \
  --n-seeds 30 \
  --n-episodes 10 \
  --pilot-type noisy \
  --pilot-noise 0.3 \
  --out-dir /outdir/ablation \
  --save-video
```

Output: `ablation_results.json` with per-condition, per-seed statistics.

**Surrogate pilot:** Use the existing `NoisyPilot` wrapper from the original codebase
(`diffusha/diffusion/evaluation/`) with `p_noisy=0.3`. This corrupts 30% of actions with
uniform random samples — simulating a human who sometimes makes imprecise inputs.

---

### Phase 7 — Visualization

**File:** `demo/visualize_ablation.py`

Generate three figures from `ablation_results.json`:

**Figure 1 — Grouped bar chart** (primary results figure)
- x-axis: 4 conditions
- 4 grouped bars per condition: success / collision / crash / timeout rates
- Error bars: 95% CI across 30 seeds
- Save as `results_bar_chart.pdf` and `results_bar_chart.png`

**Figure 2 — Trajectory overlay** (intuition figure)
- 2×2 grid, one panel per condition
- Each panel shows 10 overlaid trajectories from a representative seed
- Obstacles shown as red rectangles, goal pad as green bar
- Color encodes outcome: green=success, red=collision, orange=crash, gray=timeout
- Save as `trajectory_overlay.pdf`

**Figure 3 — γ sweep for DiffuSHA**
- x-axis: γ ∈ [0.0, 1.0]
- y-axis: success rate and collision rate (dual axis)
- Shows that γ=0.4 is the sweet spot; γ→1.0 loses goal fidelity
- Overlaid horizontal lines for teleop and MPC baselines
- Save as `gamma_sweep.pdf`

```bash
python demo/visualize_ablation.py \
  --results /outdir/ablation/ablation_results.json \
  --out-dir /outdir/figures
```

---

## Key design decisions & rationale

### Why MPC fails on the collision env

The MPC dynamics model is trained on the **original obstacle-free replay buffer** from `data-dir/`.
This is intentional — it represents a realistic failure mode where a planner's world model is
stale or incomplete. At eval time on `LunarLanderObstacle-v5`, the MPC:
1. Plans trajectories that are optimal under its internal model
2. Executes them confidently into obstacles
3. Has no mechanism to update its model at test time

This is the open-loop failure we want to demonstrate.

### Why DiffuSHA avoids obstacles

DiffuSHA is retrained on obstacle-aware demonstrations (Phase 3-4). The diffusion model
implicitly learns the distribution of obstacle-avoiding trajectories. At inference time,
even with a noisy pilot, the partial reverse diffusion "pulls" the human's action toward
the expert manifold — which includes obstacle avoidance. The lidar rays in the observation
give the model the information it needs; the diffusion prior does the rest.

### Why Condition 3 (conditioned diffusion) is intermediate

The conditioned diffusion model receives MPC proposals as part of its conditioning.
Since the MPC proposals are obstacle-blind, some of this incorrect prior leaks into
the diffusion model's corrections. It still performs better than pure MPC (because the
diffusion prior partially overrides bad proposals) but worse than DiffuSHA (which has
no MPC bias at all).

### Binary intent label derivation

```python
def get_intent_label(obs, env):
    """1 if pilot is heading toward right goal zone, 0 for left."""
    lander_x = obs[0]                          # normalized x position
    goal_x = (env.unwrapped.helipad_x1 +
               env.unwrapped.helipad_x2) / 2   # landing pad center
    # Use pilot action x-component as proxy for intent direction
    # Stable label: use lander position relative to goal
    return int(lander_x > goal_x)
```

---

## Experiment tracking

All runs log to WandB under project `diffusha-collision-ablation`.

Key run tags:
- `condition`: one of `{teleop, mpc, conditioned, diffusha}`
- `env`: `LunarLanderObstacle-v5`
- `pilot`: `noisy_p0.3`
- `phase`: `{sac_training, demo_collection, diffusion_training, evaluation}`

To compare conditions after eval:
```bash
wandb login
# View at: https://wandb.ai/<your-entity>/diffusha-collision-ablation
```

---

## Expected results

Based on the original paper and the deliberate design of the MPC failure mode:

| Condition | Success ↑ | Collision ↓ | Goal fidelity ↑ |
|-----------|-----------|------------|----------------|
| Teleop | ~30% | ~40% | ~85% |
| MPC | ~45% | ~35% | ~50% (ignores human) |
| Cond. diffusion | ~55% | ~20% | ~70% |
| DiffuSHA | ~75% | ~8% | ~80% |

The clearest signal will be **collision rate** — MPC should have the highest despite
looking "smart". This is the headline result.

---

## Troubleshooting

**Box2D import errors:**
```bash
pip install gymnasium[box2d] swig
# or inside Docker:
apt-get install -y swig && pip install box2d-py
```

**`helipad_x1` attribute not found:**
The attribute names may differ across Gymnasium versions. Inspect with:
```python
env = gym.make('LunarLanderObstacle-v5')
obs, _ = env.reset()
print(dir(env.unwrapped))  # find landing pad position attributes
```

**Dynamics model diverges during training:**
Normalize observations before training. The original replay buffer stores raw observations
in approximately [-2, 2] range. Add a running mean/std normalization layer to the dynamics
model input.

**MPC is too slow (>1s per step):**
Reduce `n_samples` to 100 or `horizon` to 8. For the demo, speed matters more than
optimal MPC performance — MPC is the baseline that fails, not the one we're optimizing.

**DiffuSHA checkpoint not loading on obstacle env:**
The original checkpoint expects 8-dim obs. The new env has 16-dim obs. You must retrain
DiffuSHA from scratch on obstacle-aware demos (Phase 3-5). Do not try to fine-tune the
original checkpoint — the observation space change is incompatible.

---

## Citation

```bibtex
@inproceedings{yoneda2023diffusha,
    author    = {Takuma Yoneda and Luzhe Sun and Ge Yang and
                 Bradly C. Stadie and Matthew R. Walter},
    title     = {To the Noise and Back: Diffusion for Shared Autonomy},
    booktitle = {Robotics: Science and Systems XIX},
    year      = {2023}
}
```
