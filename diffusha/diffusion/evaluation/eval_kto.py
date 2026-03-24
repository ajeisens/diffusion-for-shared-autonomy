"""
Post-training evaluation harness for KTO LunarLander diffusion models.

Evaluates four conditions and writes a JSON results file:
  - teleop:      Diffusion model trained on human teleop data (BC)
  - heuristic:   Diffusion model trained on heuristic data (BC)
  - kto:         Diffusion model trained on KTO planner data (BC)
  - collision_conditioned: Quality-conditioned model (π0.6 style)
                           evaluated at multiple guidance scales

Metrics per condition (over N episodes):
  - success_rate:   fraction landing on pad
  - collision_rate: fraction hitting an obstacle
  - crash_rate:     fraction hitting terrain / going out of bounds
  - timeout_rate:   fraction hitting step limit

Usage:
    python -m diffusha.diffusion.evaluation.eval_kto \\
        --teleop_model    /data/ddpm/diffusha/<run_id>/step_00099999.pt \\
        --heuristic_model /data/ddpm/diffusha/<run_id>/step_00099999.pt \\
        --kto_model       /data/ddpm/diffusha/<run_id>/step_00099999.pt \\
        --cc_model        /data/ddpm/diffusha/<run_id>/step_00099999.pt \\
        --n_episodes 100 \\
        --output results_kto.json

    # Guidance scale sweep for collision-conditioned model only
    python -m diffusha.diffusion.evaluation.eval_kto \\
        --cc_model /data/ddpm/diffusha/<run_id>/step_00099999.pt \\
        --guidance_scales 0.5 1.0 1.5 2.0 3.0 \\
        --n_episodes 100 \\
        --output results_guidance_sweep.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from diffusha.envs.lunar_lander_kto import LunarLanderKTO
from diffusha.diffusion.ddpm import DiffusionCore, DiffusionModel
from diffusha.config.default_args import Args


# ── Observation/action constants (KTO env) ──────────────────────────────
COPILOT_OBS_DIM = 6   # [x, y, θ, vx, vy, ω]
ACT_DIM = 2           # [main_thrust, side_thrust]


# ── Model loading ────────────────────────────────────────────────────────

def load_model(
    checkpoint_path: str,
    quality_cond: bool = False,
) -> DiffusionModel:
    """Load a DiffusionModel from a checkpoint .pt file.

    Args:
        checkpoint_path: Path to checkpoint saved by DiffusionModel.save_model().
        quality_cond: True for collision-conditioned models (9-dim input, cond_dim=7).
    """
    quality_dim = 1 if quality_cond else 0
    input_size = COPILOT_OBS_DIM + quality_dim + ACT_DIM
    cond_dim = COPILOT_OBS_DIM + quality_dim

    diffusion = DiffusionModel(
        diffusion_core=DiffusionCore(),
        num_diffusion_steps=Args.num_diffusion_steps,
        input_size=input_size,
        beta_schedule=Args.beta_schedule,
        beta_min=Args.beta_min,
        beta_max=Args.beta_max,
        cond_dim=cond_dim,
    )

    ckpt = torch.load(checkpoint_path, map_location=diffusion.device)
    # Support both EMA and raw model weights
    if 'ema' in ckpt:
        diffusion.model.load_state_dict(ckpt['ema'])
    else:
        diffusion.model.load_state_dict(ckpt['model'])
    diffusion.model.eval()

    print(f"Loaded {'quality-conditioned' if quality_cond else 'BC'} model "
          f"from {checkpoint_path} "
          f"(input={input_size}, cond_dim={cond_dim})")
    return diffusion


# ── Action sampling ──────────────────────────────────────────────────────

def sample_action(
    diffusion: DiffusionModel,
    obs: np.ndarray,
    quality_cond: bool = False,
    guidance_scale: float = 1.0,
) -> np.ndarray:
    """Sample an action from the diffusion model given an observation.

    For quality-conditioned models, conditions on quality=1 (collision-free)
    with optional CFG guidance scale.

    Args:
        diffusion:      Loaded DiffusionModel.
        obs:            Raw 15-dim KTO observation.
        quality_cond:   True for collision-conditioned models.
        guidance_scale: CFG blending coefficient λ (only used when quality_cond=True).

    Returns:
        action: np.ndarray of shape (2,), [main_thrust, side_thrust].
    """
    copilot_obs = obs[:COPILOT_OBS_DIM]

    if quality_cond:
        # Condition on quality=1 (no collision)
        cond_good = np.concatenate([copilot_obs, [1.0]]).astype(np.float32)
        cond_tensor = torch.tensor(cond_good)

        if guidance_scale != 1.0:
            # CFG: also provide unconditional (quality=0) condition
            cond_null = np.concatenate([copilot_obs, [0.0]]).astype(np.float32)
            uncond_tensor = torch.tensor(cond_null)
        else:
            uncond_tensor = None
    else:
        cond_tensor = torch.tensor(copilot_obs.astype(np.float32))
        uncond_tensor = None

    input_size = COPILOT_OBS_DIM + (1 if quality_cond else 0) + ACT_DIM
    shape = torch.Size([input_size])

    x, _ = diffusion.p_sample_loop(
        shape,
        cond=cond_tensor,
        uncond=uncond_tensor,
        guidance_scale=guidance_scale,
        naive_cond=True,
    )

    # Extract action from the final dims
    cond_dim = COPILOT_OBS_DIM + (1 if quality_cond else 0)
    action = x[..., cond_dim:].detach().cpu().numpy()
    action = np.clip(action, [-1.0, -1.0], [1.0, 1.0])
    action[0] = np.clip(action[0], 0.0, 1.0)  # main_thrust in [0, 1]

    return action.astype(np.float32)


# ── Episode evaluation ───────────────────────────────────────────────────

def eval_model(
    diffusion: DiffusionModel,
    n_episodes: int = 100,
    seed: int = 0,
    quality_cond: bool = False,
    guidance_scale: float = 1.0,
    verbose: bool = False,
) -> Dict:
    """Run n_episodes and return aggregate metrics.

    Returns:
        dict with keys: success_rate, collision_rate, crash_rate, timeout_rate,
                        n_episodes, guidance_scale (if quality_cond).
    """
    env = LunarLanderKTO()
    np.random.seed(seed)

    outcomes = {'success': 0, 'collision': 0, 'crashed': 0, 'timeout': 0}

    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        done = False

        while not done:
            action = sample_action(
                diffusion, obs,
                quality_cond=quality_cond,
                guidance_scale=guidance_scale,
            )
            obs, reward, done, info = env.step(action)

        goal = info.get('goal', 'timeout')
        if goal == 'landed':
            outcomes['success'] += 1
        elif goal == 'obstacle-collision':
            outcomes['collision'] += 1
        elif goal in ('terrain-crash', 'out-of-bounds'):
            outcomes['crashed'] += 1
        else:
            outcomes['timeout'] += 1

        if verbose and (ep + 1) % 20 == 0:
            print(f"  [{ep+1}/{n_episodes}] "
                  f"success={outcomes['success']} "
                  f"collision={outcomes['collision']} "
                  f"crash={outcomes['crashed']} "
                  f"timeout={outcomes['timeout']}")

    result = {
        'n_episodes': n_episodes,
        'success_rate':   outcomes['success']   / n_episodes,
        'collision_rate': outcomes['collision'] / n_episodes,
        'crash_rate':     outcomes['crashed']   / n_episodes,
        'timeout_rate':   outcomes['timeout']   / n_episodes,
    }
    if quality_cond:
        result['guidance_scale'] = guidance_scale

    return result


def print_result(label: str, result: Dict) -> None:
    gs = f"  guidance_scale={result['guidance_scale']:.1f}" if 'guidance_scale' in result else ""
    print(f"  {label}{gs}")
    print(f"    success:   {result['success_rate']:.1%}")
    print(f"    collision: {result['collision_rate']:.1%}")
    print(f"    crash:     {result['crash_rate']:.1%}")
    print(f"    timeout:   {result['timeout_rate']:.1%}")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate KTO LunarLander diffusion models"
    )
    parser.add_argument('--teleop_model',    type=str, default=None)
    parser.add_argument('--heuristic_model', type=str, default=None)
    parser.add_argument('--kto_model',       type=str, default=None)
    parser.add_argument('--cc_model',        type=str, default=None,
                        help="Collision-conditioned model checkpoint")
    parser.add_argument('--n_episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--guidance_scales', type=float, nargs='+',
                        default=[1.0, 1.5, 2.0, 3.0],
                        help="Guidance scales to sweep for the CC model")
    parser.add_argument('--output', type=str, default='results_kto.json')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    results = {}

    # Standard BC models
    for label, path in [
        ('teleop',    args.teleop_model),
        ('heuristic', args.heuristic_model),
        ('kto',       args.kto_model),
    ]:
        if path is None:
            continue
        print(f"\nEvaluating {label} model ({args.n_episodes} episodes)...")
        diffusion = load_model(path, quality_cond=False)
        result = eval_model(
            diffusion,
            n_episodes=args.n_episodes,
            seed=args.seed,
            quality_cond=False,
            verbose=args.verbose,
        )
        results[label] = result
        print_result(label, result)

    # Collision-conditioned model: sweep guidance scales
    if args.cc_model is not None:
        print(f"\nEvaluating collision-conditioned model "
              f"(guidance_scales={args.guidance_scales}, {args.n_episodes} ep each)...")
        diffusion = load_model(args.cc_model, quality_cond=True)
        cc_results = []
        for gs in args.guidance_scales:
            print(f"  guidance_scale={gs:.1f}...")
            result = eval_model(
                diffusion,
                n_episodes=args.n_episodes,
                seed=args.seed,
                quality_cond=True,
                guidance_scale=gs,
                verbose=args.verbose,
            )
            cc_results.append(result)
            print_result('collision_conditioned', result)
        results['collision_conditioned'] = cc_results

    # Write JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == '__main__':
    main()
