#!/usr/bin/env python3
"""
Lunar Lander Demo with KTO Trajectory Optimization

Integrates poslathian's Drake-based KTO planner with episode recording system.
Three control modes: Teleop (human), Heuristic (PID), and KTO (trajectory optimization).

Controls:
    Arrow Up: Main engine (Teleop mode)
    Arrow Left: Rotate left / partial side thrust (Teleop mode)
    Arrow Right: Rotate right / partial side thrust (Teleop mode)
    R: Reset
    Q/Escape: Quit
    1: Teleop mode (human control)
    2: Heuristic mode (automatic PID control)
    3: KTO mode (Drake trajectory optimization)
    F: Cycle failure level (heuristic mode only)
    E: Toggle recording

Usage:
    python demo/interactive/play.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pygame
import numpy as np
from enum import Enum
from typing import Optional, Dict, Tuple

# Import KTO environment
from diffusha.envs.lunar_lander_kto import LunarLanderKTO

# Import KTO planner and renderer
try:
    from diffusha.planning.async_planner import AsyncKTOPlanner, DRAKE_AVAILABLE
    from diffusha.rendering.kto_renderer import KTORenderer
    KTO_AVAILABLE = DRAKE_AVAILABLE
except ImportError as e:
    print(f"Warning: KTO components not available: {e}")
    KTO_AVAILABLE = False
    AsyncKTOPlanner = None
    KTORenderer = None

# Import episode recorder (will be copied from simple-human-playable-demo branch)
try:
    from episode_recorder import EpisodeRecorder
except ImportError:
    print("Warning: episode_recorder not found. Recording disabled.")
    EpisodeRecorder = None


def heuristic(env, s):
    """
    Heuristic controller for Lunar Lander (tuned for KTO physics).

    A PID-based controller that uses state feedback to land the lander.
    Gains tuned for KTO simplified rocket physics (no damping).

    Args:
        env: The environment
        s: state [x, y, theta, vx, vy, omega, lidar×8, pad_x] (15-dim for KTO env)

    Returns:
        action: [main_engine_throttle, side_engine_throttle] normalized to [0,1] × [-1,1]
    """
    # Extract state (first 6 elements)
    x, y, theta, vx, vy, omega = s[:6]

    # KTO physics constants (world coordinates)
    _G = 10.0; _MASS = 1.0; _I = 0.2
    _FMAX = 20.0; _SMAX = 5.0; _SARM = 1.0; _PAD_Y = 3.33

    pad_x = s[14]  # = 10.0 world units
    dx = x - pad_x  # lateral offset to pad, world units

    # ---- Angle control ----
    # theta > 0: tilts right → main thrust has leftward component
    # theta < 0: tilts left  → main thrust has rightward component
    # Tilt toward pad + damp lateral velocity
    theta_targ = np.clip(dx * 0.04 + vx * 0.02, -0.4, 0.4)

    # Physically-derived PD gains: omega_n=4 rad/s, critically damped
    # alpha = Fs*arm/I  →  a[1] = alpha_des * I/(SMAX*arm)
    alpha_des = 16.0 * (theta_targ - theta) - 8.0 * omega
    a1 = float(np.clip(alpha_des * _I / (_SMAX * _SARM), -1.0, 1.0))

    # ---- Vertical control ----
    # Proportional descent profile: faster high up, slow near pad
    vy_targ = float(np.clip((_PAD_Y - y) * 0.3, -3.0, -0.2))

    # Gravity compensation + vy tracking
    ay_des = 3.0 * (vy_targ - vy)
    Fm_des = _MASS * (ay_des + _G) / max(abs(np.cos(theta)), 0.5)
    a0 = float(np.clip(Fm_des / _FMAX, 0.0, 1.0))

    # Final approach: straighten up for touchdown
    if y < _PAD_Y + 2.0:
        alpha_des = 16.0 * (0.0 - theta) - 8.0 * omega
        a1 = float(np.clip(alpha_des * _I / (_SMAX * _SARM), -1.0, 1.0))

    # a0 already in [0,1], a1 already in [-1,1]
    return np.array([a0, a1])


class ControlMode(Enum):
    """Control mode enumeration"""
    TELEOP = 1      # Human keyboard control
    HEURISTIC = 2   # Heuristic PID controller
    KTO = 3         # Drake trajectory optimization
    DIFFUSION = 4   # Trained diffusion model (autonomous)
    ASSISTED = 5    # Shared autonomy: keyboard nudges diffusion position 0


# Colors (from poslathian)
SKY = (0, 0, 0)
GROUND = (50, 50, 50)
WHITE = (255, 255, 255)
BLUE = (52, 152, 219)
GREEN = (46, 204, 113)
RED = (231, 76, 60)
ORANGE = (230, 126, 34)
YELLOW = (241, 196, 15)
GRAY = (149, 165, 166)


class LunarLanderPlayer:
    """Lunar Lander player with KTO trajectory optimization"""

    def __init__(self, window_width=600, window_height=400):
        """Initialize player"""
        pygame.init()

        self.window_width = window_width
        self.window_height = window_height
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Lunar Lander - KTO + Manual + Heuristic")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 14)
        self.banner_font = pygame.font.Font(None, 36)

        # Create KTO environment
        self.env = LunarLanderKTO()
        self.obs = None
        self.done = True
        self.episode_return = 0.0
        self.steps = 0
        self.info = {}
        self.time = 0.0

        # Control mode
        self.mode = ControlMode.TELEOP

        # Failure injection for heuristic mode
        self.failure_level = 0
        self.occlusion_rates = {0: 0.0, 1: 0.10, 2: 0.30, 3: 0.50}
        self.occluded_rays = set()

        # KTO planner and renderer
        self.kto_available = KTO_AVAILABLE

        # Renderer is always available (doesn't need Drake)
        if KTORenderer:
            self.renderer = KTORenderer(scale=30.0, viewport_w=window_width, viewport_h=window_height)
        else:
            self.renderer = None

        # Planner only available if Drake is installed
        if self.kto_available:
            self.planner = AsyncKTOPlanner(time_budget=1.0, warmstart_budget=0.33)
        else:
            self.planner = None
            print("KTO mode unavailable (Drake not installed or Python version incompatible)")

        # KTO state
        self.kto_planning = False
        self.kto_plan_times = None
        self.kto_plan = None
        self.kto_constraint_xy = None
        self.kto_solve_time = 0.0
        self.kto_manual_override_step = None

        # Episode recorder
        if EpisodeRecorder:
            self.recorder = EpisodeRecorder(
                save_dir='./recorded_episodes',
                env_name='LunarLanderKTO-v1'
            )
            self.recording_enabled = True
        else:
            self.recorder = None
            self.recording_enabled = False

        # Diffusion model (loaded externally via load_diffusion_model)
        self.diffusion_model = None
        self.diffusion_model_name = 'none'
        self.diffusion_quality_cond = False
        self.diffusion_horizon = 1
        self.diffusion_exec_horizon = 1
        self.diffusion_fwd_diff_steps = 5

        # Receding horizon chunk cache for DIFFUSION mode
        self._diff_chunk_cache = None   # (exec_horizon, 2) or None
        self._diff_step_in_chunk = 0

        # Receding horizon chunk cache for ASSISTED mode
        self._assisted_chunk_cache = None  # (exec_horizon, 2) or None
        self._assisted_step_in_chunk = 0

        # Input state
        self.action = [0.0, 0.0]

        # Rendering
        self.trajectory = []

    def load_diffusion_model(self, checkpoint_path: str, name: str = None,
                             quality_cond: bool = False,
                             horizon: int = 1, exec_horizon: int = 1,
                             fwd_diff_steps: int = 5):
        """Load a trained diffusion model checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint.
            name:            Display name (default: inferred from path).
            quality_cond:    True for collision-conditioned models.
            horizon:         Action chunk length the model was trained with.
            exec_horizon:    Steps to execute per diffusion inference (receding horizon K).
        """
        try:
            from diffusha.diffusion.ddpm import DiffusionModel, DiffusionCore
            from diffusha.config.default_args import Args
            import torch

            # Auto-detect hidden_size from checkpoint weights
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            raw_state = ckpt.get('ema') or ckpt.get('model') or ckpt
            hidden_size = raw_state['lin1.lin.weight'].shape[0]

            quality_dim = 1 if quality_cond else 0
            input_size = 6 + quality_dim + 2 * horizon
            cond_dim   = 6 + quality_dim

            diffusion = DiffusionModel(
                diffusion_core=DiffusionCore(),
                num_diffusion_steps=Args.num_diffusion_steps,
                input_size=input_size,
                beta_schedule=Args.beta_schedule,
                beta_min=Args.beta_min,
                beta_max=Args.beta_max,
                cond_dim=cond_dim,
                hidden_size=hidden_size,
            )
            diffusion.model.load_state_dict(raw_state)
            diffusion.model.eval()

            self.diffusion_model = diffusion
            self.diffusion_model_name = name or Path(checkpoint_path).parent.name
            self.diffusion_quality_cond = quality_cond
            self.diffusion_horizon = horizon
            self.diffusion_exec_horizon = exec_horizon
            self.diffusion_fwd_diff_steps = fwd_diff_steps

            # Reset both chunk caches whenever a new model is loaded
            self._diff_chunk_cache = None
            self._diff_step_in_chunk = 0
            self._assisted_chunk_cache = None
            self._assisted_step_in_chunk = 0

            print(f"Loaded diffusion model: {self.diffusion_model_name} "
                  f"({'quality-conditioned' if quality_cond else 'BC'}) "
                  f"hidden={hidden_size} horizon={horizon} "
                  f"exec_horizon={exec_horizon} fwd_diff_steps={fwd_diff_steps}")
        except Exception as e:
            print(f"Failed to load diffusion model: {e}")
            self.diffusion_model = None

    def _diffusion_action(self) -> list:
        """Sample action from the loaded diffusion model with receding horizon execution.

        Re-runs inference only at chunk boundaries (every exec_horizon steps).
        Between boundaries the pre-computed chunk is replayed, keeping the game
        responsive even when exec_horizon > 1.
        """
        import torch

        # Re-infer at chunk boundary
        if self._diff_chunk_cache is None or self._diff_step_in_chunk == 0:
            copilot_obs = self.obs[:6].astype(np.float32)
            quality_cond = self.diffusion_quality_cond
            horizon = self.diffusion_horizon
            exec_horizon = self.diffusion_exec_horizon

            if quality_cond:
                cond = np.concatenate([copilot_obs, [1.0]])
            else:
                cond = copilot_obs

            cond_tensor = torch.tensor(cond).unsqueeze(0)  # (1, cond_dim)
            cond_dim = 6 + (1 if quality_cond else 0)
            input_size = cond_dim + 2 * horizon
            shape = torch.Size([1, input_size])

            x, _ = self.diffusion_model.p_sample_loop(shape, cond=cond_tensor, naive_cond=True)

            # Extract and reshape the full action block → (horizon, 2)
            act_block = x[0, cond_dim:].detach().cpu().numpy()          # (2*horizon,)
            act_chunk = act_block.reshape(horizon, 2)[:exec_horizon]    # (exec_horizon, 2)
            act_chunk = np.clip(act_chunk, -1.0, 1.0)
            act_chunk[:, 0] = np.clip(act_chunk[:, 0], 0.0, 1.0)       # main_thrust ∈ [0,1]

            self._diff_chunk_cache = act_chunk
            self._diff_step_in_chunk = 0

        action = self._diff_chunk_cache[self._diff_step_in_chunk]
        self._diff_step_in_chunk = (self._diff_step_in_chunk + 1) % self.diffusion_exec_horizon
        return [float(action[0]), float(action[1])]

    def _diffusion_assisted_action(self) -> list:
        """Shared autonomy: keyboard input conditions position 0 of the diffusion chunk.

        At each re-inference boundary:
          - Position 0:      current keyboard action, forward-noised fwd_diff_steps steps
          - Positions 1..H-1: pure Gaussian noise
          - Run fwd_diff_steps reverse denoising steps
          - Cache exec_horizon actions; pop one per env step

        Between boundaries the cached chunk plays out autonomously, so the user
        only needs to provide input once every exec_horizon steps.
        """
        import torch

        if self._assisted_chunk_cache is None or self._assisted_step_in_chunk == 0:
            copilot_obs      = self.obs[:6].astype(np.float32)
            user_act         = np.array(self.action, dtype=np.float32)  # current keyboard state
            horizon          = self.diffusion_horizon
            exec_horizon     = self.diffusion_exec_horizon
            device           = self.diffusion_model.device

            # Clamp to valid timestep range [0, num_diffusion_steps-1]
            max_t = self.diffusion_model.num_diffusion_steps - 1
            k = min(self.diffusion_fwd_diff_steps, max_t)

            obs_tensor = torch.tensor(copilot_obs).unsqueeze(0).to(device)   # (1, 6)

            # Only condition on user action when a key is actively held.
            keys = pygame.key.get_pressed()
            user_is_active = keys[pygame.K_UP] or keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]

            if user_is_active:
                # Partial reverse chain: forward-noise user action to timestep k,
                # scale positions 1..H-1 consistently, run k reverse steps.
                alpha_bar = self.diffusion_model.alphas_bar_sqrt[k] ** 2
                noise_std = float((1.0 - alpha_bar) ** 0.5)

                user_act_tensor = torch.tensor(user_act).unsqueeze(0).to(device)
                state_for_diffuse = torch.cat([obs_tensor, user_act_tensor], dim=1)
                x_k, _ = self.diffusion_model.diffuse(state_for_diffuse.float(), torch.tensor([k]))
                pos0_noise = x_k[:, 6:]                                        # (1, 2)
                rest_noise = torch.randn(1, 2 * (horizon - 1), device=device) * noise_std
                x_init = torch.cat([obs_tensor, pos0_noise, rest_noise], dim=1)

                out, _ = self.diffusion_model.p_sample_loop(
                    shape=x_init.shape,
                    start_x=x_init,
                    cond=obs_tensor,
                    naive_cond=True,
                    start_t=k,
                )
            else:
                # No key held — run full 50-step denoising from pure Gaussian noise,
                # identical to autonomous DIFFUSION mode.
                shape = torch.Size([1, 6 + 2 * horizon])
                out, _ = self.diffusion_model.p_sample_loop(
                    shape=shape,
                    cond=obs_tensor,
                    naive_cond=True,
                )

            # Extract and reshape → (exec_horizon, 2)
            act_block = out[0, 6 : 6 + 2 * horizon].detach().cpu().numpy()
            act_chunk = act_block.reshape(horizon, 2)[:exec_horizon]
            act_chunk = np.clip(act_chunk, -1.0, 1.0)
            act_chunk[:, 0] = np.clip(act_chunk[:, 0], 0.0, 1.0)   # main_thrust ∈ [0,1]

            self._assisted_chunk_cache = act_chunk
            self._assisted_step_in_chunk = 0

        action = self._assisted_chunk_cache[self._assisted_step_in_chunk]
        self._assisted_step_in_chunk = (self._assisted_step_in_chunk + 1) % self.diffusion_exec_horizon
        return [float(action[0]), float(action[1])]

    def reset(self):
        """Reset environment and start new episode"""
        # End previous episode if recording
        if self.recording_enabled and self.recorder and self.obs is not None:
            # Add KTO metadata if applicable
            kto_metadata = None
            if self.mode == ControlMode.KTO and self.kto_plan is not None:
                kto_metadata = {
                    'solve_time': self.kto_solve_time,
                    'n_obstacles': len(self.env.obstacles) if hasattr(self.env, 'obstacles') else 0,
                    'converged': True,  # TODO: Track actual convergence
                    'planned_xy': [(self.kto_plan['x'][i], self.kto_plan['y'][i])
                                   for i in range(0, len(self.kto_plan['x']), 5)] if self.kto_plan else [],
                    'constraint_xy': self.kto_constraint_xy.tolist() if self.kto_constraint_xy is not None else [],
                    'plan_duration': self.kto_plan_times[-1] if self.kto_plan_times is not None else 0.0,
                    'manual_override_at': self.kto_manual_override_step,
                }

            if kto_metadata:
                self.info['kto_metadata'] = kto_metadata

            self.recorder.end_episode()

        # Reset environment
        self.obs = self.env.reset()
        self.done = False
        self.episode_return = 0.0
        self.steps = 0
        self.time = 0.0
        self.info = {}
        self.trajectory = []

        # Reset diffusion chunk caches so next step triggers a fresh inference
        self._diff_chunk_cache = None
        self._diff_step_in_chunk = 0
        self._assisted_chunk_cache = None
        self._assisted_step_in_chunk = 0

        # Reset KTO state
        self.kto_planning = False
        self.kto_plan_times = None
        self.kto_plan = None
        self.kto_constraint_xy = None
        self.kto_solve_time = 0.0
        self.kto_manual_override_step = None

        # Start recording
        if self.recording_enabled and self.recorder:
            mode_name = self.mode.name.lower()
            self.recorder.start_episode(mode=mode_name, initial_obs=self.obs)

        # Launch KTO planning if in KTO mode
        if self.mode == ControlMode.KTO and self.kto_available:
            self._start_kto_planning()

    def _start_kto_planning(self):
        """Launch async KTO trajectory planning"""
        if not self.kto_available:
            return

        # Extract start state
        start = np.array([self.obs[0], self.obs[1], self.obs[2]])  # x, y, theta

        # Goal is always landing pad
        goal = np.array([self.env.pad_x, self.env.pad_y, 0.0])

        # Get obstacles from environment
        obstacles = self.env.obstacles if hasattr(self.env, 'obstacles') else []

        # Launch planning
        self.kto_planning = True
        self.planner.start_planning(
            start=start,
            goal=goal,
            obstacles=obstacles,
        )

    def handle_input(self):
        """Handle keyboard input"""
        keys = pygame.key.get_pressed()

        # Main engine: Up arrow (normalized to [0,1])
        self.action[0] = 1.0 if keys[pygame.K_UP] else 0.0

        # Side engines: reduced power for finer angle control
        if keys[pygame.K_LEFT]:
            self.action[1] = 0.3
        elif keys[pygame.K_RIGHT]:
            self.action[1] = -0.3
        else:
            self.action[1] = 0.0

    def step(self):
        """Step environment"""
        if self.done:
            return

        # Check if KTO planning is in progress
        if self.mode == ControlMode.KTO and self.kto_planning:
            if self.planner.is_planning():
                # Still planning, don't step environment
                return
            else:
                # Planning complete, retrieve result
                try:
                    (self.kto_plan_times, self.kto_plan, self.kto_constraint_xy,
                     _warm_xy, _knot_xy, _control_xy) = self.planner.get_result()
                    self.kto_solve_time = self.planner.get_solve_time()
                    self.kto_planning = False
                    print(f"KTO planning complete: {self.kto_solve_time:.2f}s, "
                          f"duration={self.kto_plan_times[-1]:.2f}s")
                except Exception as e:
                    print(f"KTO solve failed: {e}")
                    # Fall back to teleop
                    self.mode = ControlMode.TELEOP
                    self.kto_planning = False

        # Determine action based on control mode
        if self.mode == ControlMode.HEURISTIC:
            # Apply collision occlusion if failure level is set
            obs_for_policy = self.obs.copy()
            self.occluded_rays.clear()

            if self.failure_level > 0:
                # Occlude lidar readings (indices 6-13)
                occlusion_rate = self.occlusion_rates[self.failure_level]
                for i in range(6, 14):
                    if np.random.random() < occlusion_rate:
                        obs_for_policy[i] = 1.0
                        self.occluded_rays.add(i - 6)

            action_array = heuristic(self.env, obs_for_policy)
            action = [float(action_array[0]), float(action_array[1])]

        elif self.mode == ControlMode.KTO:
            # Check for manual override
            keys = pygame.key.get_pressed()
            manual_input = keys[pygame.K_UP] or keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]

            if manual_input and self.kto_manual_override_step is None:
                # User took over
                self.kto_manual_override_step = self.steps
                self.mode = ControlMode.TELEOP
                print(f"Manual override at step {self.steps}")

            if self.mode == ControlMode.KTO and self.kto_plan is not None:
                # Execute KTO plan
                if self.time < self.kto_plan_times[-1]:
                    idx = np.searchsorted(self.kto_plan_times, self.time, side='right') - 1
                    idx = int(np.clip(idx, 0, len(self.kto_plan_times) - 1))

                    from diffusha.planning.physics_kto import (
                        MASS, GRAVITY, INERTIA, SIDE_ARM)
                    THRUST_MAX = 2.0 * MASS * GRAVITY
                    SIDE_MAX = 0.5 * MASS * GRAVITY

                    # Acceleration replay: recompute thrusts from planned
                    # world-frame accelerations at current theta.
                    ax_p = self.kto_plan['ax'][idx]
                    ay_p = self.kto_plan['ay'][idx]
                    alpha_p = self.kto_plan['alpha'][idx]
                    theta = self.obs[2]
                    ct, st = np.cos(theta), np.sin(theta)
                    Fm = MASS * (-ax_p * st + (ay_p + GRAVITY) * ct)
                    Fs = alpha_p * INERTIA / SIDE_ARM

                    action = [float(np.clip(Fm / THRUST_MAX, 0, 1)),
                              float(np.clip(Fs / SIDE_MAX, -1, 1))]
                else:
                    # Plan complete, coast
                    action = [0.0, 0.0]
            else:
                action = [0.0, 0.0]

        elif self.mode == ControlMode.DIFFUSION:
            if self.diffusion_model is not None:
                action = self._diffusion_action()
            else:
                print("No diffusion model loaded — falling back to teleop")
                self.mode = ControlMode.TELEOP
                action = [float(self.action[0]), float(self.action[1])]

        elif self.mode == ControlMode.ASSISTED:
            if self.diffusion_model is not None:
                action = self._diffusion_assisted_action()
            else:
                print("No diffusion model loaded — falling back to teleop")
                self.mode = ControlMode.TELEOP
                action = [float(self.action[0]), float(self.action[1])]

        else:
            # Teleop: use keyboard input
            self.occluded_rays.clear()
            action = [float(self.action[0]), float(self.action[1])]

        # Step environment
        obs, reward, done, info = self.env.step(np.array(action))

        # Set control mode in info for recorder
        info['control_mode'] = self.mode.name.lower()

        # Record transition
        if self.recording_enabled and self.recorder:
            self.recorder.record_step(self.obs, action, reward, done, info)

        self.obs = obs
        self.done = done
        self.info = info
        self.episode_return += reward
        self.steps += 1
        self.time += 1.0 / 50  # 50 FPS timestep

        # Record trajectory
        x, y = obs[0], obs[1]
        self.trajectory.append((x, y))
        if len(self.trajectory) > 200:
            self.trajectory.pop(0)

    def render(self):
        """Render game state"""
        self.screen.fill(SKY)

        if self.obs is None:
            # Show start screen
            text = self.banner_font.render("Press R to start", True, WHITE)
            rect = text.get_rect(center=(self.window_width // 2, self.window_height // 2))
            self.screen.blit(text, rect)
            pygame.display.flip()
            return

        # Use KTO renderer if available, otherwise basic rendering
        if self.renderer:
            # Draw terrain
            self.renderer.draw_terrain(
                self.screen,
                self.env.terrain_xs,
                self.env.terrain_ys,
                self.env.pad_x,
                self.env.pad_y
            )

            # Draw flags
            self.renderer.draw_flags(self.screen, self.env.pad_x, self.env.pad_y)

            # Draw obstacles
            if hasattr(self.env, 'obstacles'):
                self.renderer.draw_obstacles(self.screen, self.env.obstacles)

            # Draw KTO planned trajectory
            if self.mode == ControlMode.KTO and self.kto_plan is not None:
                self.renderer.draw_planned_trajectory(
                    self.screen,
                    self.kto_plan,
                    self.kto_constraint_xy
                )

            # Draw lander trajectory trace
            if len(self.trajectory) > 1:
                self.renderer.draw_trace(self.screen, self.trajectory)

            # Draw lander
            x, y, theta = self.obs[0], self.obs[1], self.obs[2]
            # Get current thrusts (denormalize)
            from diffusha.planning.physics_kto import MASS, GRAVITY
            THRUST_MAX = 2.0 * MASS * GRAVITY
            SIDE_MAX = 0.5 * MASS * GRAVITY
            Fm = self.action[0] * THRUST_MAX
            Fs = self.action[1] * SIDE_MAX

            self.renderer.draw_lander(self.screen, x, y, theta, Fm, Fs)

            # Draw lidar rays (indices 6-13 in observation)
            lidar_readings = self.obs[6:14]
            self.renderer.draw_lidar_rays(
                self.screen, x, y, theta, lidar_readings,
                max_distance=3.0, n_rays=8,
                occluded_rays=self.occluded_rays
            )

            # Draw HUD
            state_dict = {
                'x': self.obs[0],
                'y': self.obs[1],
                'theta': self.obs[2],
                'vx': self.obs[3],
                'vy': self.obs[4],
                'omega': self.obs[5]
            }
            self.renderer.draw_hud(
                self.screen, self.font, self.time, state_dict,
                Fm, Fs, self.mode.name.lower(), self.kto_solve_time
            )

            # Draw KTO planning progress bar
            if self.kto_planning:
                progress = self.planner.get_progress()
                lander_pos = (self.obs[0], self.obs[1], self.obs[2])
                self.renderer.draw_progress_bar(
                    self.screen, self.font,
                    progress['phase'], progress['frac'],
                    lander_pos
                )

            # Draw episode end banner
            if self.done:
                outcome = self.info.get('goal', None)
                if outcome == 'landed':
                    self.renderer.draw_banner(self.screen, self.banner_font, "LANDED!", success=True)
                elif outcome == 'obstacle-collision':
                    self.renderer.draw_banner(self.screen, self.banner_font, "HIT OBSTACLE!", success=False)
                elif outcome == 'terrain-crash':
                    self.renderer.draw_banner(self.screen, self.banner_font, "TERRAIN CRASH!", success=False)
                elif outcome == 'out-of-bounds':
                    self.renderer.draw_banner(self.screen, self.banner_font, "OUT OF BOUNDS!", success=False)
                else:
                    self.renderer.draw_banner(self.screen, self.banner_font, "TIMEOUT!", success=False)

        else:
            # Fallback basic rendering (if renderer not available)
            text = self.font.render("Renderer not available", True, WHITE)
            self.screen.blit(text, (10, 10))

        # Draw controls reminder
        controls_text = self.font.render(
            "↑ Thrust  ← → Rotate  R Reset  Q Quit  1 Teleop  2 Heuristic  3 KTO  4 Diffusion  5 Assisted  F Failures  E Record",
            True, GRAY
        )
        self.screen.blit(controls_text, (10, self.window_height - 20))

        pygame.display.flip()

    def run(self):
        """Main game loop"""
        print("DEBUG: Entered run() method")
        running = True
        print("DEBUG: Starting event loop...")

        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self.reset()
                    elif event.key == pygame.K_1:
                        self.mode = ControlMode.TELEOP
                        print("Switched to TELEOP mode")
                    elif event.key == pygame.K_2:
                        self.mode = ControlMode.HEURISTIC
                        print("Switched to HEURISTIC mode")
                    elif event.key == pygame.K_3:
                        if self.kto_available:
                            self.mode = ControlMode.KTO
                            print("Switched to KTO mode")
                        else:
                            print("KTO mode unavailable (Drake not installed)")
                    elif event.key == pygame.K_4:
                        if self.diffusion_model is not None:
                            self.mode = ControlMode.DIFFUSION
                            print(f"Switched to DIFFUSION mode ({self.diffusion_model_name})")
                        else:
                            print("No diffusion model loaded. Pass --model <checkpoint.pt>")
                    elif event.key == pygame.K_5:
                        if self.diffusion_model is not None:
                            self.mode = ControlMode.ASSISTED
                            print(f"Switched to ASSISTED mode ({self.diffusion_model_name}) "
                                  f"fwd_diff_steps={self.diffusion_fwd_diff_steps} "
                                  f"exec_horizon={self.diffusion_exec_horizon}")
                        else:
                            print("No diffusion model loaded. Pass --model <checkpoint.pt>")
                    elif event.key == pygame.K_f:
                        self.failure_level = (self.failure_level + 1) % 4
                        failure_labels = {0: "None", 1: "Low (10%)", 2: "Medium (30%)", 3: "High (50%)"}
                        print(f"Failure level: {failure_labels[self.failure_level]}")
                    elif event.key == pygame.K_e:
                        self.recording_enabled = not self.recording_enabled
                        print(f"Recording: {'ON' if self.recording_enabled else 'OFF'}")

            # Handle input
            self.handle_input()

            # Step simulation
            self.step()

            # Render
            self.render()

            # Run at 50 FPS (matches DT in environment)
            self.clock.tick(50)

        pygame.quit()


def main():
    """Entry point"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None,
                        help='Path to diffusion model checkpoint (.pt)')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Display name for the model (default: inferred from path)')
    parser.add_argument('--quality_cond', action='store_true',
                        help='Load as quality-conditioned (collision-conditioned) model')
    parser.add_argument('--horizon', type=int, default=16,
                        help='Action chunk length the model was trained with (default: 16)')
    parser.add_argument('--exec_horizon', type=int, default=4,
                        help='Steps to execute per diffusion inference — receding horizon K (default: 4)')
    parser.add_argument('--fwd_diff_steps', type=int, default=20,
                        help='Forward diffusion steps applied to user action in assisted mode '
                             '(default: 20, valid range: 1..num_diffusion_steps-1=49)')
    args = parser.parse_args()

    print("=" * 70)
    print("LUNAR LANDER - KTO + HEURISTIC + TELEOP + DIFFUSION")
    print("=" * 70)
    print("\nControl Modes:")
    print("  1 - TELEOP:    Human keyboard control")
    print("  2 - HEURISTIC: PID controller with optional failures")
    print("  3 - KTO:       Drake trajectory optimization (requires Python 3.10-3.12)")
    print("  4 - DIFFUSION: Trained diffusion model autonomous (requires --model flag)")
    print("  5 - ASSISTED:  Shared autonomy — keyboard nudges diffusion (requires --model flag)")
    print("\nControls:")
    print("  ↑ Arrow Up:    Main engine (Teleop)")
    print("  ← → Arrows:    Rotate left/right — partial thrust (Teleop)")
    print("  R:             Reset episode")
    print("  Q/Escape:      Quit")
    print("  1/2/3:         Switch control modes")
    print("  F:             Cycle failure level (Heuristic mode)")
    print("  E:             Toggle episode recording")
    print("\nFeatures:")
    print("  - KTO environment with terrain + circular obstacles")
    print("  - 15-dim hybrid observation: state + lidar + pad_x")
    print("  - Episode recording with KTO metadata")
    print("  - Manual override capability in KTO mode")
    print("=" * 70)

    if not KTO_AVAILABLE:
        print("\nWARNING: Drake not available. KTO mode disabled.")
        print("To enable KTO:")
        print("  1. Create Python 3.10-3.12 virtual environment")
        print("  2. pip install drake>=1.30.0")
        print("=" * 70)

    print("\nStarting game...")
    try:
        print("DEBUG: Creating player...")
        player = LunarLanderPlayer()
        print("DEBUG: Player created successfully")

        if args.model:
            player.load_diffusion_model(
                args.model,
                name=args.model_name,
                quality_cond=args.quality_cond,
                horizon=args.horizon,
                exec_horizon=args.exec_horizon,
                fwd_diff_steps=args.fwd_diff_steps,
            )
            player.mode = ControlMode.DIFFUSION

        print("DEBUG: Calling player.run()...")
        player.run()
        print("DEBUG: player.run() returned")
    except Exception as e:
        print(f"ERROR in main(): {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
