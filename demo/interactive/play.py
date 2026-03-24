#!/usr/bin/env python3
"""
Lunar Lander Demo with KTO Trajectory Optimization

Integrates poslathian's Drake-based KTO planner with episode recording system.
Three control modes: Teleop (human), Heuristic (PID), and KTO (trajectory optimization).

Controls:
    Arrow Up: Main engine (Teleop mode)
    Arrow Left: Rotate left (Teleop mode)
    Arrow Right: Rotate right (Teleop mode)
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
    Heuristic controller for Lunar Lander (from OpenAI Gym).

    A PID-based controller that uses state feedback to land the lander.

    Args:
        env: The environment
        s: state [x, y, theta, vx, vy, omega, lidar×8, pad_x] (15-dim for KTO env)

    Returns:
        action: [main_engine_throttle, side_engine_throttle] normalized to [0,1] × [-1,1]
    """
    # Extract state (first 6 elements)
    x, y, theta, vx, vy, omega = s[:6]

    angle_targ = x * 0.5 + vx * 1.0  # angle should point towards center
    if angle_targ > 0.4:
        angle_targ = 0.4
    if angle_targ < -0.4:
        angle_targ = -0.4
    hover_targ = 0.55 * np.abs(x)

    angle_todo = (angle_targ - theta) * 0.5 - omega * 1.0
    hover_todo = (hover_targ - y) * 0.5 - vy * 0.5

    # KTO env doesn't have leg contact in observation, so check ground proximity
    if y < 3.5:  # Near ground
        angle_todo = 0
        hover_todo = -vy * 0.5

    # Continuous action
    a = np.array([hover_todo * 20 - 1, -angle_todo * 20])
    a = np.clip(a, [-1, -1], [1, 1])

    # Normalize to environment action space: [0,1] × [-1,1]
    a[0] = (a[0] + 1) / 2  # Map [-1,1] → [0,1] for main engine
    return a


class ControlMode(Enum):
    """Control mode enumeration"""
    TELEOP = 1      # Human keyboard control
    HEURISTIC = 2   # Heuristic PID controller
    KTO = 3         # Drake trajectory optimization


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
            self.planner = AsyncKTOPlanner(max_warmstart_iters=0, max_obstacle_iters=12)
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

        # Input state
        self.action = [0.0, 0.0]

        # Rendering
        self.trajectory = []

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
            num_control_points=20,
            num_dynamics_samples=40
        )

    def handle_input(self):
        """Handle keyboard input"""
        keys = pygame.key.get_pressed()

        # Main engine: Up arrow (normalized to [0,1])
        self.action[0] = 1.0 if keys[pygame.K_UP] else 0.0

        # Side engines: Left/Right arrows (normalized to [-1,1])
        if keys[pygame.K_LEFT]:
            self.action[1] = 1.0  # Positive = thrust right (rotate left)
        elif keys[pygame.K_RIGHT]:
            self.action[1] = -1.0  # Negative = thrust left (rotate right)
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
                    self.kto_plan_times, self.kto_plan, self.kto_constraint_xy = self.planner.get_result()
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
                    # Interpolate planned thrusts
                    idx = np.searchsorted(self.kto_plan_times, self.time, side='right') - 1
                    idx = np.clip(idx, 0, len(self.kto_plan_times) - 1)

                    # KTO plan has absolute thrusts, need to normalize
                    # Fm ∈ [0, THRUST_MAX], Fs ∈ [-SIDE_MAX, SIDE_MAX]
                    from diffusha.planning.physics_kto import MASS, GRAVITY
                    THRUST_MAX = 2.0 * MASS * GRAVITY
                    SIDE_MAX = 0.5 * MASS * GRAVITY

                    Fm_norm = self.kto_plan['Fm'][idx] / THRUST_MAX  # → [0, 1]
                    Fs_norm = self.kto_plan['Fs'][idx] / SIDE_MAX    # → [-1, 1]

                    action = [float(np.clip(Fm_norm, 0, 1)),
                             float(np.clip(Fs_norm, -1, 1))]
                else:
                    # Plan complete, coast
                    action = [0.0, 0.0]
            else:
                action = [0.0, 0.0]

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
                    self.renderer.draw_banner(self.screen, self.banner_font, "COLLISION!", success=False)
                elif outcome == 'terrain-crash':
                    self.renderer.draw_banner(self.screen, self.banner_font, "CRASHED!", success=False)
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
            "↑ Thrust  ← → Rotate  R Reset  Q Quit  1 Teleop  2 Heuristic  3 KTO  F Failures  E Record",
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
    print("=" * 70)
    print("LUNAR LANDER - KTO + HEURISTIC + TELEOP")
    print("=" * 70)
    print("\nControl Modes:")
    print("  1 - TELEOP:    Human keyboard control")
    print("  2 - HEURISTIC: PID controller with optional failures")
    print("  3 - KTO:       Drake trajectory optimization (requires Python 3.10-3.12)")
    print("\nControls:")
    print("  ↑ Arrow Up:    Main engine (Teleop mode)")
    print("  ← → Arrows:    Rotate left/right (Teleop mode)")
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
