"""Lunar Lander environment with KTO physics and terrain generation.

Gym-compatible environment that combines:
- Poslathian's terrain generation and circular obstacles
- KTO simplified 2D rocket physics
- Hybrid observation space (state + lidar + pad position)
- Episode recording compatibility
"""

import math
import numpy as np
from typing import Optional, Tuple, Dict, List
import gym
from gym import spaces

from diffusha.planning.physics_kto import step_physics

# World geometry constants (same as kto_solver.py, duplicated to avoid Drake import)
SCALE = 30.0
VIEWPORT_W, VIEWPORT_H = 600, 400
W = VIEWPORT_W / SCALE  # 20.0
H = VIEWPORT_H / SCALE  # 13.33
PAD_X = W / 2            # 10.0
PAD_Y = H / 4            # 3.33

# Physics constants (same as kto_solver.py)
GRAVITY = 10.0
MASS = 1.0
INERTIA = 0.2
THRUST_MAX = 2.0 * MASS * GRAVITY
SIDE_MAX = 0.5 * MASS * GRAVITY
SIDE_ARM = 1.0

# Episode parameters
DT = 1.0 / 50  # 50 FPS
TIMEOUT = 20.0
MAX_EPISODE_STEPS = int(TIMEOUT / DT)

# Lidar parameters
N_LIDAR_RAYS = 8
MAX_LIDAR_DISTANCE = 3.0  # Box2D units


class LunarLanderKTO(gym.Env):
    """
    Lunar Lander with KTO physics, terrain, and circular obstacles.

    Observation Space (15-dim):
        [0-2]: Position and angle (x, y, theta)
        [3-5]: Velocity and angular velocity (vx, vy, omega)
        [6-13]: Lidar distance readings (8 rays, normalized 0-1)
        [14]: Landing pad x-position

    Action Space (2-dim continuous):
        [0]: Main engine thrust, range [0, 1] (maps to [0, THRUST_MAX])
        [1]: Side engine thrust, range [-1, 1] (maps to [-SIDE_MAX, SIDE_MAX])

    Episode Termination:
        - Success: Landed on pad (|x - pad_x| < 2.5, slow, upright)
        - Crashed: Hit obstacle, terrain, or hard landing
        - Out of bounds: |x| > W or t > TIMEOUT
    """

    metadata = {'render.modes': []}

    def __init__(self):
        super().__init__()

        # Observation: [x, y, θ, vx, vy, ω, lidar×8, pad_x]
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(15,), dtype=np.float32
        )

        # Action: [main_thrust, side_thrust] normalized to [-1, 1]
        self.action_space = spaces.Box(
            np.array([0.0, -1.0]),
            np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # State variables
        self.state = None
        self.terrain_xs = None
        self.terrain_ys = None
        self.obstacles = None
        self.pad_x = PAD_X
        self.pad_y = PAD_Y
        self.current_control_mode = 'teleop'  # Set by play.py

        # Episode tracking
        self.episode_step = 0
        self.np_random = None

    def seed(self, seed=None):
        """Set random seed."""
        if seed is not None:
            np.random.seed(seed)
        self.np_random = np.random.RandomState(seed)
        return [seed]

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False,
              options: Optional[dict] = None):
        """Reset environment to random initial state."""
        if seed is not None:
            self.seed(seed)
        if self.np_random is None:
            self.seed(42)

        # Generate terrain
        self.terrain_xs, self.terrain_ys = self._make_terrain(seed)

        # Random start position (top region)
        start_x = self.np_random.uniform(3.0, W - 3.0)
        start_y = self.np_random.uniform(H - 2.0, H - 0.5)
        start_theta = self.np_random.uniform(-0.15, 0.15)

        # Initialize state
        self.state = {
            'x': start_x,
            'y': start_y,
            'theta': start_theta,
            'vx': 0.0,
            'vy': 0.0,
            'omega': 0.0
        }

        # Generate obstacles (avoid start and goal)
        goal = np.array([self.pad_x, self.pad_y, 0.0])
        self.obstacles = self._make_obstacles(
            np.array([start_x, start_y, start_theta]),
            goal
        )

        # Reset episode tracking
        self.episode_step = 0

        obs = self._get_obs()

        if return_info:
            return obs, {}
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one timestep."""
        self.episode_step += 1

        # Denormalize action to thrust values
        Fm = float(action[0]) * THRUST_MAX  # [0, THRUST_MAX]
        Fs = float(action[1]) * SIDE_MAX    # [-SIDE_MAX, SIDE_MAX]

        # Update physics
        self.state = step_physics(self.state, Fm, Fs, DT)

        # Check termination
        done, outcome = self._check_done()

        # Compute reward
        reward = self._compute_reward(outcome)

        # Build observation
        obs = self._get_obs()

        # Build info dict (for episode recorder)
        info = {
            'goal': outcome if done else None,
            'collision': outcome == 'obstacle-collision' if done else False,
            'crashed': outcome in ['terrain-crash', 'hard-landing', 'out-of-bounds'] if done else False,
            'control_mode': self.current_control_mode,
        }

        return obs, reward, done, info

    def _get_obs(self) -> np.ndarray:
        """Build hybrid observation: [state(6) + lidar(8) + pad_x(1)]."""
        # State vector
        state_vec = np.array([
            self.state['x'],
            self.state['y'],
            self.state['theta'],
            self.state['vx'],
            self.state['vy'],
            self.state['omega']
        ], dtype=np.float32)

        # Lidar readings
        lidar = self._compute_lidar()

        # Pad position
        pad_x = np.array([self.pad_x], dtype=np.float32)

        return np.concatenate([state_vec, lidar, pad_x])

    def _compute_lidar(self) -> np.ndarray:
        """Cast N_LIDAR_RAYS rays from lander, return normalized distances."""
        lander_x = self.state['x']
        lander_y = self.state['y']
        lander_theta = self.state['theta']

        distances = np.ones(N_LIDAR_RAYS, dtype=np.float32)

        for i in range(N_LIDAR_RAYS):
            # Ray angle in world frame
            ray_angle = lander_theta + 2 * np.pi * i / N_LIDAR_RAYS

            # Ray endpoint
            end_x = lander_x + MAX_LIDAR_DISTANCE * math.cos(ray_angle)
            end_y = lander_y + MAX_LIDAR_DISTANCE * math.sin(ray_angle)

            # Check intersection with obstacles
            min_dist = MAX_LIDAR_DISTANCE

            for cx, cy, r in self.obstacles:
                dist = self._ray_circle_intersection(
                    lander_x, lander_y, end_x, end_y, cx, cy, r
                )
                if dist is not None and dist < min_dist:
                    min_dist = dist

            # Normalize to [0, 1]
            distances[i] = min(min_dist / MAX_LIDAR_DISTANCE, 1.0)

        return distances

    def _ray_circle_intersection(self, ray_x, ray_y, end_x, end_y, cx, cy, r):
        """Compute ray-circle intersection distance, or None."""
        # Ray direction
        dx = end_x - ray_x
        dy = end_y - ray_y
        ray_len = math.sqrt(dx * dx + dy * dy)

        if ray_len < 1e-8:
            return None

        # Normalize direction
        dx /= ray_len
        dy /= ray_len

        # Vector from ray origin to circle center
        fx = cx - ray_x
        fy = cy - ray_y

        # Project onto ray
        proj = fx * dx + fy * dy

        if proj < 0:
            return None  # Circle behind ray

        # Closest point on ray to circle center
        close_x = ray_x + proj * dx
        close_y = ray_y + proj * dy

        # Distance from closest point to circle center
        dist_to_center = math.sqrt((close_x - cx)**2 + (close_y - cy)**2)

        if dist_to_center > r:
            return None  # No intersection

        # Distance from ray origin to intersection
        offset = math.sqrt(r * r - dist_to_center * dist_to_center)
        intersection_dist = proj - offset

        if intersection_dist < 0 or intersection_dist > ray_len:
            return None

        return intersection_dist

    def _check_done(self) -> Tuple[bool, Optional[str]]:
        """Check if episode should terminate. Returns (done, outcome_str)."""
        x = self.state['x']
        y = self.state['y']
        vx = self.state['vx']
        vy = self.state['vy']
        theta = self.state['theta']

        # Out of bounds
        if x < 0 or x > W:
            return True, 'out-of-bounds'

        # Timeout
        if self.episode_step >= MAX_EPISODE_STEPS:
            return True, 'timeout'

        # Obstacle collision
        for cx, cy, r in self.obstacles:
            if (x - cx) ** 2 + (y - cy) ** 2 < r * r:
                return True, 'obstacle-collision'

        # Check terrain contact
        ground_height = self._terrain_height(x)

        if y > ground_height:
            return False, None  # Still flying

        # Touched ground - check landing quality
        on_pad = abs(x - self.pad_x) < 2.5
        slow = abs(vx) < 2.0 and abs(vy) < 3.0
        upright = abs(theta) < 0.5

        if on_pad and slow and upright:
            return True, 'landed'  # Success!
        else:
            return True, 'terrain-crash'  # Crash

    def _compute_reward(self, outcome: Optional[str]) -> float:
        """Compute reward based on current state and outcome."""
        if outcome == 'landed':
            return 1000.0  # Large bonus for success

        if outcome == 'obstacle-collision':
            return -100.0

        if outcome in ['terrain-crash', 'hard-landing']:
            return -100.0

        if outcome == 'out-of-bounds':
            return -100.0

        if outcome == 'timeout':
            return -100.0

        # Intermediate reward (flying)
        # Small penalty for distance from pad
        dist_to_pad = abs(self.state['x'] - self.pad_x) + abs(self.state['y'] - self.pad_y)
        return -0.1 * dist_to_pad

    def _terrain_height(self, x: float) -> float:
        """Get terrain height at position x via interpolation."""
        return float(np.interp(x, self.terrain_xs, self.terrain_ys))

    def _make_terrain(self, seed: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate random terrain with flat landing pad (from poslathian)."""
        rng = np.random.RandomState(seed)
        xs = np.linspace(0, W, 11)
        ys = rng.uniform(H / 8, H / 3, 11)

        # Flat landing pad at center
        ys[4:7] = PAD_Y

        # Smooth terrain
        for i in range(1, 10):
            ys[i] = (ys[i - 1] + ys[i] + ys[i + 1]) / 3

        # Ensure pad stays flat
        ys[4:7] = PAD_Y

        return xs, ys

    def _make_obstacles(self, start: np.ndarray, goal: np.ndarray) -> List[Tuple[float, float, float]]:
        """Generate random circular obstacles (from poslathian)."""
        n = self.np_random.integers(2, 4)
        obs = []

        for _ in range(n * 10):  # Rejection sampling
            if len(obs) >= n:
                break

            cx = self.np_random.uniform(2.0, W - 2.0)
            cy = self.np_random.uniform(PAD_Y + 1.5, H - 2.0)
            r = self.np_random.uniform(0.8, 2.0)

            # Don't block start or goal
            if np.hypot(cx - start[0], cy - start[1]) < r + 1.5:
                continue
            if np.hypot(cx - goal[0], cy - goal[1]) < r + 1.8:
                continue

            obs.append((cx, cy, r))

        return obs

    def render(self, mode='human'):
        """Rendering handled by play.py."""
        pass

    def close(self):
        """Cleanup."""
        pass
