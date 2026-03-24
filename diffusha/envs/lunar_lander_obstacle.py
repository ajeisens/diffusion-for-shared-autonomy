#!/usr/bin/env python3
"""
LunarLander with obstacles. Gymnasium-compatible wrapper around repo's gym LunarLander.
Imports the lunar_lander module directly to avoid the block_pushing/tf_agents import chain.
"""
import math
import importlib.util
import numpy as np
from pathlib import Path
import Box2D
from Box2D.b2 import fixtureDef, polygonShape
import gymnasium
from gymnasium import spaces

# Direct file import — bypasses diffusha/data_collection/env/__init__.py
# which pulls in block_pushing -> tf_agents -> broken numpy/TF
_spec = importlib.util.spec_from_file_location(
    "lunar_lander_gym",
    Path(__file__).parent.parent / "data_collection/env/lunar_lander.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_GymLunarLander = _mod.LunarLander
SCALE     = _mod.SCALE
VIEWPORT_W = _mod.VIEWPORT_W
VIEWPORT_H = _mod.VIEWPORT_H
FPS       = _mod.FPS

NUM_OBSTACLES        = 4
OBSTACLE_HALF_WIDTH  = 0.08
OBSTACLE_HALF_HEIGHT = 0.04
N_LIDAR_RAYS         = 8
COLLISION_PENALTY    = -100.0
MAX_LIDAR_DISTANCE   = 3.0


class LunarLanderObstacle(gymnasium.Env):
    """
    Gymnasium wrapper around repo's gym LunarLander.
    Obs shape: (17,) = 9 base + 8 lidar rays.
    Step: (obs, reward, terminated, truncated, info)
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode=None):
        super().__init__()
        self._env = _GymLunarLander(
            continuous=True,
            randomize_helipad=True,
            task='land',
        )
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(9 + N_LIDAR_RAYS,), dtype=np.float32
        )
        self.render_mode = render_mode
        self.obstacles = []

    @property
    def helipad_x1(self): return self._env.helipad_x1
    @property
    def helipad_x2(self): return self._env.helipad_x2
    @property
    def helipad_y(self):  return self._env.helipad_y
    @property
    def world(self):      return self._env.world
    @property
    def lander(self):     return self._env.lander
    @property
    def np_random(self):  return self._env.np_random

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._env.reset(seed=seed)
        base_obs = self._env.reset()
        if isinstance(base_obs, tuple):
            base_obs = base_obs[0]
        if len(base_obs) > 9:
            base_obs = base_obs[:9]
        self._spawn_obstacles()
        lidar = self._compute_lidar()
        return np.concatenate([base_obs, lidar]).astype(np.float32), {}

    def step(self, action):
        result = self._env.step(action)
        if len(result) == 4:
            base_obs, reward, done, info = result
            terminated, truncated = done, False
        else:
            base_obs, reward, terminated, truncated, info = result
        if len(base_obs) > 9:
            base_obs = base_obs[:9]
        if self._check_collision():
            reward += COLLISION_PENALTY
            terminated = True
            info['collision'] = True
            info['game_over_reason'] = 'obstacle-collision'
        else:
            info.setdefault('collision', False)
        lidar = self._compute_lidar()
        return np.concatenate([base_obs, lidar]).astype(np.float32), float(reward), terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    def _spawn_obstacles(self):
        for obs in self.obstacles:
            if obs is not None:
                try: self.world.DestroyBody(obs)
                except Exception: pass
        self.obstacles = []
        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE
        helipad_cx = (self._env.helipad_x1 + self._env.helipad_x2) / 2
        for _ in range(NUM_OBSTACLES):
            for _ in range(20):
                x_norm = self._env.np_random.uniform(-1.2, 1.2)
                y_norm = self._env.np_random.uniform(0.3, 1.4)
                x = W / 2 + x_norm * (W / 2)
                y = self._env.helipad_y + y_norm * (H / 2)
                if abs(x - helipad_cx) >= 2.0:
                    break
            obstacle = self.world.CreateStaticBody(
                position=(x, y),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(OBSTACLE_HALF_WIDTH, OBSTACLE_HALF_HEIGHT)),
                    friction=0.3,
                ),
            )
            obstacle.userData = {'obstacle': True}
            self.obstacles.append(obstacle)

    def _check_collision(self):
        if self.lander is None:
            return False
        for ce in self.lander.contacts:
            contact = ce.contact
            for fix in [contact.fixtureA, contact.fixtureB]:
                ud = fix.body.userData
                if isinstance(ud, dict) and ud.get('obstacle', False):
                    return True
        return False

    def _compute_lidar(self):
        if self.lander is None:
            return np.ones(N_LIDAR_RAYS, dtype=np.float32)
        lpos = self.lander.position
        distances = np.ones(N_LIDAR_RAYS, dtype=np.float32)
        for i in range(N_LIDAR_RAYS):
            angle = 2 * np.pi * i / N_LIDAR_RAYS
            ex = lpos.x + MAX_LIDAR_DISTANCE * math.cos(angle)
            ey = lpos.y + MAX_LIDAR_DISTANCE * math.sin(angle)
            min_dist = MAX_LIDAR_DISTANCE
            for obs in self.obstacles:
                if obs is None: continue
                d = self._ray_box(lpos.x, lpos.y, ex, ey,
                                  obs.position.x, obs.position.y,
                                  OBSTACLE_HALF_WIDTH, OBSTACLE_HALF_HEIGHT)
                if d is not None and d < min_dist:
                    min_dist = d
            distances[i] = min(min_dist / MAX_LIDAR_DISTANCE, 1.0)
        return distances

    def _ray_box(self, rx, ry, ex, ey, bx, by, hw, hh):
        dx, dy = ex - rx, ey - ry
        if abs(dx) < 1e-8 and abs(dy) < 1e-8:
            return None
        t_min, t_max = 0.0, 1.0
        for (r, d, lo, hi) in [(rx, dx, bx-hw, bx+hw), (ry, dy, by-hh, by+hh)]:
            if abs(d) > 1e-8:
                t1, t2 = (lo - r) / d, (hi - r) / d
                t_min = max(t_min, min(t1, t2))
                t_max = min(t_max, max(t1, t2))
            elif not (lo <= r <= hi):
                return None
        if t_min > t_max or t_max < 0:
            return None
        if t_min >= 0:
            return t_min * math.sqrt(dx*dx + dy*dy)
        return None
