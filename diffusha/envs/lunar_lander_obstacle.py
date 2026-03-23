#!/usr/bin/env python3
"""
LunarLander environment with obstacle avoidance.
Extends the custom LunarLander from diffusha.data_collection.env.lunar_lander
"""

import math
import numpy as np
from typing import Optional

import Box2D
from Box2D.b2 import fixtureDef, polygonShape

import gym
from gym import spaces

# Import the custom LunarLander from the existing codebase
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from diffusha.data_collection.env.lunar_lander import LunarLander, SCALE, VIEWPORT_W, VIEWPORT_H, FPS

# Constants for obstacles
NUM_OBSTACLES = 4
OBSTACLE_HALF_WIDTH = 0.08  # Box2D units (8% of viewport width)
OBSTACLE_HALF_HEIGHT = 0.04  # Box2D units (4% of viewport height)
N_LIDAR_RAYS = 8
COLLISION_PENALTY = -100.0
MAX_LIDAR_DISTANCE = 3.0  # Maximum detection range in Box2D units


class LunarLanderObstacle(LunarLander):
    """
    LunarLander with static obstacles and lidar-based observation.

    Modifications from base LunarLander-v5:
    - Spawns NUM_OBSTACLES static box obstacles at random positions
    - Adds N_LIDAR_RAYS lidar rays to observation (normalized distances)
    - Detects collisions with obstacles and applies penalty
    - Observation space expands from 8-dim to (8 + N_LIDAR_RAYS)-dim
    """

    def __init__(self, **kwargs):
        # Force continuous and randomize_helipad for v5-like behavior
        kwargs['continuous'] = True
        kwargs['randomize_helipad'] = True
        kwargs['task'] = 'land'
        super().__init__(**kwargs)

        # Override observation space to include lidar
        # Base observation: 8-dim (pos_x, pos_y, vel_x, vel_y, angle, angular_vel, leg1_contact, leg2_contact)
        # + 1-dim for helipad_x (randomize_helipad=True)
        # + N_LIDAR_RAYS lidar distances
        base_obs_dim = 9  # 8 base + 1 helipad_x
        total_obs_dim = base_obs_dim + N_LIDAR_RAYS
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(total_obs_dim,), dtype=np.float32
        )

        # Storage for obstacle bodies
        self.obstacles = []

    def _destroy(self):
        """Extended destroy to also clean up obstacles"""
        # Destroy obstacles
        for obstacle in self.obstacles:
            if obstacle is not None:
                self.world.DestroyBody(obstacle)
        self.obstacles = []

        # Call parent destroy
        super()._destroy()

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        """Reset with obstacles spawned at random locations"""
        # Call parent reset first
        obs = super().reset(seed=seed, return_info=return_info, options=options)

        # Extract observation and info if return_info=True
        if return_info:
            base_obs, info = obs
        else:
            base_obs = obs
            info = {}

        # IMPORTANT: parent reset() calls step(), which already added lidar!
        # So base_obs is 17-dim (9 base + 8 lidar). Strip the lidar.
        if isinstance(base_obs, np.ndarray) and len(base_obs) > 9:
            base_obs = base_obs[:9]  # Keep only base + helipad_x

        # Spawn obstacles in the upper region of the screen
        # x ∈ [-1.2, 1.2] in normalized coords → scale to Box2D coords
        # y ∈ [0.3, 1.4] in normalized coords → scale to Box2D coords
        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        for i in range(NUM_OBSTACLES):
            # Random position (avoiding helipad region)
            x_norm = self.np_random.uniform(-1.2, 1.2)
            y_norm = self.np_random.uniform(0.3, 1.4)

            # Convert to Box2D coordinates
            x = W / 2 + x_norm * (W / 2)
            y = self.helipad_y + y_norm * (H / 2)

            # Avoid spawning too close to helipad
            while abs(x - (self.helipad_x1 + self.helipad_x2) / 2) < 2.0:
                x_norm = self.np_random.uniform(-1.2, 1.2)
                x = W / 2 + x_norm * (W / 2)

            # Create static box obstacle
            obstacle = self.world.CreateStaticBody(
                position=(x, y),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(OBSTACLE_HALF_WIDTH, OBSTACLE_HALF_HEIGHT)),
                    friction=0.3,
                ),
            )
            obstacle.userData = {'obstacle': True}
            obstacle.color1 = (180, 50, 50)  # Red color for rendering
            obstacle.color2 = (100, 30, 30)
            self.obstacles.append(obstacle)

        # Add obstacles to drawlist for rendering
        self.drawlist.extend(self.obstacles)

        # Compute lidar readings
        lidar_readings = self._compute_lidar()

        # Augment observation with lidar
        full_obs = np.concatenate([base_obs, lidar_readings]).astype(np.float32)

        if return_info:
            return full_obs, info
        else:
            return full_obs

    def _compute_lidar(self):
        """
        Cast N_LIDAR_RAYS rays from lander center, return normalized distances.
        Returns array of shape (N_LIDAR_RAYS,) with values in [0, 1].
        0 = obstacle at lander position, 1 = no obstacle within max range
        """
        if self.lander is None:
            return np.ones(N_LIDAR_RAYS, dtype=np.float32)

        lander_pos = self.lander.position
        distances = np.ones(N_LIDAR_RAYS, dtype=np.float32)

        for i in range(N_LIDAR_RAYS):
            angle = 2 * np.pi * i / N_LIDAR_RAYS

            # Ray endpoint
            end_x = lander_pos.x + MAX_LIDAR_DISTANCE * math.cos(angle)
            end_y = lander_pos.y + MAX_LIDAR_DISTANCE * math.sin(angle)

            # Find closest intersection with obstacles
            min_dist = MAX_LIDAR_DISTANCE

            for obstacle in self.obstacles:
                if obstacle is None:
                    continue

                # Simple AABB ray intersection for box obstacles
                obs_pos = obstacle.position

                # Check if ray intersects AABB of obstacle
                dist = self._ray_box_intersection(
                    lander_pos.x, lander_pos.y,
                    end_x, end_y,
                    obs_pos.x, obs_pos.y,
                    OBSTACLE_HALF_WIDTH, OBSTACLE_HALF_HEIGHT
                )

                if dist is not None and dist < min_dist:
                    min_dist = dist

            # Normalize to [0, 1]: 0 = very close, 1 = far/no obstacle
            distances[i] = min(min_dist / MAX_LIDAR_DISTANCE, 1.0)

        return distances

    def _ray_box_intersection(self, ray_x, ray_y, end_x, end_y, box_x, box_y, half_w, half_h):
        """
        Compute ray-AABB intersection. Returns distance to intersection or None.
        Uses slab method for axis-aligned bounding box intersection.
        """
        # Ray direction
        dx = end_x - ray_x
        dy = end_y - ray_y

        if abs(dx) < 1e-8 and abs(dy) < 1e-8:
            return None

        # AABB bounds
        x_min = box_x - half_w
        x_max = box_x + half_w
        y_min = box_y - half_h
        y_max = box_y + half_h

        # Slab intersection
        t_min = 0.0
        t_max = 1.0

        # X slab
        if abs(dx) > 1e-8:
            t1 = (x_min - ray_x) / dx
            t2 = (x_max - ray_x) / dx
            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))
        else:
            # Ray parallel to Y axis
            if ray_x < x_min or ray_x > x_max:
                return None

        # Y slab
        if abs(dy) > 1e-8:
            t1 = (y_min - ray_y) / dy
            t2 = (y_max - ray_y) / dy
            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))
        else:
            # Ray parallel to X axis
            if ray_y < y_min or ray_y > y_max:
                return None

        # Check if intersection exists
        if t_min > t_max or t_max < 0:
            return None

        # Return distance
        if t_min >= 0:
            dist = t_min * math.sqrt(dx * dx + dy * dy)
            return dist

        return None

    def step(self, action):
        """Extended step with obstacle collision detection"""
        # Call parent step
        base_obs, reward, done, info = super().step(action)

        # Check for collisions with obstacles
        collision_detected = False
        if self.lander is not None:
            for contact_edge in self.lander.contacts:
                # contact_edge is b2ContactEdge, actual contact is contact_edge.contact
                contact = contact_edge.contact
                # Check both fixtures in contact
                for fixture in [contact.fixtureA, contact.fixtureB]:
                    if fixture.body.userData is not None and fixture.body.userData.get('obstacle', False):
                        collision_detected = True
                        break
                if collision_detected:
                    break

        # Apply collision penalty
        if collision_detected:
            reward += COLLISION_PENALTY
            done = True
            info['collision'] = True
            info['game_over_reason'] = 'obstacle-collision'
            if self.lander is not None:
                self.lander.color1 = (255, 0, 0)  # Red color
        else:
            info['collision'] = False

        # Compute lidar readings
        lidar_readings = self._compute_lidar()

        # Augment observation with lidar
        full_obs = np.concatenate([base_obs, lidar_readings]).astype(np.float32)

        return full_obs, reward, done, info
