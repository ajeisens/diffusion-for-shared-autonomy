#!/usr/bin/env python3
"""
Environment registration for obstacle-aware LunarLander.
"""

import gym
from gym.envs.registration import register

# Register LunarLanderObstacle-v5
register(
    id='LunarLanderObstacle-v5',
    entry_point='diffusha.envs.lunar_lander_obstacle:LunarLanderObstacle',
    max_episode_steps=1000,
)
