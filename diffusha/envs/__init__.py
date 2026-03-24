#!/usr/bin/env python3
"""
Environment registration for obstacle-aware LunarLander.
"""
from gymnasium.envs.registration import register

register(
    id='LunarLanderObstacle-v5',
    entry_point='diffusha.envs.lunar_lander_obstacle:LunarLanderObstacle',
    max_episode_steps=1000,
)
