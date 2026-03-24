#!/usr/bin/env python3

import pfrl
import gym

from diffusha.data_collection.env.assistance_wrappers import (
    BlockPushMirrorObsWrapper, LunarLanderSplitObsWrapper,
    BlockPushSplitObsWrapper, BlockPushExpandObsWrapper
)
from diffusha.data_collection.env.multiprocess_vector_env import MultiprocessVectorEnv

# BlockPushMultimodal is NOT imported at module level — it pulls in
# tf_agents which is broken with numpy>=1.20. Import lazily only when needed.


class Spec:
    def __init__(self, name) -> None:
        self.name = name


def make_env(env_name, test, seed=0, terminate_at_any_goal=False, split_obs=False, user_goal='target', **kwargs):
    if split_obs:
        assert "maze" not in env_name

    if "LunarLander" in env_name:
        if kwargs != {}:
            print('extra kwargs to make_env:', kwargs)
        from .lunar_lander import LunarLander
        version = env_name.split('-')[-1]
        if version == 'v1':
            env = LunarLander(continuous=True, task='reach', spec=Spec(f'LunarLander-{version}'), **kwargs)
        elif version == 'v4':
            env = LunarLander(continuous=True, task='float', fuel_penalty=False, spec=Spec(f'LunarLander-{version}'), **kwargs)
        elif version == 'v5':
            env = LunarLander(continuous=True, randomize_helipad=True, spec=Spec(f'LunarLander-{version}'), **kwargs)
        else:
            env = LunarLander(continuous=True, randomize_helipad=False, spec=Spec(f'LunarLander-{version}'), **kwargs)
        time_limit = 1000

    elif 'maze' in env_name:
        import d4rl
        if terminate_at_any_goal:
            env = gym.make(env_name, reward_type='sparse', terminate_at_any_goal=True, **kwargs)
        else:
            env = gym.make(env_name, reward_type='sparse', terminate_at_goal=True, **kwargs)
        time_limit = 300

    else:
        # Lazy import — only reached for block pushing
        from diffusha.data_collection.env.block_pushing.block_pushing_multimodal_1block import BlockPushMultimodal
        env = gym.make('BlockPushMultimodal-v1', user_goal=user_goal)
        time_limit = 200

    env_seed = 2**32 - 1 - seed if test else seed
    env.seed(env_seed)

    if "LunarLander" in env_name:
        if env_name == "LunarLander-v3":
            from .reward_wrapper import LunarLanderRewardWrapper
            env = LunarLanderRewardWrapper(env)

    elif 'maze' in env_name:
        from .pointmaze_wrapper import PointMazeTerminationWrapper
        env = PointMazeTerminationWrapper(env)

    elif 'Push' in env_name:
        env = BlockPushExpandObsWrapper(env)

    else:
        raise Exception("Unexpected Env Name")

    env = pfrl.wrappers.CastObservationToFloat32(env)
    env = pfrl.wrappers.NormalizeActionSpace(env)

    if 'LunarLander' in env_name and split_obs:
        env = LunarLanderSplitObsWrapper(env)

    if 'Push' in env_name and split_obs:
        env = BlockPushSplitObsWrapper(env)

    if 'Push' not in env_name:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=time_limit)

    return env


def is_maze2d(env):
    return 'maze2d' in env.unwrapped.spec.name

def is_lunarlander(env):
    return 'LunarLander' in env.unwrapped.spec.name

def is_blockpush(env):
    return 'BlockPush' in env.unwrapped.spec.name
