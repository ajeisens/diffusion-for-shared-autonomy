#!/usr/bin/env python3
"""
CEM-based MPC planner for LunarLander.

Uses Cross-Entropy Method (CEM) to optimize action sequences under a learned
dynamics model. Intentionally trained on obstacle-free data, so it is blind
to obstacles at test time.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# Import dynamics model and Actor base
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from diffusha.baseline.train_dynamics import DynamicsModel
from diffusha.actor.base import Actor


class CEMPlanner:
    """
    Cross-Entropy Method planner for trajectory optimization.

    Samples action sequences, evaluates them under a dynamics model,
    selects elite samples, and refits distribution iteratively.
    """
    def __init__(
        self,
        dynamics_model,
        horizon=10,
        n_samples=200,
        n_elites=20,
        cem_iterations=5,
        action_dim=2,
        action_low=-1.0,
        action_high=1.0,
        device='cuda'
    ):
        """
        Args:
            dynamics_model: Trained DynamicsModel
            horizon: Planning horizon (number of steps to plan ahead)
            n_samples: Number of action sequences to sample per iteration
            n_elites: Number of elite samples to use for refitting
            cem_iterations: Number of CEM iterations
            action_dim: Action space dimension
            action_low: Lower bound for actions
            action_high: Upper bound for actions
            device: 'cuda' or 'cpu'
        """
        self.dynamics_model = dynamics_model
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_elites = n_elites
        self.cem_iterations = cem_iterations
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.device = device

    def plan(self, initial_state, goal_pos):
        """
        Plan optimal action sequence using CEM.

        Args:
            initial_state: (state_dim,) tensor, current state
            goal_pos: (2,) tuple/array, goal (x, y) position

        Returns:
            action: (action_dim,) numpy array, first action of best plan
        """
        # Initialize mean and std for action distribution
        # Shape: (horizon, action_dim)
        mean = torch.zeros(self.horizon, self.action_dim, device=self.device)
        std = torch.ones(self.horizon, self.action_dim, device=self.device)

        goal_x, goal_y = goal_pos

        for iteration in range(self.cem_iterations):
            # Sample action sequences
            # Shape: (n_samples, horizon, action_dim)
            noise = torch.randn(self.n_samples, self.horizon, self.action_dim,
                                device=self.device)
            action_sequences = mean.unsqueeze(0) + std.unsqueeze(0) * noise

            # Clip to action bounds
            action_sequences = torch.clamp(action_sequences, self.action_low, self.action_high)

            # Evaluate each sequence
            costs = self._evaluate_sequences(initial_state, action_sequences, goal_x, goal_y)

            # Select elites (lowest cost)
            elite_indices = torch.argsort(costs)[:self.n_elites]
            elite_sequences = action_sequences[elite_indices]

            # Refit distribution
            mean = elite_sequences.mean(dim=0)
            std = elite_sequences.std(dim=0) + 1e-6  # Add small epsilon for numerical stability

        # Return first action of mean plan
        return mean[0].cpu().numpy()

    def _evaluate_sequences(self, initial_state, action_sequences, goal_x, goal_y):
        """
        Evaluate action sequences by rolling out under dynamics model.

        Args:
            initial_state: (state_dim,) tensor
            action_sequences: (n_samples, horizon, action_dim) tensor
            goal_x, goal_y: Goal position

        Returns:
            costs: (n_samples,) tensor of costs (lower is better)
        """
        n_samples = action_sequences.shape[0]

        # Initialize states
        # Shape: (n_samples, state_dim)
        states = initial_state.unsqueeze(0).repeat(n_samples, 1)

        total_costs = torch.zeros(n_samples, device=self.device)

        # Rollout horizon steps
        for t in range(self.horizon):
            actions = action_sequences[:, t, :]

            # Predict next state
            with torch.no_grad():
                delta = self.dynamics_model(states, actions)
                states = states + delta

            # Compute cost (negative reward = cost)
            # Reward function: negative distance to goal
            # State format: [pos_x_norm, pos_y_norm, vel_x, vel_y, angle, angular_vel, leg1, leg2]
            # pos_x_norm is normalized to [-1, 1] where 0 is center of viewport
            # We need to denormalize or use normalized goal

            # For simplicity, use normalized state directly
            # LunarLander state[0] = (pos.x - W/2) / (W/2), so pos_x_norm = 0 is center
            # Goal is passed in absolute coords, but we need normalized

            # Actually, let's compute distance in normalized space
            # state[0] is x position normalized
            # state[1] is y position normalized

            pos_x_norm = states[:, 0]
            pos_y_norm = states[:, 1]

            # Normalize goal position
            # W = VIEWPORT_W / SCALE, typically 20
            # goal_x is in [0, W], center at W/2 = 10
            # Normalized: (goal_x - W/2) / (W/2)
            # Since we don't have W here, we'll assume the caller provides normalized goal
            # OR we pass in the goal in the same normalized format

            # For now, assume goal_x and goal_y are already normalized to match state
            # Distance cost
            dist_sq = (pos_x_norm - goal_x)**2 + (pos_y_norm - goal_y)**2
            cost = torch.sqrt(dist_sq)

            # Add velocity penalty (prefer slow approach)
            vel_penalty = 0.1 * (states[:, 2]**2 + states[:, 3]**2)

            # Angle penalty (prefer upright)
            angle_penalty = 0.1 * states[:, 4]**2

            total_costs += cost + vel_penalty + angle_penalty

        return total_costs


class MPCAgent(Actor):
    """
    MPC agent using CEM planner with learned dynamics model.

    This agent is INTENTIONALLY BLIND to obstacles because:
    1. It strips lidar readings from observations (uses only base 8-dim state)
    2. Its dynamics model was trained on obstacle-free demonstrations

    This is the baseline we expect to fail on obstacle environments.
    """
    def __init__(
        self,
        obs_space,
        act_space,
        dynamics_model_path,
        horizon=10,
        n_samples=200,
        n_elites=20,
        cem_iterations=5,
        device='cuda',
        helipad_x=0.0,  # Normalized helipad x position (updated per episode)
    ):
        """
        Args:
            obs_space: Observation space (17-dim for LunarLanderObstacle-v5)
            act_space: Action space
            dynamics_model_path: Path to trained dynamics model checkpoint
            horizon: MPC horizon
            n_samples: CEM samples
            n_elites: CEM elites
            cem_iterations: CEM iterations
            device: 'cuda' or 'cpu'
            helipad_x: Goal x position (normalized)
        """
        super().__init__(obs_space, act_space)

        # Load dynamics model
        print(f"Loading dynamics model from {dynamics_model_path}...")
        checkpoint = torch.load(dynamics_model_path, map_location=device)
        state_dim = checkpoint['state_dim']
        action_dim = checkpoint['action_dim']
        hidden_dim = checkpoint['hidden_dim']

        self.dynamics_model = DynamicsModel(state_dim, action_dim, hidden_dim).to(device)
        self.dynamics_model.load_state_dict(checkpoint['model_state_dict'])
        self.dynamics_model.eval()

        print(f"Loaded model: state_dim={state_dim}, action_dim={action_dim}, hidden_dim={hidden_dim}")

        # Create CEM planner
        self.planner = CEMPlanner(
            dynamics_model=self.dynamics_model,
            horizon=horizon,
            n_samples=n_samples,
            n_elites=n_elites,
            cem_iterations=cem_iterations,
            action_dim=action_dim,
            action_low=act_space.low[0],
            action_high=act_space.high[0],
            device=device
        )

        self.device = device
        self.helipad_x = helipad_x
        self.helipad_y = 0.0  # Typically around 0 in normalized coords

    def set_goal(self, helipad_x, helipad_y=0.0):
        """Update goal position (called at episode reset)"""
        self.helipad_x = helipad_x
        self.helipad_y = helipad_y

    def act(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        """
        Select action using MPC planning.

        CRITICAL: Strips lidar observations (last 8 dims) before planning.
        This makes the planner blind to obstacles.

        Args:
            obs: (17,) observation [8 base + 1 helipad + 8 lidar]

        Returns:
            action: (2,) continuous action
        """
        # Strip lidar readings (last 8 dims) and helipad x (dim 8)
        # Keep only base 8-dim state: [pos_x, pos_y, vel_x, vel_y, angle, ang_vel, leg1, leg2]
        base_obs = obs[:8]

        # Convert to tensor
        state = torch.from_numpy(base_obs).float().to(self.device)

        # Plan using CEM
        # Goal is helipad position (normalized)
        action = self.planner.plan(state, (self.helipad_x, self.helipad_y))

        return action


def test_mpc():
    """Quick test of MPC agent (requires trained model)"""
    import gym
    import diffusha

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained dynamics model')
    parser.add_argument('--env-name', type=str, default='LunarLanderObstacle-v5',
                        help='Environment name')
    parser.add_argument('--n-episodes', type=int, default=5,
                        help='Number of test episodes')
    parser.add_argument('--horizon', type=int, default=10,
                        help='MPC horizon')
    parser.add_argument('--n-samples', type=int, default=200,
                        help='CEM samples')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])

    args = parser.parse_args()

    # Create environment
    env = gym.make(args.env_name)

    # Create MPC agent
    agent = MPCAgent(
        obs_space=env.observation_space,
        act_space=env.action_space,
        dynamics_model_path=args.model_path,
        horizon=args.horizon,
        n_samples=args.n_samples,
        device=args.device
    )

    # Run episodes
    for ep in range(args.n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        step = 0

        # Update goal from environment
        if hasattr(env.unwrapped, 'helipad_x1'):
            helipad_x = (env.unwrapped.helipad_x1 + env.unwrapped.helipad_x2) / 2
            # Normalize to [-1, 1]
            from diffusha.data_collection.env.lunar_lander import VIEWPORT_W, SCALE
            W = VIEWPORT_W / SCALE
            helipad_x_norm = (helipad_x - W/2) / (W/2)
            agent.set_goal(helipad_x_norm, 0.0)

        while not done and step < 1000:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            step += 1

            if info.get('collision', False):
                print(f"  Episode {ep}: COLLISION at step {step}!")
                break

        print(f"Episode {ep}: reward={ep_reward:.2f}, steps={step}, "
              f"collision={info.get('collision', False)}")

    env.close()


if __name__ == "__main__":
    test_mpc()
