#!/usr/bin/env python3
"""
Train dynamics model for MPC baseline.

Trains a simple MLP to predict state transitions delta_s = s_{t+1} - s_t
given current state s_t and action a_t.

IMPORTANT: This is trained on obstacle-free demonstrations (LunarLander-v5)
so it will be blind to obstacles at test time. This is the intentional failure
mode we want to demonstrate.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import ReplayBuffer from data_collection
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from diffusha.data_collection.generate_data import ReplayBuffer


class DynamicsDataset(Dataset):
    """
    Dataset for dynamics model training.

    Loads transitions from ReplayBuffer and returns (state, action, next_state).
    """
    def __init__(self, replay_dir, state_dim=8, action_dim=2, max_samples=None):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Load replay buffer
        print(f"Loading replay buffer from {replay_dir}...")
        self.replay_buffer = ReplayBuffer(replay_dir, state_dim, action_dim)

        # Sample all transitions
        # ReplayBuffer.sample() returns single transition, so we need to sample many
        # For efficiency, we'll load chunks directly
        self.transitions = []

        replay_path = Path(replay_dir)
        chunk_files = sorted(replay_path.glob("chunk_*.pt"))

        if not chunk_files:
            raise ValueError(f"No chunk files found in {replay_dir}")

        print(f"Found {len(chunk_files)} chunk files")

        for chunk_file in chunk_files:
            chunk_data = torch.load(chunk_file)
            # chunk_data shape: (chunk_size, state_dim + action_dim + 1)
            # Format: [state (8), action (2), q_val (1)]

            # Extract state, action pairs
            # Note: We need consecutive transitions to compute next_state
            # For now, we'll load all data and create transitions
            self.transitions.append(chunk_data)

        # Concatenate all chunks
        self.transitions = np.concatenate(self.transitions, axis=0)

        if max_samples is not None and len(self.transitions) > max_samples:
            # Random subsample
            indices = np.random.choice(len(self.transitions), max_samples, replace=False)
            self.transitions = self.transitions[indices]

        print(f"Loaded {len(self.transitions)} transitions")

        # Note: For delta prediction, we need pairs of consecutive states
        # Since replay buffer stores individual transitions, we'll assume
        # consecutive entries are from same trajectory (mostly true for demos)
        # Alternatively, we could track episode boundaries, but for MLP training
        # this approximation is acceptable

    def __len__(self):
        # Return length minus 1 since we need pairs
        return len(self.transitions) - 1

    def __getitem__(self, idx):
        """
        Returns (state_t, action_t, delta_state)
        where delta_state = state_{t+1} - state_t
        """
        # Current transition
        trans_t = self.transitions[idx]
        state_t = trans_t[:self.state_dim]
        action_t = trans_t[self.state_dim:self.state_dim + self.action_dim]

        # Next transition
        trans_t1 = self.transitions[idx + 1]
        state_t1 = trans_t1[:self.state_dim]

        # Compute delta
        delta_state = state_t1 - state_t

        return (
            torch.from_numpy(state_t).float(),
            torch.from_numpy(action_t).float(),
            torch.from_numpy(delta_state).float()
        )


class DynamicsModel(nn.Module):
    """
    Simple MLP dynamics model.

    Input: [state (8-dim), action (2-dim)] = 10-dim
    Output: delta_state (8-dim)

    Architecture: 3-layer MLP with SiLU activations, hidden_dim=256
    """
    def __init__(self, state_dim=8, action_dim=2, hidden_dim=256):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        input_dim = state_dim + action_dim
        output_dim = state_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, state, action):
        """
        Predict delta_state given state and action.

        Args:
            state: (batch, state_dim)
            action: (batch, action_dim)

        Returns:
            delta_state: (batch, state_dim)
        """
        x = torch.cat([state, action], dim=-1)
        delta = self.net(x)
        return delta

    def predict_next_state(self, state, action):
        """Predict next state (for use in MPC)"""
        with torch.no_grad():
            delta = self.forward(state, action)
            return state + delta


def train_dynamics_model(
    replay_dir,
    output_path,
    state_dim=8,
    action_dim=2,
    hidden_dim=256,
    batch_size=512,
    num_steps=100_000,
    lr=3e-4,
    device='cuda',
    max_samples=None
):
    """
    Train dynamics model on replay buffer data.

    Args:
        replay_dir: Path to replay buffer directory
        output_path: Where to save trained model
        state_dim: State dimension (8 for LunarLander base obs)
        action_dim: Action dimension (2 for continuous control)
        hidden_dim: Hidden layer size
        batch_size: Training batch size
        num_steps: Number of gradient steps
        lr: Learning rate
        device: 'cuda' or 'cpu'
        max_samples: Maximum number of samples to use (None = all)
    """
    # Create dataset and dataloader
    dataset = DynamicsDataset(replay_dir, state_dim, action_dim, max_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Create model
    model = DynamicsModel(state_dim, action_dim, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    model.train()
    step = 0
    losses = []

    print(f"\nTraining dynamics model for {num_steps} steps...")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")

    pbar = tqdm(total=num_steps)

    while step < num_steps:
        for state_t, action_t, delta_true in dataloader:
            if step >= num_steps:
                break

            # Move to device
            state_t = state_t.to(device)
            action_t = action_t.to(device)
            delta_true = delta_true.to(device)

            # Forward pass
            delta_pred = model(state_t, action_t)
            loss = criterion(delta_pred, delta_true)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss
            losses.append(loss.item())

            # Update progress
            if step % 100 == 0:
                avg_loss = np.mean(losses[-100:]) if losses else 0.0
                pbar.set_postfix({'loss': f'{avg_loss:.6f}'})

            pbar.update(1)
            step += 1

    pbar.close()

    # Save model
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'state_dim': state_dim,
        'action_dim': action_dim,
        'hidden_dim': hidden_dim,
        'final_loss': np.mean(losses[-100:]) if losses else 0.0
    }, output_path)

    print(f"\nModel saved to {output_path}")
    print(f"Final average loss: {np.mean(losses[-100:]):.6f}")


def main():
    parser = argparse.ArgumentParser(description='Train dynamics model for MPC')
    parser.add_argument('--replay-dir', type=str, required=True,
                        help='Path to replay buffer directory')
    parser.add_argument('--out-path', type=str, required=True,
                        help='Output path for trained model (.pt file)')
    parser.add_argument('--state-dim', type=int, default=8,
                        help='State dimension (default: 8 for LunarLander base obs)')
    parser.add_argument('--action-dim', type=int, default=2,
                        help='Action dimension (default: 2)')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden layer dimension')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Training batch size')
    parser.add_argument('--num-steps', type=int, default=100_000,
                        help='Number of training steps')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to train on')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples (None = all)')

    args = parser.parse_args()

    train_dynamics_model(
        replay_dir=args.replay_dir,
        output_path=args.out_path,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        lr=args.lr,
        device=args.device,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
