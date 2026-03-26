#!/usr/bin/env python3
"""
Episode Recorder for Lunar Lander Demo

Records episodes with success/collision annotations for diffusion model training.
Inspired by π*0.6 RECAP method for learning from heterogeneous data.
"""

import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any


class EpisodeRecorder:
    """
    Records episodes with success/collision annotations for diffusion training.

    Stores trajectories in format compatible with existing diffusion training pipeline.
    Each episode is saved as individual .pkl file with metadata annotations.
    """

    def __init__(self, save_dir: str = './recorded_episodes', env_name: str = 'LunarLanderObstacle-v5'):
        """
        Initialize episode recorder.

        Args:
            save_dir: Directory to save recorded episodes
            env_name: Environment name for metadata
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.env_name = env_name
        self.metadata_file = self.save_dir / 'metadata.json'

        # Current episode buffer
        self.current_episode = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'infos': [],
        }
        self.current_mode = None
        self.current_initial_obs = None

        # Statistics tracking
        self.episodes = []
        self.stats = {
            'total_episodes': 0,
            'teleop':    {'count': 0, 'success_count': 0, 'collision_count': 0, 'total_return': 0.0, 'total_length': 0},
            'heuristic': {'count': 0, 'success_count': 0, 'collision_count': 0, 'total_return': 0.0, 'total_length': 0},
            'kto':       {'count': 0, 'success_count': 0, 'collision_count': 0, 'total_return': 0.0, 'total_length': 0},
            'mpc':       {'count': 0, 'success_count': 0, 'collision_count': 0, 'total_return': 0.0, 'total_length': 0},
            'assisted':  {'count': 0, 'success_count': 0, 'collision_count': 0, 'total_return': 0.0, 'total_length': 0},
        }

        # Load existing metadata if available
        self._load_metadata()

    def _load_metadata(self):
        """Load existing metadata from disk"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    self.episodes = data.get('episodes', [])
                    self.stats = data.get('statistics', self.stats)
                    # Update total_episodes count
                    self.stats['total_episodes'] = len(self.episodes)
                print(f"Loaded {len(self.episodes)} existing episodes from metadata")
            except Exception as e:
                print(f"Warning: Could not load metadata: {e}")

    def _save_metadata(self):
        """Save metadata summary to disk"""
        try:
            # Compute statistics per mode
            teleop_episodes    = [e for e in self.episodes if e['mode'] == 'teleop']
            heuristic_episodes = [e for e in self.episodes if e['mode'] == 'heuristic']
            kto_episodes       = [e for e in self.episodes if e['mode'] == 'kto']
            mpc_episodes       = [e for e in self.episodes if e['mode'] == 'mpc']
            assisted_episodes  = [e for e in self.episodes if e['mode'] == 'assisted']

            def compute_stats(episodes):
                if not episodes:
                    return {
                        'count': 0,
                        'success_rate': 0.0,
                        'collision_rate': 0.0,
                        'crash_rate': 0.0,
                        'avg_return': 0.0,
                        'avg_length': 0.0
                    }
                return {
                    'count': len(episodes),
                    'success_rate': sum(e['success'] for e in episodes) / len(episodes),
                    'collision_rate': sum(e['collision'] for e in episodes) / len(episodes),
                    'crash_rate': sum(e.get('crashed', False) for e in episodes) / len(episodes),
                    'avg_return': sum(e['return'] for e in episodes) / len(episodes),
                    'avg_length': sum(e['length'] for e in episodes) / len(episodes),
                }

            metadata = {
                'env_name': self.env_name,
                'total_episodes': len(self.episodes),
                'episodes': self.episodes,
                'statistics': {
                    'teleop':    compute_stats(teleop_episodes),
                    'heuristic': compute_stats(heuristic_episodes),
                    'kto':       compute_stats(kto_episodes),
                    'mpc':       compute_stats(mpc_episodes),
                    'assisted':  compute_stats(assisted_episodes),
                }
            }

            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Update internal stats
            self.stats = metadata['statistics']
            self.stats['total_episodes'] = len(self.episodes)

        except Exception as e:
            print(f"Warning: Could not save metadata: {e}")

    def start_episode(self, mode: str, initial_obs: Optional[np.ndarray] = None):
        """
        Start recording a new episode.

        Args:
            mode: Control mode ('teleop', 'heuristic', 'kto', or 'mpc')
            initial_obs: Initial observation (optional, for first recording)
        """
        # Reset episode buffer
        self.current_episode = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'infos': [],
        }
        self.current_mode = mode
        self.current_initial_obs = initial_obs

    def record_step(self, obs: np.ndarray, action: np.ndarray, reward: float, done: bool, info: Dict):
        """
        Record a single transition.

        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            done: Episode termination flag
            info: Environment info dict
        """
        self.current_episode['observations'].append(obs.copy())
        self.current_episode['actions'].append(action.copy() if isinstance(action, np.ndarray) else np.array(action))
        self.current_episode['rewards'].append(reward)
        self.current_episode['dones'].append(done)
        self.current_episode['infos'].append(info.copy())

    def end_episode(self) -> Optional[Dict]:
        """
        End current episode and save to disk.

        Returns:
            Episode metadata dict, or None if episode is empty
        """
        # Check if episode has any transitions
        if len(self.current_episode['observations']) == 0:
            return None

        # Convert lists to numpy arrays
        observations = np.array(self.current_episode['observations'], dtype=np.float32)
        actions = np.array(self.current_episode['actions'], dtype=np.float32)
        rewards = np.array(self.current_episode['rewards'], dtype=np.float32)
        dones = np.array(self.current_episode['dones'], dtype=bool)

        # Compute episode statistics
        episode_return = float(np.sum(rewards))
        episode_length = len(rewards)

        # Extract success/collision/crashed from final info
        final_info = self.current_episode['infos'][-1] if self.current_episode['infos'] else {}

        # Success detection (from eval.py patterns)
        success = ("goal" in final_info and
                   (final_info["goal"] == "landed" or final_info["goal"] == "target-reached"))

        # Collision detection (hit obstacle)
        collision = final_info.get('collision', False)

        # Crash detection (hit terrain, OOB, underground, etc.)
        crashed = final_info.get('crashed', False)

        # Create episode data structure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"episode_{timestamp}_{self.current_mode}.pkl"

        # Base metadata
        metadata = {
            'mode': self.current_mode,
            'success': bool(success),
            'collision': bool(collision),
            'crashed': bool(crashed),
            'episode_return': episode_return,
            'episode_length': episode_length,
            'timestamp': timestamp,
            'env_name': self.env_name,
        }

        # Add KTO-specific metadata if available
        if 'kto_metadata' in final_info:
            metadata['kto'] = final_info['kto_metadata']

        episode_data = {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'infos': self.current_episode['infos'],
            'metadata': metadata
        }

        # Save episode to disk
        try:
            episode_path = self.save_dir / filename
            torch.save(episode_data, episode_path)
            print(f"Saved episode: {filename} (return={episode_return:.1f}, success={success}, collision={collision}, crashed={crashed})")
        except Exception as e:
            print(f"Error saving episode: {e}")
            return None

        # Update metadata
        episode_metadata = {
            'filename': filename,
            'mode': self.current_mode,
            'success': bool(success),
            'collision': bool(collision),
            'crashed': bool(crashed),
            'return': episode_return,
            'length': episode_length,
            'timestamp': timestamp,
        }
        self.episodes.append(episode_metadata)

        # Save updated metadata
        self._save_metadata()

        return episode_metadata

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current recording statistics.

        Returns:
            Dictionary with statistics including success/collision rates
        """
        return {
            'total_episodes': self.stats['total_episodes'],
            'success_rate': self._compute_overall_rate('success_rate'),
            'collision_rate': self._compute_overall_rate('collision_rate'),
            'crash_rate': self._compute_overall_rate('crash_rate'),
            'avg_return': self._compute_overall_avg('avg_return'),
            'avg_length': self._compute_overall_avg('avg_length'),
            'teleop': self.stats.get('teleop', {}),
            'heuristic': self.stats.get('heuristic', {}),
            'kto': self.stats.get('kto', {}),
            'mpc': self.stats.get('mpc', {}),
            'assisted': self.stats.get('assisted', {}),
        }

    def _compute_overall_rate(self, key: str) -> float:
        """Compute overall rate across all modes"""
        modes = ['teleop', 'heuristic', 'kto', 'mpc', 'assisted']
        total_count = sum(self.stats.get(mode, {}).get('count', 0) for mode in modes)

        if total_count == 0:
            return 0.0

        weighted_sum = sum(
            self.stats.get(mode, {}).get(key, 0.0) * self.stats.get(mode, {}).get('count', 0)
            for mode in modes
        )

        return weighted_sum / total_count

    def _compute_overall_avg(self, key: str) -> float:
        """Compute overall average across all modes"""
        modes = ['teleop', 'heuristic', 'kto', 'mpc', 'assisted']
        total_count = sum(self.stats.get(mode, {}).get('count', 0) for mode in modes)

        if total_count == 0:
            return 0.0

        weighted_sum = sum(
            self.stats.get(mode, {}).get(key, 0.0) * self.stats.get(mode, {}).get('count', 0)
            for mode in modes
        )

        return weighted_sum / total_count

    def get_episode_count(self) -> int:
        """Get total number of recorded episodes"""
        return self.stats['total_episodes']

    def clear_all_episodes(self):
        """Clear all recorded episodes and reset statistics (use with caution!)"""
        import shutil

        # Remove all episode files
        for episode_file in self.save_dir.glob("episode_*.pkl"):
            episode_file.unlink()

        # Reset statistics
        self.episodes = []
        self.stats = {
            'total_episodes': 0,
            'teleop':    {'count': 0, 'success_count': 0, 'collision_count': 0, 'total_return': 0.0, 'total_length': 0},
            'heuristic': {'count': 0, 'success_count': 0, 'collision_count': 0, 'total_return': 0.0, 'total_length': 0},
            'kto':       {'count': 0, 'success_count': 0, 'collision_count': 0, 'total_return': 0.0, 'total_length': 0},
            'mpc':       {'count': 0, 'success_count': 0, 'collision_count': 0, 'total_return': 0.0, 'total_length': 0},
            'assisted':  {'count': 0, 'success_count': 0, 'collision_count': 0, 'total_return': 0.0, 'total_length': 0},
        }

        # Save empty metadata
        self._save_metadata()

        print("Cleared all recorded episodes")
