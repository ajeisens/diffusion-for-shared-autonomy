"""Async wrapper for KTO trajectory planner with progress tracking.

Enables non-blocking trajectory planning with UI progress updates.
Ported from poslathian/lunar_lander/game.py:_solve_with_progress
"""

import threading
import time
from typing import Optional, Tuple, Dict, Callable, List

# Import will fail if Drake not installed - caller should handle
try:
    from .kto_solver import solve
    DRAKE_AVAILABLE = True
except ImportError:
    DRAKE_AVAILABLE = False
    solve = None


class AsyncKTOPlanner:
    """Background thread wrapper for KTO trajectory optimization.

    Usage:
        planner = AsyncKTOPlanner()
        planner.start_planning(start, goal, obstacles)

        while planner.is_planning():
            progress = planner.get_progress()
            # Draw progress bar with progress['phase'] and progress['frac']

        result = planner.get_result()
        times, plan, constraint_xy = result
    """

    def __init__(self, max_warmstart_iters: int = 0, max_obstacle_iters: int = 12):
        """Initialize async planner.

        Args:
            max_warmstart_iters: SNOPT iteration cap for warmstart phase (0=uncapped)
            max_obstacle_iters: SNOPT iteration cap for obstacle phase (~0.08s per iter)
        """
        if not DRAKE_AVAILABLE:
            raise ImportError(
                "Drake is not installed. Cannot use KTO planner.\n"
                "Install Drake in a Python 3.10-3.12 environment:\n"
                "  pip install drake>=1.30.0"
            )

        self.max_warmstart_iters = max_warmstart_iters
        self.max_obstacle_iters = max_obstacle_iters

        # Thread state
        self.thread: Optional[threading.Thread] = None
        self.progress: Dict[str, any] = {"phase": "idle", "frac": 0.0}
        self.result_holder: Dict[str, any] = {
            "data": None,
            "error": None,
            "elapsed": 0.0
        }

    def start_planning(self, start, goal, obstacles=(), num_control_points=20,
                      num_dynamics_samples=40):
        """Launch planning in background thread.

        Args:
            start: [x, y, theta] start pose
            goal: [x, y, theta] goal pose
            obstacles: List of (cx, cy, radius) tuples
            num_control_points: B-spline control points (default: 20)
            num_dynamics_samples: Constraint sample points (default: 40)
        """
        if self.thread and self.thread.is_alive():
            raise RuntimeError("Planning already in progress")

        # Reset state
        self.progress = {"phase": "starting", "frac": 0.0}
        self.result_holder = {"data": None, "error": None, "elapsed": 0.0}

        def _on_progress(phase: str, frac: float):
            """Progress callback from solver."""
            self.progress["phase"] = phase
            self.progress["frac"] = frac

        def _worker():
            """Background worker thread."""
            try:
                t0 = time.monotonic()
                result = solve(
                    start=start,
                    goal=goal,
                    obstacles=obstacles,
                    num_control_points=num_control_points,
                    num_dynamics_samples=num_dynamics_samples,
                    on_progress=_on_progress,
                    max_warmstart_iters=self.max_warmstart_iters,
                    max_obstacle_iters=self.max_obstacle_iters,
                )
                self.result_holder["data"] = result
                self.result_holder["elapsed"] = time.monotonic() - t0
            except Exception as e:
                self.result_holder["error"] = e

        self.thread = threading.Thread(target=_worker, daemon=True)
        self.thread.start()

    def is_planning(self) -> bool:
        """Check if planning is still in progress."""
        return self.thread is not None and self.thread.is_alive()

    def get_progress(self) -> Dict[str, any]:
        """Get current planning progress.

        Returns:
            Dict with keys:
            - 'phase': str (e.g., 'warmstart', 'obstacles', 'sampling', 'done')
            - 'frac': float in [0, 1]
        """
        return self.progress.copy()

    def get_solve_time(self) -> float:
        """Get solver elapsed time in seconds (after completion)."""
        return self.result_holder["elapsed"]

    def get_result(self) -> Tuple:
        """Get planning result after completion.

        Returns:
            (times, plan_dict, constraint_xy) tuple where:
            - times: np.ndarray of timesteps
            - plan_dict: dict with keys ['x', 'y', 'theta', 'vx', 'vy', 'omega', 'Fm', 'Fs']
            - constraint_xy: (N, 2) array of constraint enforcement points

        Raises:
            RuntimeError: If planning still in progress
            Exception: If solver raised an exception
        """
        if self.is_planning():
            raise RuntimeError("Planning still in progress, call after is_planning() returns False")

        if self.result_holder["error"]:
            raise self.result_holder["error"]

        if self.result_holder["data"] is None:
            raise RuntimeError("No planning result available")

        return self.result_holder["data"]

    def wait(self, timeout: Optional[float] = None):
        """Block until planning completes.

        Args:
            timeout: Max wait time in seconds (None = infinite)
        """
        if self.thread:
            self.thread.join(timeout=timeout)
