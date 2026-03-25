"""Async wrapper for KTO trajectory planner with progress tracking.

Enables non-blocking trajectory planning with UI progress updates.
Ported from poslathian/lunar_lander/game.py:_solve_with_progress
"""

import threading
import time
from typing import Optional, Tuple, Dict

# Import will fail if Drake not installed - caller should handle
try:
    from .kto_solver import solve, STRATEGIES
    DRAKE_AVAILABLE = True
except ImportError:
    DRAKE_AVAILABLE = False
    solve = None
    STRATEGIES = {}


class AsyncKTOPlanner:
    """Background thread wrapper for KTO trajectory optimization.

    Usage:
        planner = AsyncKTOPlanner()
        planner.start_planning(start, goal, obstacles)

        while planner.is_planning():
            progress = planner.get_progress()
            # Draw progress bar with progress['phase'] and progress['frac']

        times, plan, constraint_xy, warm_xy, knot_xy, control_xy = planner.get_result()
    """

    def __init__(self, time_budget: float = 1.0, warmstart_budget: float = 0.33,
                 strategy: str = "default"):
        """Initialize async planner.

        Args:
            time_budget: Total wall-clock seconds for both solver phases.
            warmstart_budget: Max seconds for warm-start phase.
                Unused time is donated to the obstacle phase.
            strategy: Solver strategy name (see kto_solver.STRATEGIES).
        """
        if not DRAKE_AVAILABLE:
            raise ImportError(
                "Drake is not installed. Cannot use KTO planner.\n"
                "Install Drake in a Python 3.10-3.12 environment:\n"
                "  pip install drake>=1.30.0"
            )

        self.time_budget = time_budget
        self.warmstart_budget = warmstart_budget
        self.strategy = strategy

        # Thread state
        self.thread: Optional[threading.Thread] = None
        self.progress: Dict = {"phase": "idle", "frac": 0.0}
        self.result_holder: Dict = {
            "data": None,
            "error": None,
            "elapsed": 0.0
        }

    def start_planning(self, start, goal, obstacles=(), terrain=None):
        """Launch planning in background thread.

        Args:
            start: [x, y, theta] start pose
            goal: [x, y, theta] goal pose
            obstacles: List of (cx, cy, radius) tuples
            terrain: (txs, tys) terrain arrays for ground-avoidance, or None
        """
        if self.thread and self.thread.is_alive():
            raise RuntimeError("Planning already in progress")

        # Reset state
        self.progress = {"phase": "starting", "frac": 0.0}
        self.result_holder = {"data": None, "error": None, "elapsed": 0.0}

        def _on_progress(phase: str, frac: float):
            self.progress["phase"] = phase
            self.progress["frac"] = frac

        def _worker():
            try:
                t0 = time.monotonic()
                result = solve(
                    start=start,
                    goal=goal,
                    obstacles=obstacles,
                    terrain=terrain,
                    on_progress=_on_progress,
                    time_budget=self.time_budget,
                    warmstart_budget=self.warmstart_budget,
                    strategy=self.strategy,
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

    def get_progress(self) -> Dict:
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
            (times, plan_dict, constraint_xy, warm_xy, knot_xy, control_xy) where:
            - times: np.ndarray of timesteps
            - plan_dict: dict with keys ['x', 'y', 'theta', 'vx', 'vy', 'omega',
                                         'Fm', 'Fs', 'ax', 'ay', 'alpha']
            - constraint_xy: (N, 2) array of constraint enforcement points
            - warm_xy: (300, 2) array of warm-start path positions
            - knot_xy: (n_knots, 2) array of B-spline knot positions
            - control_xy: (n_cp, 2) array of B-spline control point positions

        Raises:
            RuntimeError: If planning still in progress or no result available
            Exception: If solver raised an exception
        """
        if self.is_planning():
            raise RuntimeError("Planning still in progress")

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
