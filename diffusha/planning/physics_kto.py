"""Physics simulation for KTO lunar lander.

Simplified 2D rocket dynamics extracted from poslathian/lunar_lander/game.py.
"""

import numpy as np
from typing import Dict

# Physics constants (same as kto_solver.py to avoid Drake dependency)
GRAVITY = 10.0
MASS = 1.0
INERTIA = 0.2
SIDE_ARM = 1.0


def step_physics(state: Dict[str, float], Fm: float, Fs: float, dt: float) -> Dict[str, float]:
    """Integrate lander dynamics forward by dt seconds.

    Args:
        state: Dictionary with keys {x, y, theta, vx, vy, omega}
        Fm: Main engine thrust (body +y direction) [N]
        Fs: Side engine thrust (body +x direction) [N]
        dt: Time step [s]

    Returns:
        New state dictionary with updated position/velocity
    """
    x, y, th = state["x"], state["y"], state["theta"]
    vx, vy, om = state["vx"], state["vy"], state["omega"]

    ct, st = np.cos(th), np.sin(th)

    # Acceleration in world frame
    ax = (-Fm * st + Fs * ct) / MASS
    ay = (Fm * ct + Fs * st) / MASS - GRAVITY
    alpha = Fs * SIDE_ARM / INERTIA

    # Euler integration
    vx += ax * dt
    vy += ay * dt
    om += alpha * dt

    x += vx * dt
    y += vy * dt
    th += om * dt

    return dict(x=x, y=y, theta=th, vx=vx, vy=vy, omega=om)


def compute_inverse_dynamics(state: Dict[str, float],
                            acc: np.ndarray) -> tuple[float, float]:
    """Compute required thrusts to achieve desired acceleration.

    Given a state and desired acceleration [ax, ay, alpha], compute
    the main and side engine thrusts needed.

    Args:
        state: Dictionary with keys {x, y, theta, vx, vy, omega}
        acc: Desired acceleration [ax, ay, alpha] in world frame

    Returns:
        (Fm, Fs): Main and side engine thrusts [N]
    """
    th = state["theta"]
    ax, ay, alpha = acc

    ct, st = np.cos(th), np.sin(th)

    # Invert dynamics equations
    Fm = MASS * (-ax * st + (ay + GRAVITY) * ct)
    Fs = MASS * ( ax * ct + (ay + GRAVITY) * st)

    return Fm, Fs


def hovering_thrust(theta: float) -> tuple[float, float]:
    """Compute thrust needed to hover at given angle.

    Args:
        theta: Lander angle [rad]

    Returns:
        (Fm, Fs): Main and side engine thrusts to counteract gravity
    """
    ct, st = np.cos(theta), np.sin(theta)

    # Zero acceleration except counteracting gravity
    Fm = MASS * GRAVITY * ct
    Fs = MASS * GRAVITY * st

    return Fm, Fs
