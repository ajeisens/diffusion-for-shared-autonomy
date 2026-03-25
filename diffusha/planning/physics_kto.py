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


def _derivs(s: np.ndarray, Fm: float, Fs: float) -> np.ndarray:
    """Compute state derivatives for [x, y, th, vx, vy, om]."""
    th = s[2]
    ct, st = np.cos(th), np.sin(th)
    ax = (-Fm * st + Fs * ct) / MASS
    ay = (Fm * ct + Fs * st) / MASS - GRAVITY
    alpha = Fs * SIDE_ARM / INERTIA
    return np.array([s[3], s[4], s[5], ax, ay, alpha])


def step_physics(state: Dict[str, float], Fm: float, Fs: float, dt: float) -> Dict[str, float]:
    """Integrate lander dynamics forward by dt seconds using RK4.

    RK4 is used instead of Euler because thrust direction depends on theta,
    so Euler methods accumulate large rotational errors during open-loop replay.

    Args:
        state: Dictionary with keys {x, y, theta, vx, vy, omega}
        Fm: Main engine thrust (body +y direction) [N]
        Fs: Side engine thrust (body +x direction) [N]
        dt: Time step [s]

    Returns:
        New state dictionary with updated position/velocity
    """
    s = np.array([state["x"], state["y"], state["theta"],
                  state["vx"], state["vy"], state["omega"]])
    k1 = _derivs(s, Fm, Fs)
    k2 = _derivs(s + 0.5 * dt * k1, Fm, Fs)
    k3 = _derivs(s + 0.5 * dt * k2, Fm, Fs)
    k4 = _derivs(s + dt * k3, Fm, Fs)
    s += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return dict(x=s[0], y=s[1], theta=s[2], vx=s[3], vy=s[4], omega=s[5])


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
