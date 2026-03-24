"""Open-loop lunar lander trajectory optimization using Drake's KTO.

Ported from poslathian/lunar_lander/solver.py

Uses KinematicTrajectoryOptimization to find a smooth [x, y, theta] B-spline
trajectory from a start position down to the landing pad. Nonlinear dynamics
constraints at sampled times ensure every acceleration along the path is
achievable within the lander's thrust limits. Circular obstacles are avoided
via sampled distance constraints.

Physics model (simplified 2D rigid-body rocket):

    x''  = (-Fm sin theta  +  Fs cos theta) / m
    y''  = ( Fm cos theta  +  Fs sin theta) / m  -  g
    th'' =   Fs * arm / I

Given (x'', y'', theta) we can invert for the required thrusts:

    Fm =  m * ( -x'' sin theta  + (y'' + g) cos theta )
    Fs =  m * (  x'' cos theta  + (y'' + g) sin theta )

and the torque-consistency requirement:

    th''  ==  Fs * arm / I
"""

import numpy as np
from pydrake.planning import KinematicTrajectoryOptimization
from pydrake.solvers import Solve, SnoptSolver, SolverOptions
from pydrake.trajectories import BsplineTrajectory
from scipy.interpolate import BSpline

# ── World geometry (matches Gymnasium LunarLander viewport) ──────────────
SCALE = 30.0
VIEWPORT_W, VIEWPORT_H = 600, 400
W = VIEWPORT_W / SCALE  # 20.0  world units
H = VIEWPORT_H / SCALE  # 13.33 world units
PAD_X = W / 2            # 10.0
PAD_Y = H / 4            # 3.33

# ── Simplified 2D rocket physics ─────────────────────────────────────────
GRAVITY = 10.0
MASS = 1.0
INERTIA = 0.2
THRUST_MAX = 2.0 * MASS * GRAVITY   # main engine  (body +y)
SIDE_MAX = 0.5 * MASS * GRAVITY     # side engines  (body +/-x)
SIDE_ARM = 1.0                       # side-engine moment arm

# ── Default boundary conditions ──────────────────────────────────────────
START = np.array([PAD_X + 4.0, H - 0.5, 0.0])
GOAL = np.array([PAD_X, PAD_Y, 0.0])


# ─────────────────────────────────────────────────────────────────────────
# B-spline evaluation
# ─────────────────────────────────────────────────────────────────────────

def _basis_weights(basis, s, deriv=0):
    """Evaluate each B-spline basis function (or its k-th derivative) at s.

    Returns a length-N weight vector w such that  r^(k)(s) = control_points @ w.
    """
    knots = np.array(basis.knots())
    degree = basis.order() - 1
    n = basis.num_basis_functions()
    w = np.zeros(n)
    for i in range(n):
        c = np.zeros(n)
        c[i] = 1.0
        b = BSpline(knots, c, degree)
        if deriv:
            b = b.derivative(deriv)
        w[i] = float(b(s))
    return w


# ─────────────────────────────────────────────────────────────────────────
# Trajectory optimization
# ─────────────────────────────────────────────────────────────────────────

def solve(start=None, goal=None, obstacles=(),
          num_control_points=20, num_dynamics_samples=40,
          on_progress=None,
          max_warmstart_iters=0, max_obstacle_iters=12):
    """Plan a minimum-time landing trajectory.

    Parameters
    ----------
    start, goal : array-like (3,), optional
        [x, y, theta].  Defaults to module-level START / GOAL.
    obstacles : sequence of (cx, cy, radius) tuples
        Circular no-fly zones the trajectory must avoid.
    on_progress : callable(phase_str, frac) or None
        Called at milestones so callers can update a UI.
    max_warmstart_iters, max_obstacle_iters : int
        SNOPT major-iteration caps for each phase (~0.08 s per iter).

    Uses a two-phase approach when obstacles are present:
      1. Solve without obstacles, capped at *max_warmstart_iters*
      2. Re-solve with obstacle constraints seeded from phase 1,
         capped at *max_obstacle_iters*
    """
    start = np.asarray(start if start is not None else START, dtype=float)
    goal = np.asarray(goal if goal is not None else GOAL, dtype=float)

    def _report(phase, frac):
        if on_progress:
            on_progress(phase, frac)

    def _build(initial_guess_traj=None):
        """Build a fresh KTO + MathematicalProgram."""
        kto = KinematicTrajectoryOptimization(
            num_positions=3,
            num_control_points=num_control_points,
            spline_order=4,
            duration=3.0,
        )
        prog = kto.get_mutable_prog()

        kto.AddPathPositionConstraint(start, start, 0.0)
        kto.AddPathPositionConstraint(goal, goal, 1.0)
        z = np.zeros((3, 1))
        kto.AddPathVelocityConstraint(z, z, 0.0)
        kto.AddPathVelocityConstraint(z, z, 1.0)

        kto.AddPositionBounds(
            np.array([0.0, PAD_Y - 1.0, -np.pi / 3]),
            np.array([W,   H + 1.0,      np.pi / 3]),
        )
        kto.AddVelocityBounds(
            np.array([-5.0, -10.0, -3.0]),
            np.array([ 5.0,   1.0,  3.0]),
        )
        a_max = THRUST_MAX / MASS + GRAVITY
        alpha_max = SIDE_MAX * SIDE_ARM / INERTIA
        kto.AddAccelerationBounds(
            np.array([-a_max, -a_max, -alpha_max]),
            np.array([ a_max,  a_max,  alpha_max]),
        )
        kto.AddDurationConstraint(2.0, 10.0)
        kto.AddDurationCost(1.0)
        kto.AddPathEnergyCost(1.0)

        if initial_guess_traj is not None:
            kto.SetInitialGuess(initial_guess_traj)
        else:
            n_cp = kto.num_control_points()
            cp0 = np.column_stack(
                [start + (goal - start) * t for t in np.linspace(0, 1, n_cp)]
            )
            kto.SetInitialGuess(BsplineTrajectory(kto.basis(), cp0))

        _add_dynamics_constraints(kto, prog, num_dynamics_samples)
        return kto, prog

    # Phase 1: solve without obstacles (fast, typically <1s)
    _report("building", 0.0)
    kto, prog = _build()
    _report("warm-start", 0.2)
    if max_warmstart_iters:
        opts1 = SolverOptions()
        opts1.SetOption(SnoptSolver.id(), "Major iterations limit",
                        max_warmstart_iters)
        result = Solve(prog, solver_options=opts1)
    else:
        result = Solve(prog)
    warm_traj = kto.ReconstructTrajectory(result)

    if not obstacles:
        _report("sampling", 0.9)
        out = _sample(warm_traj, n_constraint_pts=num_dynamics_samples)
        _report("done", 1.0)
        return out

    # Phase 2: re-solve with obstacles, capped at max_obstacle_iters.
    # Even if SNOPT doesn't fully converge, the best iterate is a smooth
    # B-spline that's usually quite close to feasible.
    _report("obstacles", 0.4)
    kto2, prog2 = _build(initial_guess_traj=warm_traj)
    _add_obstacle_constraints(kto2, prog2, obstacles, num_dynamics_samples)

    _report("solving", 0.6)
    opts = SolverOptions()
    opts.SetOption(SnoptSolver.id(), "Major iterations limit",
                   max_obstacle_iters)
    result2 = Solve(prog2, solver_options=opts)
    best_traj = kto2.ReconstructTrajectory(result2)

    if not result2.is_success():
        # Check if the best iterate is reasonable (goal reached)
        p2_end = best_traj.value(best_traj.end_time()).flatten()
        goal_err = np.linalg.norm(p2_end - goal)
        if goal_err > 1.0:
            print("WARNING: obstacle solve diverged, using warm-start plan")
            best_traj = warm_traj
        else:
            print(f"NOTE: obstacle solve used {max_obstacle_iters} iters "
                  f"(best-effort, goal err={goal_err:.3f})")

    _report("sampling", 0.9)
    out = _sample(best_traj, n_constraint_pts=num_dynamics_samples)
    _report("done", 1.0)
    return out


# ─────────────────────────────────────────────────────────────────────────
# Constraint helpers
# ─────────────────────────────────────────────────────────────────────────

def _add_dynamics_constraints(kto, prog, n_samples):
    """Constrain implied thrusts to be within physical limits at each sample.

    Outputs per sample:  [Fm, Fs, torque_error]
    Bounds:              [0, THRUST_MAX] x [-SIDE_MAX, SIDE_MAX] x {0}
    """
    cp = kto.control_points()
    T = kto.duration()
    basis = kto.basis()
    n_cp = kto.num_control_points()
    all_vars = np.concatenate([cp.flatten(), [T]])

    TORQUE_TOL = 0.05  # small tolerance on torque consistency
    lb = np.array([0.0,       -SIDE_MAX, -TORQUE_TOL])
    ub = np.array([THRUST_MAX, SIDE_MAX,  TORQUE_TOL])

    for s in np.linspace(0, 1, n_samples):
        w_pos = _basis_weights(basis, s, deriv=0)
        w_acc = _basis_weights(basis, s, deriv=2)

        def _make(w_pos, w_acc):
            def constraint(v):
                P = v[:-1].reshape(3, n_cp)
                dur = v[-1]
                pos = P @ w_pos
                acc = P @ w_acc / dur ** 2

                ct, st = np.cos(pos[2]), np.sin(pos[2])
                Fm = MASS * (-acc[0] * st + (acc[1] + GRAVITY) * ct)
                Fs = MASS * ( acc[0] * ct + (acc[1] + GRAVITY) * st)
                torque_err = acc[2] - Fs * SIDE_ARM / INERTIA
                return np.array([Fm, Fs, torque_err])
            return constraint

        prog.AddConstraint(_make(w_pos, w_acc), lb, ub, all_vars)


def _add_obstacle_constraints(kto, prog, obstacles, n_samples):
    """For each obstacle, constrain  (x-cx)^2 + (y-cy)^2 >= (r+margin)^2
    at every sample point along the path.

    Uses squared distance (no sqrt) so the constraint stays smooth everywhere.
    Only depends on control-point variables (not duration).
    """
    if not obstacles:
        return
    basis = kto.basis()
    n_cp = kto.num_control_points()
    cp_flat = kto.control_points().flatten()
    MARGIN = 0.3  # safety buffer around each obstacle

    for cx, cy, r in obstacles:
        r_eff = r + MARGIN
        for s in np.linspace(0, 1, n_samples):
            w = _basis_weights(basis, s, deriv=0)

            def _make(w, cx, cy, r_eff):
                def constraint(v):
                    P = v.reshape(3, n_cp)
                    pos = P @ w
                    dx, dy = pos[0] - cx, pos[1] - cy
                    return np.array([dx * dx + dy * dy - r_eff * r_eff])
                return constraint

            prog.AddConstraint(
                _make(w, cx, cy, r_eff),
                np.array([0.0]), np.array([np.inf]),
                cp_flat,
            )


# ─────────────────────────────────────────────────────────────────────────
# Trajectory sampling
# ─────────────────────────────────────────────────────────────────────────

def _sample(traj, n=300, n_constraint_pts=40):
    """Sample the solved trajectory, computing thrusts via inverse dynamics.

    Returns (times, plan_dict, constraint_xy) where constraint_xy is an
    (n_constraint_pts, 2) array of the [x,y] positions at the spline
    parameter values where obstacle/dynamics constraints were enforced.
    """
    times = np.linspace(traj.start_time(), traj.end_time(), n)
    S = {k: np.empty(n) for k in
         ("x", "y", "theta", "vx", "vy", "omega", "Fm", "Fs")}

    for i, t in enumerate(times):
        q   = traj.value(t).flatten()
        qd  = traj.EvalDerivative(t, 1).flatten()
        qdd = traj.EvalDerivative(t, 2).flatten()

        S["x"][i], S["y"][i], S["theta"][i] = q
        S["vx"][i], S["vy"][i], S["omega"][i] = qd

        ct, st = np.cos(q[2]), np.sin(q[2])
        S["Fm"][i] = MASS * (-qdd[0] * st + (qdd[1] + GRAVITY) * ct)
        S["Fs"][i] = MASS * ( qdd[0] * ct + (qdd[1] + GRAVITY) * st)

    # Sample the constraint enforcement points (same s values used in
    # _add_dynamics_constraints / _add_obstacle_constraints)
    t0, t1 = traj.start_time(), traj.end_time()
    cpts = np.empty((n_constraint_pts, 2))
    for i, s in enumerate(np.linspace(0, 1, n_constraint_pts)):
        t = t0 + s * (t1 - t0)
        q = traj.value(t).flatten()
        cpts[i] = q[:2]

    return times, S, cpts


if __name__ == "__main__":
    times, s, _ = solve()
    print(f"Duration     : {times[-1]:.2f} s")
    print(f"Final pos    : ({s['x'][-1]:.3f}, {s['y'][-1]:.3f})")
    print(f"Final vel    : ({s['vx'][-1]:.3f}, {s['vy'][-1]:.3f})")
    print(f"Thrust  Fm   : [{s['Fm'].min():.2f}, {s['Fm'].max():.2f}]")
    print(f"Thrust  Fs   : [{s['Fs'].min():.2f}, {s['Fs'].max():.2f}]")
