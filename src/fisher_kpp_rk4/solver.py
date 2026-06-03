import numpy as np


def apply_dirichlet_bc(u, left_bc, right_bc):
    u = np.asarray(u, dtype=float).copy()
    u[0] = left_bc
    u[-1] = right_bc
    return u


def fisher_kpp_rhs(u, dx, D, r):
    """Semi-discrete RHS: u_t = D u_xx + r u(1-u)."""
    dudt = np.zeros_like(u)
    lap = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dx**2
    reaction = r * u[1:-1] * (1.0 - u[1:-1])
    dudt[1:-1] = D * lap + reaction
    return dudt


def rk4_step(u, dt, dx, D, r, left_bc, right_bc):
    """One classical RK4 step for the MOL-FDM Fisher-KPP system."""
    u0 = apply_dirichlet_bc(u, left_bc, right_bc)

    k1 = fisher_kpp_rhs(u0, dx, D, r)

    u2 = apply_dirichlet_bc(u0 + 0.5 * dt * k1, left_bc, right_bc)
    k2 = fisher_kpp_rhs(u2, dx, D, r)

    u3 = apply_dirichlet_bc(u0 + 0.5 * dt * k2, left_bc, right_bc)
    k3 = fisher_kpp_rhs(u3, dx, D, r)

    u4 = apply_dirichlet_bc(u0 + dt * k3, left_bc, right_bc)
    k4 = fisher_kpp_rhs(u4, dx, D, r)

    u_next = u0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return apply_dirichlet_bc(u_next, left_bc, right_bc)


def check_rk4_stability(dx, dt, D, r, dim=1, safety=0.95):
    """Practical explicit RK4 stability estimate for diffusion-reaction setting."""
    if D <= 0 or r <= 0:
        raise ValueError("D and r must be positive.")
    dt_diff_limit = 0.69 * dx**2 / (dim * D)
    dt_reaction_scale = 1.0 / r
    dt_practical = safety * min(dt_diff_limit, dt_reaction_scale)
    return {
        "dt": dt,
        "dt_diff_limit": dt_diff_limit,
        "dt_reaction_scale": dt_reaction_scale,
        "dt_practical": dt_practical,
        "is_practically_safe": dt <= dt_practical,
    }


def front_position(x, u, level=0.5):
    """Linear interpolation of the first crossing u=level."""
    diff = u - level
    idx = np.where(diff[:-1] * diff[1:] <= 0.0)[0]
    if len(idx) == 0:
        return np.nan

    i = idx[0]
    x0, x1 = x[i], x[i + 1]
    u0, u1 = u[i], u[i + 1]
    if abs(u1 - u0) < 1e-14:
        return x0
    return x0 + (level - u0) * (x1 - x0) / (u1 - u0)


def relative_l2(u_num, u_ref):
    denom = np.linalg.norm(u_ref)
    if denom == 0.0:
        return np.nan
    return np.linalg.norm(u_num - u_ref) / denom


def solve_rk4(
    x,
    dt,
    Nt,
    D,
    r,
    initial_condition,
    left_bc,
    right_bc,
    save_interval=5.0,
):
    dx = x[1] - x[0]
    u = apply_dirichlet_bc(initial_condition(x), left_bc, right_bc)

    snapshots = []
    times = []
    fronts = []
    next_save_t = 0.0

    for n in range(Nt + 1):
        t = n * dt
        if t >= next_save_t - 1e-12 or n == Nt:
            snapshots.append(u.copy())
            times.append(t)
            fronts.append(front_position(x, u, level=0.5))
            next_save_t += save_interval

        if n == Nt:
            break
        u = rk4_step(u, dt, dx, D, r, left_bc, right_bc)

    return {
        "x": x,
        "times": np.asarray(times),
        "snapshots": np.asarray(snapshots),
        "fronts": np.asarray(fronts),
        "u_final": u,
    }


def estimate_front_speed(times, fronts, t_min=None, x_max=None):
    times = np.asarray(times)
    fronts = np.asarray(fronts)
    mask = np.isfinite(fronts)
    if t_min is not None:
        mask &= times >= t_min
    if x_max is not None:
        mask &= fronts <= x_max
    if np.count_nonzero(mask) < 2:
        return np.nan
    slope, _ = np.polyfit(times[mask], fronts[mask], deg=1)
    return slope
