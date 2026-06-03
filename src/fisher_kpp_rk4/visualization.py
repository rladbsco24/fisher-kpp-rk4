import numpy as np


def circular_initial_condition(x, y, radius=35.0, width=5.0):
    xx, yy = np.meshgrid(x, y, indexing="xy")
    cx = 0.5 * (x[0] + x[-1])
    cy = 0.5 * (y[0] + y[-1])
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return 1.0 / (1.0 + np.exp((rr - radius) / width))


def fisher_kpp_rhs_2d(u, dx, dy, D, r):
    u_pad = np.pad(u, pad_width=1, mode="edge")
    u_xx = (u_pad[1:-1, 2:] - 2.0 * u + u_pad[1:-1, :-2]) / dx**2
    u_yy = (u_pad[2:, 1:-1] - 2.0 * u + u_pad[:-2, 1:-1]) / dy**2
    return D * (u_xx + u_yy) + r * u * (1.0 - u)


def rk4_step_2d(u, dt, dx, dy, D, r):
    k1 = fisher_kpp_rhs_2d(u, dx, dy, D, r)
    k2 = fisher_kpp_rhs_2d(u + 0.5 * dt * k1, dx, dy, D, r)
    k3 = fisher_kpp_rhs_2d(u + 0.5 * dt * k2, dx, dy, D, r)
    k4 = fisher_kpp_rhs_2d(u + dt * k3, dx, dy, D, r)
    u_next = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return np.clip(u_next, 0.0, 1.0)


def solve_2d_fisher_kpp_snapshots(
    L=200.0,
    n=201,
    T=60.0,
    dt=0.05,
    D=1.0,
    r=1.0,
    snapshot_times=(0.0, 15.0, 30.0, 45.0, 60.0),
):
    x = np.linspace(0.0, L, n)
    y = np.linspace(0.0, L, n)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    Nt = int(round(T / dt))
    dt = T / Nt

    targets = np.asarray(snapshot_times, dtype=float)
    target_steps = np.rint(targets / dt).astype(int)
    snapshots = {}

    u = circular_initial_condition(x, y)
    for step in range(Nt + 1):
        if step in target_steps:
            t = step * dt
            snapshots[float(targets[np.argmin(np.abs(target_steps - step))])] = u.copy()
        if step == Nt:
            break
        u = rk4_step_2d(u, dt, dx, dy, D, r)

    return {
        "x": x,
        "y": y,
        "times": targets,
        "snapshots": np.asarray([snapshots[float(t)] for t in targets]),
        "D": D,
        "r": r,
        "dt": dt,
        "dx": dx,
        "dy": dy,
    }


def plot_cross_section_view(result, output_path, title="2D Fisher-KPP Equation - Cross-Section View"):
    import matplotlib.pyplot as plt

    x = result["x"]
    y = result["y"]
    times = result["times"]
    snapshots = result["snapshots"]

    fig, axes = plt.subplots(1, len(times), figsize=(16, 3.8), constrained_layout=True)
    image = None

    for ax, t, u in zip(axes, times, snapshots):
        image = ax.imshow(
            u,
            origin="lower",
            extent=[x[0], x[-1], y[0], y[-1]],
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            aspect="equal",
        )
        ax.set_title(f"Time = {t:.1f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(title, fontsize=15)
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.92)
    cbar.set_label("Density (u)")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
