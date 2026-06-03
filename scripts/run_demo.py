import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fisher_kpp_rk4 import check_rk4_stability, estimate_front_speed, solve_rk4
from fisher_kpp_rk4.config import (
    D,
    L,
    Nt,
    Nx,
    T,
    dt,
    dx,
    initial_condition,
    left_bc,
    r,
    right_bc,
    x,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    info = check_rk4_stability(dx=dx, dt=dt, D=D, r=r, dim=1)
    print("=== Fisher-KPP RK4 setup ===")
    print(f"D={D}, r={r}, L={L}, T={T}")
    print(f"Nx={Nx}, dx={dx:.6g}, Nt={Nt}, dt={dt:.6g}")
    print(f"Diffusion RK4 limit: {info['dt_diff_limit']:.6g}")
    print(f"Practical dt safe? {info['is_practically_safe']}")

    result = solve_rk4(
        x=x,
        dt=dt,
        Nt=Nt,
        D=D,
        r=r,
        initial_condition=initial_condition,
        left_bc=left_bc,
        right_bc=right_bc,
        save_interval=5.0,
    )

    c_min = 2.0 * np.sqrt(D * r)
    c_num = estimate_front_speed(result["times"], result["fronts"], t_min=5.0, x_max=0.85 * L)
    print(f"Theoretical minimal KPP speed c* = {c_min:.6g}")
    print(f"Estimated front speed before boundary interaction: {c_num:.6g}")

    np.savez(
        OUTPUT_DIR / "fisher_kpp_rk4_results.npz",
        x=result["x"],
        times=result["times"],
        snapshots=result["snapshots"],
        fronts=result["fronts"],
        u_final=result["u_final"],
        D=D,
        r=r,
        L=L,
        T=T,
        dx=dx,
        dt=dt,
    )
    print("Saved: outputs/fisher_kpp_rk4_results.npz")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        for t, u in zip(result["times"], result["snapshots"]):
            plt.plot(result["x"], u, label=f"t={t:.0f}")
        plt.xlabel("x")
        plt.ylabel("u(x,t)")
        plt.ylim(-0.05, 1.05)
        plt.title("Fisher-KPP solved by MOL-FDM + RK4")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "snapshots.png", dpi=200)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.plot(result["times"], result["fronts"], marker="o")
        plt.xlabel("t")
        plt.ylabel("front position, u=0.5")
        plt.title("Front propagation")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "front_position.png", dpi=200)
        plt.close()

        print("Saved: outputs/snapshots.png")
        print("Saved: outputs/front_position.png")
    except ImportError:
        print("matplotlib not installed; skipped plots.")


if __name__ == "__main__":
    main()
