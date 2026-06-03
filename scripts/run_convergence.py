import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fisher_kpp_rk4 import relative_l2, solve_rk4

OUTPUT_DIR = PROJECT_ROOT / "outputs"

D = 1.0
r = 1.0
L = 200.0
T = 20.0
left_bc = 1.0
right_bc = 0.0


def initial_condition(x):
    return 1.0 / (1.0 + np.exp((x - 50.0) / 5.0))


def run_case(Nx, dt):
    x = np.linspace(0.0, L, Nx)
    Nt = int(round(T / dt))
    dt = T / Nt
    return solve_rk4(
        x=x,
        dt=dt,
        Nt=Nt,
        D=D,
        r=r,
        initial_condition=initial_condition,
        left_bc=left_bc,
        right_bc=right_bc,
        save_interval=T,
    )


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    ref = run_case(Nx=1601, dt=0.00125)
    x_ref = ref["x"]
    u_ref = ref["u_final"]

    cases = [(201, 0.01), (401, 0.01), (801, 0.005)]
    rows = []

    for Nx, dt in cases:
        sol = run_case(Nx=Nx, dt=dt)
        u_ref_interp = np.interp(sol["x"], x_ref, u_ref)
        err = relative_l2(sol["u_final"], u_ref_interp)
        rows.append({"Nx": Nx, "dx": L / (Nx - 1), "dt": dt, "relative_l2": err})
        print(f"Nx={Nx:4d}, dx={L/(Nx-1):.6g}, dt={dt:.6g}, relL2={err:.6e}")

    with open(OUTPUT_DIR / "convergence_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Nx", "dx", "dt", "relative_l2"])
        writer.writeheader()
        writer.writerows(rows)

    print("Saved: outputs/convergence_summary.csv")


if __name__ == "__main__":
    main()
