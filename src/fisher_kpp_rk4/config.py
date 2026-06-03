import numpy as np

# PDE parameters
D = 1.0
r = 1.0

# Domain
L = 200.0
T = 60.0

# Grid
Nx = 401
dx = L / (Nx - 1)
x = np.linspace(0.0, L, Nx)

# Time
dt = 0.01
Nt = int(round(T / dt))
dt = T / Nt


def initial_condition(x):
    return 1.0 / (1.0 + np.exp((x - 50.0) / 5.0))


# Dirichlet boundary condition
left_bc = 1.0
right_bc = 0.0

# Kept for compatibility with implicit solvers; RK4 does not use them.
tol = 1e-10
max_iter = 30
