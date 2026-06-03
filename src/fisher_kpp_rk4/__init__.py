"""Utilities for solving the 1D Fisher-KPP equation with RK4."""

from .solver import (
    apply_dirichlet_bc,
    check_rk4_stability,
    estimate_front_speed,
    fisher_kpp_rhs,
    front_position,
    relative_l2,
    rk4_step,
    solve_rk4,
)

__all__ = [
    "apply_dirichlet_bc",
    "check_rk4_stability",
    "estimate_front_speed",
    "fisher_kpp_rhs",
    "front_position",
    "relative_l2",
    "rk4_step",
    "solve_rk4",
]
