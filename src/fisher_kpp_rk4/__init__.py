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
from .visualization import (
    circular_initial_condition,
    fisher_kpp_rhs_2d,
    plot_cross_section_view,
    rk4_step_2d,
    solve_2d_fisher_kpp_snapshots,
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
    "circular_initial_condition",
    "fisher_kpp_rhs_2d",
    "plot_cross_section_view",
    "rk4_step_2d",
    "solve_2d_fisher_kpp_snapshots",
]
