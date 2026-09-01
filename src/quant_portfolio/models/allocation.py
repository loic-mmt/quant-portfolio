"""Convex allocation with explicit, numerically checked hard constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import cvxpy as cp
import numpy as np

from quant_portfolio.models.covariance import ensure_positive_definite


class OptimizationError(RuntimeError):
    """No validated feasible solution was produced."""


@dataclass(frozen=True)
class AllocationResult:
    weights: np.ndarray
    status: str
    solver: str
    objective: float
    max_constraint_violation: float


def constraint_violations(
    weights: np.ndarray,
    previous: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
    *,
    allow_cash: bool,
    turnover_cap: float | None = None,
    exposure_change_cap: float | None = None,
    sectors: list[str | None] | None = None,
    sector_caps: Mapping[str, float] | None = None,
    fixed_exposure: float | None = None,
    composition: np.ndarray | None = None,
    min_exposure: float = 0.0,
    max_exposure: float = 1.0,
) -> dict[str, float]:
    w = np.asarray(weights)
    if not np.isfinite(w).all():
        return {"finite": float("inf")}
    exposure = float(w.sum())
    violation = {
        "minimum": float(np.maximum(minimum - w, 0).max()),
        "maximum": float(np.maximum(w - maximum, 0).max()),
        "exposure": max(0.0, exposure - max_exposure, min_exposure - exposure)
        if allow_cash
        else abs(exposure - 1.0),
    }
    if turnover_cap is not None:
        turnover = 0.5 * (np.abs(w - previous).sum() + abs(exposure - previous.sum()))
        violation["turnover"] = max(0.0, float(turnover - turnover_cap))
    if exposure_change_cap is not None:
        violation["speed"] = max(0.0, float(abs(exposure - previous.sum()) - exposure_change_cap))
    if fixed_exposure is not None:
        violation["fixed_exposure"] = abs(exposure - fixed_exposure)
    if composition is not None:
        violation["composition"] = float(np.abs(w - composition * exposure).max())
    for sector, cap in (sector_caps or {}).items():
        violation[f"sector:{sector}"] = max(0.0, float(w[np.array(sectors) == sector].sum() - cap))
    return violation


def solve_allocation(
    covariance: np.ndarray,
    previous: np.ndarray,
    *,
    minimum: np.ndarray,
    maximum: np.ndarray,
    desired_exposure: float = 1.0,
    allow_cash: bool = True,
    fixed_exposure: float | None = None,
    turnover_penalty: float = 0.002,
    transaction_cost_rate: float = 0.0003,
    distance_penalty: float = 0.01,
    exposure_penalty: float = 100.0,
    target_vol: float = 0.15,
    target_vol_penalty: float = 1.0,
    turnover_cap: float | None = None,
    exposure_change_cap: float | None = None,
    sectors: list[str | None] | None = None,
    sector_caps: Mapping[str, float] | None = None,
    composition: np.ndarray | None = None,
    solver: str = "CLARABEL",
    fallback: str = "error",
    min_exposure: float = 0.0,
    max_exposure: float = 1.0,
    tolerance: float = 1e-7,
) -> AllocationResult:
    """Minimize annual variance + trading costs + turnover/distance/vol penalties.

    Exposure is a soft tracking objective when cash is allowed, to keep speed and
    turnover limits hard even during a stress cut. No post-solve normalization.
    """
    previous = np.asarray(previous, dtype=float)
    minimum, maximum = np.asarray(minimum, dtype=float), np.asarray(maximum, dtype=float)
    n = len(previous)
    if n == 0 or any(x.shape != (n,) for x in (minimum, maximum)):
        raise ValueError("Invalid allocation dimensions")
    if not np.isfinite(np.r_[previous, minimum, maximum]).all():
        raise ValueError("Allocation inputs must be finite")
    if (minimum < 0).any() or (maximum < minimum).any() or (maximum > 1).any():
        raise ValueError("Allocation bounds must satisfy 0 <= minimum <= maximum <= 1")
    if covariance.shape != (n, n) or (previous < 0).any() or previous.sum() > 1 + tolerance:
        raise ValueError("Invalid covariance or previous weights")
    if sector_caps and (sectors is None or len(sectors) != n or any(s is None for s in sectors)):
        raise ValueError("Sector caps require metadata for every asset")
    if fallback not in {"error", "hold"}:
        raise ValueError("fallback must be error or hold")
    cov = ensure_positive_definite(covariance, eps=1e-12)
    root = np.linalg.cholesky(cov)
    w = cp.Variable(n)
    exposure = cp.sum(w)
    turnover = 0.5 * (cp.norm1(w - previous) + cp.abs(exposure - previous.sum()))
    constraints = (
        [w >= minimum, w <= maximum, exposure <= max_exposure, exposure >= min_exposure]
        if allow_cash
        else [
            w >= minimum,
            w <= maximum,
            exposure == 1,
        ]
    )
    if fixed_exposure is not None:
        constraints.append(exposure == fixed_exposure)
    if turnover_cap is not None:
        constraints.append(turnover <= turnover_cap)
    if exposure_change_cap is not None:
        constraints.append(cp.abs(exposure - previous.sum()) <= exposure_change_cap)
    if composition is not None:
        constraints.append(w == composition * exposure)
    for sector, cap in (sector_caps or {}).items():
        constraints.append(cp.sum(w[np.array(sectors) == sector]) <= cap)
    objective = (
        252 * cp.quad_form(w, cp.psd_wrap(cov))
        + (turnover_penalty + transaction_cost_rate) * turnover
        + distance_penalty * cp.sum_squares(w - previous)
        + exposure_penalty * cp.square(exposure - desired_exposure)
        + target_vol_penalty * cp.square(cp.pos(np.sqrt(252) * cp.norm(root.T @ w) - target_vol))
    )
    problem = cp.Problem(cp.Minimize(objective), constraints)
    failure = ""
    try:
        options = (
            {"tol_gap_abs": 1e-10, "tol_feas": 1e-10, "tol_gap_rel": 1e-10} if solver == "CLARABEL" else {}
        )
        problem.solve(solver=solver, **options)
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or w.value is None:
            failure = f"solver status={problem.status}"
    except cp.error.SolverError as exc:
        failure = f"solver error: {exc}"

    def violations(candidate):
        return constraint_violations(
            candidate,
            previous,
            minimum,
            maximum,
            allow_cash=allow_cash,
            turnover_cap=turnover_cap,
            exposure_change_cap=exposure_change_cap,
            sectors=sectors,
            sector_caps=sector_caps,
            fixed_exposure=fixed_exposure,
            composition=composition,
            min_exposure=min_exposure,
            max_exposure=max_exposure,
        )

    if not failure:
        candidate = np.asarray(w.value).ravel()
        # Remove numerical dust only, then revalidate every coupled constraint.
        candidate[np.abs(candidate) < 1e-12] = 0.0
        max_violation = max(violations(candidate).values())
        if max_violation <= tolerance and (candidate >= 0).all() and candidate.sum() <= 1 + 1e-10:
            return AllocationResult(
                candidate, str(problem.status), solver, float(problem.value), max_violation
            )
        failure = f"post-solve constraint violation={max_violation:.3g}"
    if fallback == "hold" and max(violations(previous).values()) <= tolerance:
        return AllocationResult(previous.copy(), f"fallback_hold: {failure}", solver, float("nan"), 0.0)
    raise OptimizationError(f"No feasible validated allocation: {failure}")
