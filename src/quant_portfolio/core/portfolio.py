"""Shared close-to-close accounting for the strategy and its backtest."""

from __future__ import annotations

import numpy as np


def execution_weights(
    previous: np.ndarray,
    requested: np.ndarray,
    turnover_cap: float | None = None,
    exposure_change_cap: float | None = None,
) -> np.ndarray:
    """Govern a delayed order against holdings actually present at execution."""
    delta = np.asarray(requested) - np.asarray(previous)
    exposure_change = abs(float(delta.sum()))
    turnover = 0.5 * (float(np.abs(delta).sum()) + exposure_change)
    scale = 1.0
    if turnover_cap is not None and turnover > turnover_cap:
        scale = min(scale, turnover_cap / turnover)
    if exposure_change_cap is not None and exposure_change > exposure_change_cap:
        scale = min(scale, exposure_change_cap / exposure_change)
    return np.asarray(previous) + scale * delta


def drift_weights(
    weights: np.ndarray,
    returns: np.ndarray,
    cash_return: float = 0.0,
    missing_policy: str = "cash",
) -> tuple[np.ndarray, float, float]:
    """Apply simple returns to actual holdings; costs are paid pro rata upstream."""
    weights = np.asarray(weights, dtype=float)
    returns = np.asarray(returns, dtype=float)
    if weights.shape != returns.shape or not np.isfinite(weights).all():
        raise ValueError("Invalid portfolio weight/return dimensions or weights")
    if np.isinf(returns).any() or (returns[np.isfinite(returns)] < -1).any():
        raise ValueError("Simple returns must be finite or missing and >= -1")
    if missing_policy not in {"cash", "zero", "error"}:
        raise ValueError("Unknown missing-return policy")
    if missing_policy == "error" and (np.isnan(returns) & (weights > 1e-14)).any():
        raise ValueError("missing returns for held assets")
    replacement = cash_return if missing_policy == "cash" else 0.0
    simple = np.where(np.isnan(returns), replacement, returns)
    cash = max(0.0, 1.0 - float(weights.sum()))
    gross = float(weights @ simple + cash * cash_return)
    growth = 1.0 + gross
    if not np.isfinite(growth) or growth <= 0:
        raise ValueError("portfolio value became non-positive or non-finite")
    risky = weights * (1.0 + simple) / growth
    cash = cash * (1.0 + cash_return) / growth
    total = float(risky.sum() + cash)
    return risky / total, cash / total, gross
