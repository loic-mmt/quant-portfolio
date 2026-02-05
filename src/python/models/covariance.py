from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from sklearn.covariance import LedoitWolf


@dataclass
class CovarianceConfig:
    """Configuration for covariance estimation and conditioning."""
    method: Literal["sample", "shrink_diag", "ledoit_wolf"]
    shrinkage: float
    min_periods: int
    eps: float


def clean_returns(returns: pd.DataFrame, min_obs: int = 2) -> pd.DataFrame:
    """Drop empty rows/columns and keep assets with enough observations."""
    if returns is None or returns.empty:
        raise ValueError("Returns are empty.")
    data = returns.copy()
    data = data.dropna(how="all")
    if data.empty:
        raise ValueError("Returns are empty after cleaning.")
    valid_cols = data.columns[data.count() >= min_obs]
    data = data[valid_cols]
    if data.empty:
        raise ValueError("No columns with sufficient observations.")
    return data


def sample_covariance(returns: pd.DataFrame, min_periods: int = 2) -> np.ndarray:
    """Compute sample covariance matrix from returns."""
    if returns is None or returns.empty:
        raise ValueError("Returns are empty.")
    _returns = clean_returns(returns, min_periods)
    cov_returns = _returns.cov(min_periods=min_periods)
    if cov_returns is None or cov_returns.empty:
        raise ValueError("Covariance on returns is empty.")
    cov_returns = cov_returns.to_numpy()
    return cov_returns




def shrink_to_diagonal(cov: np.ndarray, shrinkage: float) -> np.ndarray:
    """Shrink covariance toward its diagonal target."""
    if cov is None or cov.size == 0:
        raise ValueError("Covariance is empty.")
    if not (0.0 <= shrinkage <= 1.0):
        raise ValueError("shrinkage must be in [0, 1].")
    diag_target = np.diag(np.diag(cov))
    return (1.0 - shrinkage) * cov + shrinkage * diag_target



def ledoit_wolf_covariance(returns: pd.DataFrame) -> np.ndarray:
    """Compute Ledoit-Wolf shrinkage covariance from returns."""
    if returns is None or returns.empty:
        raise ValueError("Returns are empty.")
    _returns = clean_returns(returns)
    cov = LedoitWolf().fit(_returns.to_numpy())
    return cov.covariance_


def ensure_positive_definite(cov: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Clip eigenvalues to enforce positive definiteness."""
    if cov is None or cov.size == 0:
        raise ValueError("Covariance is empty.")
    cov = np.asarray(cov, dtype=float)
    cov = (cov + cov.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(cov)  # Symmetric eigendecomposition: cov = V diag(λ) Vᵀ
    # V = eigvecs (matrice des vecteurs propres)
    # λ = eigvals (valeurs propres)
    eigvals_clipped = np.clip(eigvals, eps, None)  # Enforce minimum eigenvalue for positive definiteness ≥ eps
    # @ = produit matriciel / équivalent de np.dot
    return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T  # Reconstruct adjusted covariance matrix
    


def compute_covariance(
    returns: pd.DataFrame,
    cfg: CovarianceConfig,
) -> np.ndarray:
    """Compute covariance matrix with the chosen estimator and conditioning."""
    if returns is None or returns.empty:
        raise ValueError("Returns are empty.")
    _returns = clean_returns(returns, cfg.min_periods)
    method = cfg.method

    if method == "sample":
        cov = sample_covariance(_returns, cfg.min_periods)
    elif method == "shrink_diag":
        cov = sample_covariance(_returns, cfg.min_periods)
        cov = shrink_to_diagonal(cov, cfg.shrinkage)
    elif method == "ledoit_wolf":
        cov = ledoit_wolf_covariance(_returns)
    else:
        raise ValueError(f"Unknown covariance method: {method}")

    return ensure_positive_definite(cov, cfg.eps)
