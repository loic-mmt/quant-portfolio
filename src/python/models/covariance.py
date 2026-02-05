from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from sklearn.covariance import LedoitWolf


@dataclass
class CovarianceConfig:
    """Configuration for covariance estimation and conditioning.

    Attributes
    ----------
    method : Literal["sample", "shrink_diag", "ledoit_wolf"]
        Estimation method for the raw covariance.
    shrinkage : float
        Shrinkage intensity used when method="shrink_diag". Must be in [0, 1].
    min_periods : int
        Minimum observations required per asset when computing sample covariances.
    eps : float
        Minimum eigenvalue enforced in the positive-definite adjustment.
    """
    method: Literal["sample", "shrink_diag", "ledoit_wolf"]
    shrinkage: float
    min_periods: int
    eps: float


def clean_returns(returns: pd.DataFrame, min_obs: int = 2) -> pd.DataFrame:
    """Clean a returns matrix by removing empty rows/columns.

    Parameters
    ----------
    returns : pd.DataFrame
        Raw returns with dates as index and assets as columns.
    min_obs : int, default 2
        Minimum non-null observations required per asset column.

    Returns
    -------
    pd.DataFrame
        Filtered returns with only sufficiently observed assets.

    Raises
    ------
    ValueError
        If the input is empty or no columns satisfy `min_obs`.
    """
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
    """Compute a sample covariance matrix from returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Returns matrix.
    min_periods : int, default 2
        Minimum observations per pair for covariance computation.

    Returns
    -------
    np.ndarray
        Sample covariance matrix (n_assets x n_assets).

    Raises
    ------
    ValueError
        If the returns are empty or covariance cannot be computed.
    """
    if returns is None or returns.empty:
        raise ValueError("Returns are empty.")
    _returns = clean_returns(returns, min_periods)
    cov_returns = _returns.cov(min_periods=min_periods)
    if cov_returns is None or cov_returns.empty:
        raise ValueError("Covariance on returns is empty.")
    cov_returns = cov_returns.to_numpy()
    return cov_returns




def shrink_to_diagonal(cov: np.ndarray, shrinkage: float) -> np.ndarray:
    """Shrink a covariance matrix toward its diagonal target.

    Parameters
    ----------
    cov : np.ndarray
        Input covariance matrix (square).
    shrinkage : float
        Shrinkage intensity in [0, 1]. 0 keeps cov, 1 uses only the diagonal.

    Returns
    -------
    np.ndarray
        Shrunk covariance matrix.

    Raises
    ------
    ValueError
        If input is empty or shrinkage is outside [0, 1].
    """
    if cov is None or cov.size == 0:
        raise ValueError("Covariance is empty.")
    if not (0.0 <= shrinkage <= 1.0):
        raise ValueError("shrinkage must be in [0, 1].")
    diag_target = np.diag(np.diag(cov))
    return (1.0 - shrinkage) * cov + shrinkage * diag_target



def ledoit_wolf_covariance(returns: pd.DataFrame) -> np.ndarray:
    """Compute Ledoit-Wolf shrinkage covariance from returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Returns matrix.

    Returns
    -------
    np.ndarray
        Ledoit-Wolf covariance matrix.
    """
    if returns is None or returns.empty:
        raise ValueError("Returns are empty.")
    _returns = clean_returns(returns)
    cov = LedoitWolf().fit(_returns.to_numpy())
    return cov.covariance_


def ensure_positive_definite(cov: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Clip eigenvalues to enforce positive definiteness.

    Parameters
    ----------
    cov : np.ndarray
        Symmetric covariance matrix.
    eps : float, default 1e-6
        Minimum eigenvalue to enforce.

    Returns
    -------
    np.ndarray
        Adjusted covariance matrix with eigenvalues >= eps.
    """
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
    """Compute covariance with the chosen estimator and conditioning.

    This applies:
    1) cleaning of the returns matrix,
    2) the estimator selected in ``cfg.method``,
    3) a final positive-definite adjustment.

    Parameters
    ----------
    returns : pd.DataFrame
        Returns matrix.
    cfg : CovarianceConfig
        Estimation and conditioning configuration.

    Returns
    -------
    np.ndarray
        Positive-definite covariance matrix.

    Raises
    ------
    ValueError
        If the method is unknown or input data are empty.
    """
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
