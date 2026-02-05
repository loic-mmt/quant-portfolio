from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from sklearn.covariance import LedoitWolf


@dataclass
class CovarianceConfig:
    method: Literal["sample", "shrink_diag", "ledoit_wolf"]
    shrinkage: float
    min_periods: int
    eps: float


def clean_returns(returns: pd.DataFrame, min_obs: int = 2) -> pd.DataFrame:
    # TODO: drop rows that are all NaN
    # TODO: keep only columns with at least min_obs non-NaN values
    # TODO: return cleaned returns
    if returns is None or returns.empty:
        raise ValueError("Returns are empty.")
    returns.dropna()
    if returns is None or returns.empty:
        raise ValueError("Returns are empty after cleaning.")
    for i in returns.columns:
        if len(returns.values[i]) < min_obs:
            returns.columns.drop()
    return returns


def sample_covariance(returns: pd.DataFrame, min_periods: int = 2) -> np.ndarray:
    # TODO: compute sample covariance on cleaned returns
    # TODO: handle empty or invalid input
    # TODO: return covariance as numpy array
    if returns is None or returns.empty:
        raise ValueError("Returns are empty.")
    _returns = clean_returns(returns, min_periods)
    cov_returns = _returns.cov()
    if cov_returns is None or cov_returns.empty:
        raise ValueError("Covariance on returns is empty.")
    cov_returns = cov_returns.to_numpy()
    return cov_returns




def shrink_to_diagonal(cov: np.ndarray, shrinkage: float) -> np.ndarray:
    # TODO: validate shrinkage in [0, 1]
    # TODO: build diagonal target from cov
    # TODO: return (1-shrinkage)*cov + shrinkage*diag
    if cov is None or cov.empty:
        raise ValueError("Covariance is empty.")
    if not 0 < shrinkage > 1: 
        raise ValueError("shrinkage must be in [0, 1].")
    diag_target = cov.diagonal()
    return (1 - shrinkage) * cov + shrinkage * diag_target



def ledoit_wolf_covariance(returns: pd.DataFrame) -> np.ndarray:
    # TODO: implement Ledoit-Wolf (or use sklearn if allowed)
    # TODO: return covariance as numpy array
    if returns is None or returns.empty:
        raise ValueError("Returns are empty.")
    _returns = clean_returns(returns)
    cov = LedoitWolf().fit(_returns)
    return cov.covariance_


def ensure_positive_definite(cov: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # TODO: add eps to diagonal if needed
    # TODO: optionally clip negative eigenvalues
    # TODO: return adjusted covariance
    diag = cov.diagonal()
    diag = diag + eps
    cliped_eigenvalues = diag.copy()
    for i in diag:
        if i < 0:
            cliped_eigenvalues[i] = 0
    return cliped_eigenvalues
    


def compute_covariance(
    returns: pd.DataFrame,
    cfg: CovarianceConfig,
) -> np.ndarray:
    # TODO: clean returns
    # TODO: select method (sample/shrink_diag/ledoit_wolf)
    # TODO: apply ensure_positive_definite
    # TODO: return covariance
    if returns is None or returns.empty:
        raise ValueError("Returns are empty.")
    _returns = clean_returns(returns, cfg.min_periods)
    if cfg.method is not None:
        method = cfg.method
    else :
        method = "ledoit_wolf"

    
    if method == "sample":
        cov = sample_covariance(_returns, cfg.min_periods)
        covariance = ensure_positive_definite(cov, cfg.eps)
    elif method == "shrink_diag":
        cov = sample_covariance(_returns, cfg.min_periods)
        cov = shrink_to_diagonal(cov, cfg.shrinkage)
        covariance = ensure_positive_definite(cov, cfg.eps)
    elif method == "ledoit_wolf":
        cov = ledoit_wolf_covariance(_returns)
        covariance = ensure_positive_definite(cov, cfg.eps)
    return covariance