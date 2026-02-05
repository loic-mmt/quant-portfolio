from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


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
    raise NotImplementedError


def ledoit_wolf_covariance(returns: pd.DataFrame) -> np.ndarray:
    # TODO: implement Ledoit-Wolf (or use sklearn if allowed)
    # TODO: return covariance as numpy array
    raise NotImplementedError


def ensure_positive_definite(cov: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # TODO: add eps to diagonal if needed
    # TODO: optionally clip negative eigenvalues
    # TODO: return adjusted covariance
    raise NotImplementedError


def compute_covariance(
    returns: pd.DataFrame,
    cfg: CovarianceConfig,
) -> np.ndarray:
    # TODO: clean returns
    # TODO: select method (sample/shrink_diag/ledoit_wolf)
    # TODO: apply ensure_positive_definite
    # TODO: return covariance
    raise NotImplementedError
