from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import time


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
PRICES_DIR = DATA_DIR / "parquet/prices"
WEIGHTS_DIR = DATA_DIR / "parquet/weights"
CONFIG_PATH = ROOT / "config/optimize.yaml"


@dataclass
class OptimizeConfig:
    rebal_freq: str
    max_weight: float
    allow_cash: bool
    min_weight: float
    lookback: int


def load_optimize_config() -> OptimizeConfig:
    # TODO: read config/optimize.yaml (or set defaults if missing)
    # TODO: validate numeric ranges (0 <= min_weight <= max_weight <= 1)
    # TODO: return OptimizeConfig
    raise NotImplementedError


def load_prices_dataset(tickers: list[str] | None = None) -> pd.DataFrame:
    # TODO: read parquet dataset from PRICES_DIR (hive)
    # TODO: optionally filter by tickers
    # TODO: parse date column to datetime
    # TODO: return tidy frame with date, ticker, adj_close
    raise NotImplementedError


def build_returns_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    # TODO: pivot prices to date x ticker
    # TODO: compute log returns
    # TODO: drop rows with all NaN
    # TODO: return returns matrix
    raise NotImplementedError


def build_rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    # TODO: implement same logic as backtest.build_rebalance_dates
    # TODO: align to available trading dates
    # TODO: return DatetimeIndex of rebal dates
    raise NotImplementedError


def min_variance_weights(cov: np.ndarray, max_weight: float, min_weight: float) -> np.ndarray:
    # TODO: solve a simple long-only min-variance (could use a heuristic)
    # TODO: enforce bounds [min_weight, max_weight]
    # TODO: normalize to sum to 1
    # TODO: return weight vector
    raise NotImplementedError


def compute_covariance(returns_window: pd.DataFrame) -> np.ndarray:
    # TODO: compute covariance matrix (simple sample cov for now)
    # TODO: handle missing values or drop columns with too many NaNs
    # TODO: return covariance as numpy array
    raise NotImplementedError


def optimize_over_time(
    returns: pd.DataFrame,
    rebal_dates: pd.DatetimeIndex,
    cfg: OptimizeConfig,
) -> pd.DataFrame:
    # TODO: for each rebalance date, take lookback window
    # TODO: compute covariance and solve min-variance weights
    # TODO: build a DataFrame of weights indexed by date and ticker
    # TODO: return tidy weights frame (date, ticker, weight)
    raise NotImplementedError


def write_weights_dataset(
    weights: pd.DataFrame,
    base_dir: Path,
    partition_cols: list[str],
    run_id: str | None = None,
    existing_data_behavior: str = "overwrite_or_ignore",
) -> None:
    # TODO: add run_id column if provided (or generate one)
    # TODO: add year partition
    # TODO: write to WEIGHTS_DIR as hive-partitioned parquet
    raise NotImplementedError


def run_optimize_pipeline(run_id: str | None = None) -> str:
    # TODO: load config
    # TODO: load prices and build returns
    # TODO: build rebal dates
    # TODO: optimize over time
    # TODO: write weights dataset and return run_id
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: add argparse for run_id and config overrides
    run_optimize_pipeline()
