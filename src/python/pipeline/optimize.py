from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import time
import yaml


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


DEFAULT_CFG = OptimizeConfig(
    rebal_freq="W",
    max_weight=0.3,
    allow_cash=False,
    min_weight=0.01,
    lookback=10
)


def load_optimize_config() -> OptimizeConfig:
    # TODO: read config/optimize.yaml (or set defaults if missing)
    # TODO: validate numeric ranges (0 <= min_weight <= max_weight <= 1)
    # TODO: return OptimizeConfig
    if not CONFIG_PATH.exists():
        return DEFAULT_CFG
    content = CONFIG_PATH.read_text().strip()
    if not content:
        return DEFAULT_CFG
    if yaml is None:
        raise ImportError("PyYAML is required to parse config/backtest.yaml.")
    
    data = yaml.safe_load(content)

    if not isinstance(data, dict):
        raise ValueError("optimize.yaml must contain a YAML mapping (dict).")

    allowed = {f.name for f in fields(OptimizeConfig)}
    unknown = set(data.keys()) - allowed
    if unknown:
        raise ValueError(f"Unknown fields in backtest.yaml: {sorted(unknown)}")
    
    merged = {**DEFAULT_CFG.__dict__,**data}
    cfg = OptimizeConfig(**merged)

    if 0 >= cfg.max_weight >= 1 or cfg.max_weight < cfg.min_weight:
        raise ValueError("max_weight must be between 0 and 1 and over min_weight")
    if 0 >= cfg.min_weight >= 1 or cfg.max_weight > cfg.min_weight:
        raise ValueError("min_weight must be between 0 and 1 and under max_weight")
    return cfg

def load_prices_dataset(tickers: list[str] | None = None) -> pd.DataFrame:
    dataset = ds.dataset(str(PRICES_DIR), format="parquet", partitioning="hive")

    if tickers:
        tickers = [t.upper().strip() for t in tickers]
        filt = ds.field("ticker").isin(tickers)
        table = dataset.to_table(filter = filt, columns=["date", "ticker", "adj_close"])
    else:
        table = dataset.to_table(columns = ["date", "ticker", "adj_close"])
    
    df = table.to_pandas()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna(subset=["date", "ticker", "adj_close"]).sort_values(["date","ticker"])
    return df


def build_returns_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Prices data are empty.")
    required = {"date", "ticker", "adj_close"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise KeyError(f"Missing required columns: {missing}")
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"])

    prices = (data.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last").sort_index())
    returns = np.log(prices / prices.shift(1))
    returns = returns.dropna(how = "all")
    return returns


def build_rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """
    Build rebalance dates aligned to available trading dates.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Trading dates (must be sortable, usually business days).
    freq : str
        "D", "W", "2W", "M"

    Returns
    -------
    pd.DatetimeIndex
        Rebalance dates (subset of index).
    """
    if index is None or len(index) == 0:
        return pd.DatetimeIndex([])
    freq = freq.upper().strip()
    if freq not in {"D", "W", "2W", "M"}:
        raise ValueError("Frequency must be one of: D, W, 2W, M")
    
    # sorted and unique
    idx = pd.DatetimeIndex(pd.to_datetime(index)).sort_values().unique()

    # Daily
    if freq == "D":
        return idx
    
    # Map to pandas resample frequency (anchored to Friday for weekly periods)
    resample_map = {
        "W": "W-FRI",
        "2W": "2W-FRI",
        "M": "M",
    }

    s = pd.Series(idx, index=idx) # dummy series
    rebal = s.resample(resample_map[freq]).last().dropna().values

    rebal_idx = pd.DatetimeIndex(rebal)

    if rebal_idx.empty or rebal_idx[0] != idx[0]:
        rebal_idx = rebal_idx.insert(0, idx[0])
    
    rebal_idx = rebal_idx.unique()
    return rebal_idx


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
