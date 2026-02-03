from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import time
try:
    import yaml
except Exception:
    yaml = None


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
    if not CONFIG_PATH.exists():
        return DEFAULT_CFG
    content = CONFIG_PATH.read_text().strip()
    if not content:
        return DEFAULT_CFG
    if yaml is None:
        raise ImportError("PyYAML is required to parse config/optimize.yaml.")
    
    data = yaml.safe_load(content)

    if not isinstance(data, dict):
        raise ValueError("optimize.yaml must contain a YAML mapping (dict).")

    allowed = {f.name for f in fields(OptimizeConfig)}
    unknown = set(data.keys()) - allowed
    if unknown:
        raise ValueError(f"Unknown fields in optimize.yaml: {sorted(unknown)}")
    
    merged = {**DEFAULT_CFG.__dict__,**data}
    cfg = OptimizeConfig(**merged)

    if not (0 <= cfg.min_weight <= cfg.max_weight <= 1):
        raise ValueError("min_weight/max_weight must satisfy 0 <= min_weight <= max_weight <= 1")
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
    # Simple inverse-variance heuristic (long-only).
    if cov is None or cov.size == 0:
        raise ValueError("cov is empty")
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov must be a square 2D array")

    var = np.diag(cov)
    if np.any(var <= 0):
        raise ValueError("cov has non-positive variances")
    w = 1.0 / var
    w = np.clip(w, min_weight, max_weight)
    s = w.sum()
    if s <= 0:
        raise ValueError("weights sum to zero after clipping")
    return w / s

def compute_covariance(returns_window: pd.DataFrame) -> np.ndarray:
    if returns_window is None or returns_window.empty:
        raise ValueError("returns window empty")
    returns_window = returns_window.fillna(0.0)
    return returns_window.cov().to_numpy()



def optimize_over_time(
    returns: pd.DataFrame,
    rebal_dates: pd.DatetimeIndex,
    cfg: OptimizeConfig,
) -> pd.DataFrame:
    if returns is None or returns.empty:
        raise ValueError("returns is empty")
    if rebal_dates is None or len(rebal_dates) == 0:
        raise ValueError("rebal_dates is empty")

    window = int(cfg.lookback)
    if window <= 1:
        raise ValueError("lookback must be > 1")

    weights_rows = []
    returns = returns.sort_index()
    tickers = list(returns.columns)

    for dt in pd.DatetimeIndex(rebal_dates):
        hist = returns.loc[:dt].tail(window)
        if hist.empty:
            continue
        cov = compute_covariance(hist)
        w = min_variance_weights(cov, cfg.max_weight, cfg.min_weight)
        if w.shape[0] != len(tickers):
            raise ValueError("weights length mismatch with tickers")
        for t, wt in zip(tickers, w, strict=False):
            weights_rows.append({"date": dt, "ticker": t, "weight": float(wt)})

    if not weights_rows:
        raise ValueError("No weights computed (check lookback/rebal_dates).")
    return pd.DataFrame(weights_rows)



def write_weights_dataset(
    weights: pd.DataFrame,
    base_dir: Path,
    partition_cols: list[str],
    run_id: str | None = None,
    existing_data_behavior: str = "overwrite_or_ignore",
) -> None:
    if weights is None or weights.empty:
        raise ValueError("weights empty")
    base_dir.mkdir(parents=True, exist_ok=True)

    df = weights.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "weight"])
    if run_id is None:
        run_id = str(int(time.time()))
    df["run_id"] = str(run_id)
    df["year"] = df["date"].dt.year.astype("int32")

    table = pa.Table.from_pandas(df, preserve_index=False)
    schema = []
    for col in partition_cols:
        if col not in df.columns:
            raise KeyError(f"Missing partition column: {col}")
        if col == "year":
            schema.append((col, pa.int32()))
        else:
            schema.append((col, pa.string()))
    partitioning = ds.partitioning(pa.schema(schema), flavor="hive")
    if existing_data_behavior == "overwrite":
        existing_data_behavior = "delete_matching"
    ds.write_dataset(
        table,
        base_dir=str(base_dir),
        format="parquet",
        partitioning=partitioning,
        existing_data_behavior=existing_data_behavior,
    )
    return str(run_id)

def run_optimize_pipeline(
    run_id: str | None = None,
    existing_data_behavior: str = "overwrite_or_ignore",
) -> str:
    config = load_optimize_config()
    prices = load_prices_dataset()
    returns = build_returns_matrix(prices)
    rebal_dates = build_rebalance_dates(returns.index, freq=config.rebal_freq)
    weights = optimize_over_time(returns, rebal_dates, config)
    out_run_id = write_weights_dataset(
        weights,
        WEIGHTS_DIR,
        partition_cols=["year", "run_id"],
        run_id=run_id,
        existing_data_behavior=existing_data_behavior,
    )
    return out_run_id



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute minimum variance weights")
    parser.add_argument(
        "--existing-data-behavior",
        default="overwrite_or_ignore",
        choices=["overwrite_or_ignore", "overwrite", "error", "delete_matching"],
        help="Behavior when target dataset already has data.",
    )
    parser.add_argument(
        "--run_id",
        default=None,
    )
    args = parser.parse_args()
    run_optimize_pipeline(existing_data_behavior=args.existing_data_behavior, run_id=args.run_id)
