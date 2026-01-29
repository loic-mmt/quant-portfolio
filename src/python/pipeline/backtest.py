from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import sqlite3
import time


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
PRICES_DIR = DATA_DIR / "parquet/prices"
WEIGHTS_DIR = DATA_DIR / "parquet/weights"
BACKTEST_DIR = DATA_DIR / "parquet/backtests"
DB_PATH = DATA_DIR / "_meta.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = ROOT / "config/backtest.yaml"

try:
    import yaml
except Exception:
    yaml = None

@dataclass
class BacktestConfig:
    rebal_freq: str
    transaction_bps: float
    slippage_bps: float
    turnover_cap: float | None
    initial_capital: float

DEFAULT_CFG = BacktestConfig(
    rebal_freq="W",
    transaction_bps= 1.0,
    slippage_bps=2.0,
    turnover_cap=None,
    initial_capital=100_000.0
)


def load_backtest_config() -> BacktestConfig:
    if not CONFIG_PATH.exists():
        return DEFAULT_CFG
    
    content = CONFIG_PATH.read_text().strip()
    if not content:
        return DEFAULT_CFG
    
    if yaml is None:
        raise ImportError("PyYAML is required to parse config/backtest.yaml.")
    
    data = yaml.safe_load(content)

    # YAML doit être un dict
    if not isinstance(data, dict):
        raise ValueError("backtest.yaml must contain a YAML mapping (dict).")
    
    # champs autorisés
    allowed = {f.name for f in fields(BacktestConfig)}
    unknown = set(data.keys()) - allowed
    if unknown:
        raise ValueError(f"Unknown fields in backtest.yaml: {sorted(unknown)}")
    
    merged = {**DEFAULT_CFG.__dict__, **data}
    cfg = BacktestConfig(**merged)

    if cfg.initial_capital <= 0:
        raise ValueError("initial_capital must be > 0")
    if cfg.transaction_bps < 0 or cfg.slippage_bps < 0:
        raise ValueError("transaction_bps and slippage_bps must be >= 0")
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


def load_target_weights(run_id: str | None = None) -> pd.DataFrame:
    dataset = ds.dataset(str(WEIGHTS_DIR), format="parquet", partitioning="hive")
    if run_id:
        run_id = int(run_id)
        filt = ds.field(f"{run_id}").isin(run_id)
        table = dataset.to_table(filter = filt, columns=["date", "ticker", "weight"])
    else:
        table = dataset.to_table(columns = ["date", "ticker", "weight"])
    df = table.to_pandas()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "weight"]).sort_values(["date", "ticker"])
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



def align_weights_to_dates(
    target_weights: pd.DataFrame,
    rebal_dates: pd.DatetimeIndex,
    tickers: list[str],
) -> pd.DataFrame:
    if target_weights is None or target_weights.empty:
        raise ValueError("target_weights is empty")
    
    required = {"date", "ticker", "weight"}
    if not required.issubset(target_weights.columns):
        missing = ", ".join(sorted(required - set(target_weights.columns)))
        raise KeyError(f"Missing required columns: {missing}")
    
    tw = target_weights.copy()
    tw["date"] = pd.to_datetime(tw["date"], errors= "coerce")
    tw = tw.dropna(subset=["date", "ticker", "weight"])
    tw = tw.sort_values(["date", "ticker"])

    # pivot weights : date x ticker
    W = tw.pivot_table(index="date", columns="ticker", values="weight", aggfunc="last").sort_index()
    W = W.reindex(columns=tickers).fillna(0.0)
    W = W.reindex(pd.DatetimeIndex(rebal_dates).sort_values(), method="ffill").fillna(0.0)

    # normalisation
    total_exposure = W.sum(axis=1)
    for dt in W.index:
        if total_exposure.loc[dt] > 1.0 + 1e-12: # tolérance pour éviter 1.00000000002
            W.loc[dt] =  W.loc[dt] / total_exposure.loc[dt]

    return W


def compute_turnover(prev_w: np.ndarray, next_w: np.ndarray) -> float:
    prev_w = np.array(prev_w, dtype=float)
    next_w = np.array(next_w, dtype=float)

    if prev_w.shape != next_w.shape:
        raise ValueError(f"Shape mismatch: prev_w {prev_w.shape} vs next_w {next_w.shape}")
    turnover = 0.5 * np.sum(np.abs(next_w - prev_w))
    return float(turnover)


def apply_turnover_cap(
    prev_w: np.ndarray,
    next_w: np.ndarray,
    cap: float | None,
) -> np.ndarray:
    prev_w = np.asarray(prev_w, dtype=float)
    next_w = np.asarray(next_w, dtype=float)

    if prev_w.shape != next_w.shape:
        raise ValueError(f"Shape mismatch: prev_w {prev_w.shape} vs next_w {next_w.shape}")

    if cap is None:
        return next_w

    if cap < 0:
        raise ValueError("cap must be >= 0 or None")

    turnover = compute_turnover(prev_w, next_w)
    if turnover <= cap:
        return next_w

    alpha = cap / turnover  # alpha in (0,1)
    capped_w = prev_w + alpha * (next_w - prev_w)

    return capped_w



def estimate_transaction_costs(turnover: float, bps: float, slippage_bps: float) -> float:
    if turnover < 0:
        raise ValueError("turnover must be >= 0")
    if bps < 0:
        raise ValueError("bps must be >= 0")
    if slippage_bps < 0:
        raise ValueError("slippage_bps must be >= 0")

    fee = bps / 10_000
    slippage = slippage_bps / 10_000

    total_cost = turnover * (fee + slippage)
    return float(total_cost)


def simulate_portfolio(
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    cfg: BacktestConfig,
) -> pd.DataFrame:
    
    if returns is None or returns.empty:
         raise ValueError("returns is empty")
    tickers = list(returns.columns)
    dates = pd.DatetimeIndex(returns.index).sort_values()

    rebal_dates = build_rebalance_dates(dates, freq=cfg.rebal_freq)

    W_target = align_weights_to_dates(target_weights, rebal_dates, tickers)

    V = float(cfg.initial_capital)
    w = W_target.iloc[0].to_numpy(dtype=float) # poids initial
    w = w / w.sum() if w.sum() > 1.0 else w

    port_value = []
    port_ret = []
    turnover_list = []
    cost_list = []
    is_rebal = []

    rebal_set = set(pd.DatetimeIndex(rebal_dates))

    for dt in dates :
        turnover = 0
        cost = 0
        rebalanced = False

        if dt in rebal_set:
            target_w = W_target.loc[dt].to_numpy(dtype=float)
            turnover = compute_turnover(w, target_w) # turnover vs current weight
            capped_w = apply_turnover_cap(w, target_w, cfg.turnover_cap)
            turnover = compute_turnover(w, capped_w)
            cost = estimate_transaction_costs(turnover, cfg.transaction_bps, cfg.slippage_bps)

            V *= (1.0 - cost)

            w = capped_w
            rebalanced = True
        # daily portfolio return
        r_vec = returns.loc[dt].to_numpy(dtype=float)
        r_p = float(np.nansum(w * r_vec)) # sum of weighted returns

        # update value
        V *= (1.0 + r_p)

        port_value.append(V)
        port_ret.append(r_p)
        turnover_list.append(turnover)
        cost_list.append(cost)
        is_rebal.append(rebalanced)
    
    results = pd.DataFrame(
        {
            "portfolio_value": port_value,
            "portfolio_return": port_ret,
            "turnover": turnover_list,
            "cost": cost_list,
            "is_rebalance": is_rebal,
        },
        index=dates
    )
    results["pnl"] = results["portfolio_value"] - cfg.initial_capital
    return results


def summarize_performance(results: pd.DataFrame) -> dict[str, float]:
    if results is None or results.empty:
        raise ValueError("results is empty")
    
    required = {"portfolio_value", "portfolio_return", "turnover"}
    missing = required - set(results.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    
    trading_days = 252
    n = len(results)
    years = n / trading_days

    V0 = float(results["portfolio_value"].iloc[0])
    Vend = float(results["portfolio_value"].iloc[-1])

    # CAGR
    if V0 <= 0 or years <= 0:
        cagr = np.nan
    else:
        cagr = (Vend / V0) ** (1 / years) -1

    # Volatility (annualisée)
    daily_ret = results["portfolio_return"].astype(float)
    vol = float(daily_ret.std(ddof=1) * np.sqrt(trading_days))

    # Sharpe (rf annuel fixe)
    rf = 0.05
    mu_annual = float(daily_ret.mean() * trading_days)
    sharpe = np.nan
    if vol > 0:
        sharpe = (mu_annual - rf) / vol

    # Max Drawdown
    pv = results["portfolio_value"].astype(float)
    running_max = pv.cummax()
    drawdown = (pv / running_max) -1.0
    max_drawdown = float(drawdown.min())
    max_drawdown = abs(max_drawdown)

    # Turnover
    turnover = results["turnover"].astype(float)
    turnover_mean = float(turnover.mean())
    turnover_vol = float(turnover.std(ddof=1))

    turnover_annualised = float(turnover.sum() / years) if years > 0 else np.nan
    
    summary = {
        "CAGR": float(cagr),
        "volatility": vol,
        "Sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
        "turnover_annualised": turnover_annualised,
        "turnover_mean": turnover_mean,
        "turnover_vol": turnover_vol,
    }

    return summary


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS backtests (
          tickers     TEXT NOT NULL,
          date       TEXT NOT NULL,  -- YYYY-MM-DD
          CAGR                  REAL,
          volatility            REAL,
          Sharpe                REAL,
          max_drawdown          REAL,
          turnover_annualised   REAL,
          turnover_mean         REAL,
          turnover_vol          REAL,
          run_id                PRIMARY KEY
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS run_id ON backtests(run_id);")
    conn.commit()



def write_backtest_outputs(
    results: pd.DataFrame,
    summary: dict[str, float],
    partition_cols: list[str],
    base_dir: Path,
    existing_data_behavior: str = "overwrite_or_ignore",
    run_id: str = ...,
) -> None:
    # TODO: write results to BACKTEST_DIR as parquet (hive by year/run_id)
    # TODO: optionally persist summary to SQLite or JSON
    # TODO: ensure directories exist
    if results is None or results.empty:
        raise ValueError("results empty")
    if summary is None or summary.empty:
        raise ValueError("results empty")
    
    dates = pd.DatetimeIndex(results.index).sort_values()
    n_cols = ...
    run_id = ", ".join()

    if "year" not in results.columns and "date" in results.columns:
        dt = pd.to_datetime(results["date"], errors = "coerce")
        ok = dt.notna()
        df = results.loc[ok].copy()
        df["year"] = dt.loc[ok].dt.year.astype("int32")

    table = pa.Table.from_pandas(df, preserve_index = False)
    schema = []
    for col in partition_cols :
        if col not in df.columns:
            raise KeyError(f"Missing partition column: {col}")
        if col == "year":
            schema.append((col, pa.int32()))
        else:
            schema.append((col, pa.string()))
    partitioning = ds.partitioning(pa.schema(schema), flavor="hive")
    ds.write_dataset(
        table,
        base_dir=str(base_dir),
        format="parquet",
        partitioning=partitioning,
        existing_data_behavior=existing_data_behavior,
    )


def run_backtest_pipeline(run_id: str | None = None) -> None:
    # TODO: load config
    # TODO: load prices and weights
    # TODO: build returns matrix
    # TODO: simulate portfolio
    # TODO: summarize and write outputs
    # TODO: log key metrics
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: add argparse for run_id and config overrides
    run_backtest_pipeline()
