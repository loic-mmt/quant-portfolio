from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import sqlite3
import time
import matplotlib.pyplot as plt


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
    schema_names = set(dataset.schema.names)
    cols = ["date", "ticker", "weight"]
    if "run_id" in schema_names:
        cols.append("run_id")
    if run_id is not None:
        if "run_id" not in schema_names:
            raise KeyError("weights dataset has no run_id column.")
        filt = ds.field("run_id") == str(run_id)
        table = dataset.to_table(filter=filt, columns=cols)
    else:
        table = dataset.to_table(columns=cols)
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


def compute_baseline_hold(prices: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """
    Baseline hold: equal-weight buy-and-hold from the first available date.
    Returns a results DataFrame aligned with trading dates (same columns as simulate_portfolio).
    """
    if prices is None or prices.empty:
        raise ValueError("prices is empty")
    required = {"date", "ticker", "adj_close"}
    if not required.issubset(prices.columns):
        missing = ", ".join(sorted(required - set(prices.columns)))
        raise KeyError(f"Missing required columns: {missing}")

    data = prices.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date", "ticker", "adj_close"])
    price_matrix = (
        data.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last")
        .sort_index()
    )
    price_matrix = price_matrix.dropna(how="all")
    if price_matrix.empty:
        raise ValueError("price_matrix is empty after cleaning.")

    dates = pd.DatetimeIndex(price_matrix.index)
    first_prices = price_matrix.iloc[0]
    live = first_prices.notna()
    if live.sum() == 0:
        raise ValueError("No valid prices on the first date for baseline.")

    weights = np.zeros(len(first_prices), dtype=float)
    weights[live.values] = 1.0 / float(live.sum())

    # Normalize price relatives to 1 at t0.
    rel = price_matrix.div(first_prices, axis=1)
    port_value = (rel * weights).sum(axis=1) * float(cfg.initial_capital)
    port_return = port_value.pct_change().fillna(0.0)

    results = pd.DataFrame(
        {
            "portfolio_value": port_value.values,
            "portfolio_return": port_return.values,
            "turnover": np.zeros(len(port_value), dtype=float),
            "cost": np.zeros(len(port_value), dtype=float),
            "is_rebalance": np.zeros(len(port_value), dtype=bool),
        },
        index=dates,
    )
    results["pnl"] = results["portfolio_value"] - cfg.initial_capital
    return results


def plot_backtest_vs_baseline(
    results: pd.DataFrame,
    baseline: pd.DataFrame,
    title: str = "Backtest vs Baseline",
) -> None:
    if results is None or results.empty:
        raise ValueError("results is empty")
    if baseline is None or baseline.empty:
        raise ValueError("baseline is empty")

    for name, df in {"results": results, "baseline": baseline}.items():
        if "portfolio_value" not in df.columns:
            raise KeyError(f"{name} missing 'portfolio_value' column")

    plt.figure(figsize=(10, 5))
    plt.plot(results.index, results["portfolio_value"], label="Strategy", linewidth=2)
    plt.plot(baseline.index, baseline["portfolio_value"], label="Baseline", linewidth=2, linestyle="--")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


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
          date_debut     TEXT NOT NULL,
          date_fin       TEXT NOT NULL,  -- YYYY-MM-DD
          CAGR                  REAL,
          volatility            REAL,
          Sharpe                REAL,
          max_drawdown          REAL,
          turnover_annualised   REAL,
          turnover_mean         REAL,
          turnover_vol          REAL,
          run_id                TEXT PRIMARY KEY
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_backtests_run_id ON backtests(run_id);")
    conn.commit()


def upsert_summary(conn: sqlite3.Connection, summary: dict[str, float], run_id: str, date_start: str, date_end: str) -> None:
    if not summary:
        return
    sql = """
    INSERT INTO backtests (
      date_debut, date_fin, CAGR, volatility, Sharpe, max_drawdown,
      turnover_annualised, turnover_mean, turnover_vol, run_id
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(run_id) DO UPDATE SET
      date_debut = excluded.date_debut,
      date_fin = excluded.date_fin,
      CAGR = excluded.CAGR,
      volatility = excluded.volatility,
      Sharpe = excluded.Sharpe,
      max_drawdown = excluded.max_drawdown,
      turnover_annualised = excluded.turnover_annualised,
      turnover_mean = excluded.turnover_mean,
      turnover_vol = excluded.turnover_vol;
    """
    row = (
        date_start,
        date_end,
        float(summary.get("CAGR", np.nan)),
        float(summary.get("volatility", np.nan)),
        float(summary.get("Sharpe", np.nan)),
        float(summary.get("max_drawdown", np.nan)),
        float(summary.get("turnover_annualised", np.nan)),
        float(summary.get("turnover_mean", np.nan)),
        float(summary.get("turnover_vol", np.nan)),
        str(run_id),
    )
    conn.execute(sql, row)
    conn.commit()

def write_backtest_outputs(
    results: pd.DataFrame,
    summary: dict[str, float],
    partition_cols: list[str],
    base_dir: Path,
    existing_data_behavior: str = "overwrite_or_ignore",
    run_id: str | None = None,
) -> None:
    if results is None or results.empty:
        raise ValueError("results empty")
    if not summary:
        raise ValueError("summary empty")
    if run_id is None:
        run_id = str(int(time.time()))

    df = results.copy()
    df["date"] = pd.to_datetime(df.index, errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    df["year"] = df["date"].dt.year.astype("int32")
    df["run_id"] = str(run_id)

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
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    dates = df["date"].sort_values()
    date_start = dates.iloc[0].strftime("%Y-%m-%d")
    date_end = dates.iloc[-1].strftime("%Y-%m-%d")
    upsert_summary(conn, summary, run_id, date_start, date_end)
    conn.close()
    return None




def run_backtest_pipeline(run_id: str | None = None) -> None:
    cfg = load_backtest_config()
    prices = load_prices_dataset()
    weights = load_target_weights(run_id)
    returns = build_returns_matrix(prices)
    results = simulate_portfolio(returns, weights, cfg)
    baseline = compute_baseline_hold(prices, cfg)
    summary = summarize_performance(results)
    write_backtest_outputs(
        results,
        summary,
        partition_cols=["year", "run_id"],
        base_dir=BACKTEST_DIR,
        run_id=run_id,
    )
    # Baseline is returned for plotting comparisons (not stored).
    return results, baseline



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run backtest pipeline.")
    parser.add_argument("--run-id", default=None, help="Optional run identifier to select weights.")
    args = parser.parse_args()
    run_backtest_pipeline(run_id=args.run_id)
