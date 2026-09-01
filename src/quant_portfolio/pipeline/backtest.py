from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
import logging
import json

import numpy as np
import pandas as pd
import sqlite3


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
PRICES_DIR = DATA_DIR / "parquet/prices"
WEIGHTS_DIR = DATA_DIR / "parquet/weights"
BACKTEST_DIR = DATA_DIR / "parquet/backtests"
TRADES_DIR = DATA_DIR / "parquet/trades"
DB_PATH = DATA_DIR / "_meta.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = ROOT / "config/backtest.yaml"

try:
    import yaml
except Exception:
    yaml = None

from quant_portfolio.core.db import init_backtest_db, upsert_backtest_summary
from quant_portfolio.core.ids import ensure_run_id
from quant_portfolio.core.settings import load_base_config
from quant_portfolio.core.portfolio import drift_weights, execution_weights
from quant_portfolio.pipeline.data_quality import load_reference_price_dataset


LOGGER = logging.getLogger(__name__)
BASE_CONFIG = load_base_config()

@dataclass
class BacktestConfig:
    """Configuration for the backtest pipeline.

    Attributes
    ----------
    rebal_freq : str
        Rebalance frequency ("D", "W", "2W", "M").
    transaction_bps : float
        Transaction fee in basis points, applied on turnover.
    slippage_bps : float
        Slippage cost in basis points, applied on turnover.
    turnover_cap : float | None
        Optional cap on turnover per rebalance (0..1). None disables.
    initial_capital : float
        Starting portfolio value.
    """
    rebal_freq: str
    transaction_bps: float
    slippage_bps: float
    turnover_cap: float | None
    initial_capital: float
    cash_rate_annual: float
    execution_lag_days: int
    return_type: str
    missing_return_policy: str
    start_date: str | None
    end_date: str | None
    exposure_change_cap: float | None = None

    def __post_init__(self) -> None:
        if type(self.execution_lag_days) is not int or self.execution_lag_days < 1:
            raise ValueError("execution_lag_days must be an integer >= 1")
        if not np.isfinite(self.cash_rate_annual) or self.cash_rate_annual <= -1:
            raise ValueError("cash_rate_annual must be finite and > -1")
        if self.return_type not in {"simple", "log"}:
            raise ValueError("return_type must be simple or log")
        if self.missing_return_policy not in {"cash", "zero", "error"}:
            raise ValueError("missing_return_policy must be cash, zero or error")
        for cap in (self.turnover_cap, self.exposure_change_cap):
            if cap is not None and not 0 <= cap <= 1:
                raise ValueError("Execution limits must be in [0, 1] or null")

DEFAULT_CFG = BacktestConfig(
    rebal_freq="W",
    transaction_bps= 1.0,
    slippage_bps=2.0,
    turnover_cap=None,
    initial_capital=100_000.0,
    cash_rate_annual=0.0,
    execution_lag_days=1,
    return_type="simple",
    missing_return_policy="cash",
    start_date=None,
    end_date=None,
)

def load_backtest_config() -> BacktestConfig:
    """Load backtest configuration from YAML or defaults.

    Returns
    -------
    BacktestConfig
        The validated configuration, using defaults when missing.

    Raises
    ------
    ImportError
        If PyYAML is not available and a config file exists.
    ValueError
        If the YAML structure or parameters are invalid.
    """
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
    if cfg.turnover_cap is not None and not 0 <= float(cfg.turnover_cap) <= 1:
        raise ValueError("turnover_cap must be between 0 and 1 or null")
    if int(cfg.execution_lag_days) < 1:
        raise ValueError("execution_lag_days must be >= 1 to prevent same-day look-ahead")
    if cfg.return_type not in {"simple", "log"}:
        raise ValueError("return_type must be 'simple' or 'log'")
    if cfg.missing_return_policy not in {"cash", "zero", "error"}:
        raise ValueError("missing_return_policy must be cash, zero or error")
    if cfg.start_date and cfg.end_date:
        if pd.Timestamp(cfg.start_date) > pd.Timestamp(cfg.end_date):
            raise ValueError("start_date must be <= end_date")
    return cfg


def build_returns_matrix(df: pd.DataFrame, return_type: str = "simple") -> pd.DataFrame:
    """Pivot prices into a returns matrix indexed by date.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form prices with columns date, ticker, adj_close.

    Returns
    -------
    pd.DataFrame
        Returns with dates as index and tickers as columns.
    """
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
    if return_type == "simple":
        returns = prices.pct_change(fill_method=None)
    elif return_type == "log":
        returns = np.log(prices / prices.shift(1))
    else:
        raise ValueError("return_type must be 'simple' or 'log'")
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

    Raises
    ------
    ValueError
        If `freq` is not one of the supported values.
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
    """Align target weights to rebalance dates and full ticker universe.

    Parameters
    ----------
    target_weights : pd.DataFrame
        Long-form weights with date/ticker/weight.
    rebal_dates : pd.DatetimeIndex
        Target rebalance dates.
    tickers : list[str]
        Full universe used to reindex weights.

    Returns
    -------
    pd.DataFrame
        Wide matrix (date x ticker) of aligned weights.
    """
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


def compute_turnover(
    prev_w: np.ndarray,
    next_w: np.ndarray,
    prev_cash: float | None = None,
    next_cash: float | None = None,
) -> float:
    """Compute one-way turnover between two weight vectors.

    Turnover is defined as 0.5 * sum(|w_t - w_{t-1}|).
    """
    prev_w = np.array(prev_w, dtype=float)
    next_w = np.array(next_w, dtype=float)

    if prev_w.shape != next_w.shape:
        raise ValueError(f"Shape mismatch: prev_w {prev_w.shape} vs next_w {next_w.shape}")
    absolute_change = float(np.sum(np.abs(next_w - prev_w)))
    if (prev_cash is None) != (next_cash is None):
        raise ValueError("prev_cash and next_cash must be provided together")
    if prev_cash is not None and next_cash is not None:
        absolute_change += abs(float(next_cash) - float(prev_cash))
    turnover = 0.5 * absolute_change
    return float(turnover)


def apply_turnover_cap(
    prev_w: np.ndarray,
    next_w: np.ndarray,
    cap: float | None,
    prev_cash: float | None = None,
    next_cash: float | None = None,
) -> np.ndarray:
    """Cap turnover by shrinking moves toward target weights.

    If turnover exceeds `cap`, weights are moved proportionally toward
    the target so that the resulting turnover equals `cap`.
    """
    prev_w = np.asarray(prev_w, dtype=float)
    next_w = np.asarray(next_w, dtype=float)

    if prev_w.shape != next_w.shape:
        raise ValueError(f"Shape mismatch: prev_w {prev_w.shape} vs next_w {next_w.shape}")

    if cap is None:
        return next_w

    if cap < 0:
        raise ValueError("cap must be >= 0 or None")

    turnover = compute_turnover(prev_w, next_w, prev_cash, next_cash)
    if turnover <= cap:
        return next_w

    alpha = cap / turnover  # alpha in (0,1)
    capped_w = prev_w + alpha * (next_w - prev_w)

    return capped_w


def build_target_weight_matrix(
    target_weights: pd.DataFrame,
    tickers: list[str],
) -> pd.DataFrame:
    """Validate long-only target weights and pivot them by decision date."""
    if target_weights is None or target_weights.empty:
        raise ValueError("target_weights is empty")
    required = {"date", "ticker", "weight"}
    missing = required - set(target_weights.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    frame = target_weights.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")
    frame = frame.dropna(subset=["date", "ticker", "weight"])
    if frame.empty:
        raise ValueError("target_weights has no valid observations")
    if (frame["weight"] < -1e-12).any():
        raise ValueError("negative target weights are not allowed")

    matrix = frame.pivot_table(
        index="date",
        columns="ticker",
        values="weight",
        aggfunc="last",
    ).sort_index()
    matrix = matrix.reindex(columns=tickers).fillna(0.0).clip(lower=0.0)
    exposure = matrix.sum(axis=1)
    if (exposure > 1.0 + 1e-10).any():
        bad_date = exposure[exposure > 1.0 + 1e-10].index[0]
        raise ValueError(f"target exposure exceeds 1.0 on {bad_date:%Y-%m-%d}")
    return matrix


def build_execution_schedule(
    decision_weights: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    lag_days: int,
) -> dict[pd.Timestamp, tuple[pd.Timestamp, np.ndarray]]:
    """Map each decision to a strictly later trading date.

    ``lag_days=1`` means the first available trading day after the decision.
    When multiple decisions map to the same execution date, the latest decision wins.
    """
    if lag_days < 1:
        raise ValueError("lag_days must be >= 1")
    dates = pd.DatetimeIndex(trading_dates).sort_values().unique()
    schedule: dict[pd.Timestamp, tuple[pd.Timestamp, np.ndarray]] = {}
    for decision_date, row in decision_weights.sort_index().iterrows():
        first_strictly_after = int(dates.searchsorted(pd.Timestamp(decision_date), side="right"))
        execution_position = first_strictly_after + lag_days - 1
        if execution_position >= len(dates):
            continue
        execution_date = pd.Timestamp(dates[execution_position])
        schedule[execution_date] = (
            pd.Timestamp(decision_date),
            row.to_numpy(dtype=float),
        )
    return schedule



def estimate_transaction_costs(turnover: float, bps: float, slippage_bps: float) -> float:
    """Estimate total cost from turnover, fees, and slippage.

    Parameters
    ----------
    turnover : float
        One-way turnover between current and target weights.
    bps : float
        Transaction fee in basis points.
    slippage_bps : float
        Slippage cost in basis points.

    Returns
    -------
    float
        Fractional cost deducted from portfolio value.
    """
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
    return_details: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate portfolio value and returns with periodic rebalancing.

    Parameters
    ----------
    returns : pd.DataFrame
        Returns matrix (date x ticker).
    target_weights : pd.DataFrame
        Long-form target weights with date/ticker/weight.
    cfg : BacktestConfig
        Backtest parameters.

    Returns
    -------
    pd.DataFrame
        Results indexed by date with value, returns, turnover, and costs.
    """
    
    if returns is None or returns.empty:
         raise ValueError("returns is empty")
    data = returns.copy().sort_index()
    data.index = pd.DatetimeIndex(pd.to_datetime(data.index))
    if cfg.start_date:
        data = data.loc[data.index >= pd.Timestamp(cfg.start_date)]
    if cfg.end_date:
        data = data.loc[data.index <= pd.Timestamp(cfg.end_date)]
    if data.empty:
        raise ValueError("returns is empty after applying the configured date range")

    tickers = [str(column) for column in data.columns]
    dates = pd.DatetimeIndex(data.index)
    decisions = build_target_weight_matrix(target_weights, tickers)
    schedule = build_execution_schedule(decisions, dates, int(cfg.execution_lag_days))

    value = float(cfg.initial_capital)
    risky_weights = np.zeros(len(tickers), dtype=float)
    cash_weight = 1.0
    daily_cash_return = (1.0 + float(cfg.cash_rate_annual)) ** (1.0 / 252.0) - 1.0

    result_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    position_rows: list[dict[str, object]] = []

    for date in dates:
        start_value = value
        turnover = 0.0
        cost_rate = 0.0
        cost_amount = 0.0
        decision_date: pd.Timestamp | None = None
        executed = False

        scheduled = schedule.get(pd.Timestamp(date))
        if scheduled is not None:
            decision_date, requested_weights = scheduled
            executed_weights = execution_weights(
                risky_weights,
                requested_weights,
                cfg.turnover_cap,
                cfg.exposure_change_cap,
            )
            executed_cash = max(0.0, 1.0 - float(executed_weights.sum()))
            turnover = compute_turnover(
                risky_weights,
                executed_weights,
                cash_weight,
                executed_cash,
            )
            cost_rate = estimate_transaction_costs(
                turnover,
                cfg.transaction_bps,
                cfg.slippage_bps,
            )
            if cost_rate >= 1.0:
                raise ValueError("transaction costs consume the entire portfolio")
            cost_amount = start_value * cost_rate
            value = start_value - cost_amount

            deltas = executed_weights - risky_weights
            for ticker, delta in zip(tickers, deltas, strict=False):
                if abs(float(delta)) <= 1e-14:
                    continue
                trade_rows.append(
                    {
                        "decision_date": decision_date,
                        "execution_date": pd.Timestamp(date),
                        "ticker": ticker,
                        "side": "BUY" if delta > 0 else "SELL",
                        "delta_weight": float(delta),
                        "notional": float(delta * value),
                    }
                )
            cash_delta = executed_cash - cash_weight
            if abs(cash_delta) > 1e-14:
                trade_rows.append(
                    {
                        "decision_date": decision_date,
                        "execution_date": pd.Timestamp(date),
                        "ticker": "__CASH__",
                        "side": "CASH_IN" if cash_delta > 0 else "CASH_OUT",
                        "delta_weight": float(cash_delta),
                        "notional": float(cash_delta * value),
                    }
                )
            risky_weights = executed_weights
            cash_weight = executed_cash
            executed = True

        raw_returns = data.loc[date].to_numpy(dtype=float)
        simple_returns = np.expm1(raw_returns) if cfg.return_type == "log" else raw_returns.copy()
        risky_weights, cash_weight, gross_return = drift_weights(
            risky_weights, simple_returns, daily_cash_return, cfg.missing_return_policy
        )
        growth = 1.0 + gross_return
        if growth <= 0:
            raise ValueError(f"portfolio value became non-positive on {date}")
        value *= growth
        net_return = value / start_value - 1.0

        result_rows.append(
            {
                "date": pd.Timestamp(date),
                "portfolio_value": float(value),
                "portfolio_return": float(net_return),
                "gross_return": gross_return,
                "turnover": turnover,
                "cost": cost_rate,
                "cost_amount": cost_amount,
                "is_rebalance": executed,
                "decision_date": decision_date,
                "cash_weight": float(cash_weight),
                "gross_exposure": float(risky_weights.sum()),
            }
        )
        for ticker, weight in zip(tickers, risky_weights, strict=False):
            position_rows.append(
                {
                    "date": pd.Timestamp(date),
                    "ticker": ticker,
                    "weight": float(weight),
                    "value": float(weight * value),
                    "is_cash": False,
                }
            )
        position_rows.append(
            {
                "date": pd.Timestamp(date),
                "ticker": "__CASH__",
                "weight": float(cash_weight),
                "value": float(cash_weight * value),
                "is_cash": True,
            }
        )

    results = pd.DataFrame(result_rows).set_index("date")
    results["pnl"] = results["portfolio_value"] - cfg.initial_capital
    results.attrs["initial_capital"] = float(cfg.initial_capital)
    results.attrs["execution_lag_days"] = int(cfg.execution_lag_days)
    trades = pd.DataFrame(trade_rows)
    positions = pd.DataFrame(position_rows)
    if return_details:
        return results, trades, positions
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
    if cfg.start_date:
        price_matrix = price_matrix.loc[price_matrix.index >= pd.Timestamp(cfg.start_date)]
    if cfg.end_date:
        price_matrix = price_matrix.loc[price_matrix.index <= pd.Timestamp(cfg.end_date)]
    if price_matrix.empty:
        raise ValueError("price_matrix is empty after applying the configured date range")
    # Choose baseline universe as tickers available on the first date.
    first_date = price_matrix.index.min()
    first_row = price_matrix.loc[first_date]
    live_cols = first_row[first_row.notna()].index.tolist()
    if not live_cols:
        raise ValueError("No tickers available on the first date for baseline.")
    price_matrix = price_matrix[live_cols]
    price_matrix = price_matrix.ffill()
    price_matrix = price_matrix.dropna()
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
    """Plot strategy equity curve against the baseline.

    Parameters
    ----------
    results : pd.DataFrame
        Backtest results from `simulate_portfolio`.
    baseline : pd.DataFrame
        Baseline results from `compute_baseline_hold`.
    title : str
        Figure title.
    """
    if results is None or results.empty:
        raise ValueError("results is empty")
    if baseline is None or baseline.empty:
        raise ValueError("baseline is empty")

    for name, df in {"results": results, "baseline": baseline}.items():
        if "portfolio_value" not in df.columns:
            raise KeyError(f"{name} missing 'portfolio_value' column")

    import matplotlib.pyplot as plt

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


def summarize_performance(
    results: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Compute summary performance statistics from backtest results.

    Returns CAGR, volatility, Sharpe ratio, max drawdown, and turnover stats.
    """
    if results is None or results.empty:
        raise ValueError("results is empty")
    
    required = {"portfolio_value", "portfolio_return", "turnover"}
    missing = required - set(results.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    
    trading_days = 252
    elapsed_days = max(1, int((results.index[-1] - results.index[0]).days) + 1)
    years = elapsed_days / 365.25

    V0 = float(results.attrs.get("initial_capital", results["portfolio_value"].iloc[0]))
    Vend = float(results["portfolio_value"].iloc[-1])

    # CAGR
    if V0 <= 0 or years <= 0:
        cagr = np.nan
    else:
        cagr = (Vend / V0) ** (1 / years) -1

    # Volatility (annualisée)
    daily_ret = results["portfolio_return"].astype(float)
    vol = float(daily_ret.std(ddof=1) * np.sqrt(trading_days))

    daily_rf = (1.0 + float(risk_free_rate)) ** (1.0 / trading_days) - 1.0
    excess = daily_ret - daily_rf
    sharpe = np.nan
    if vol > 0:
        sharpe = float(excess.mean() / daily_ret.std(ddof=1) * np.sqrt(trading_days))

    # Max Drawdown
    pv = results["portfolio_value"].astype(float)
    running_max = pv.cummax().clip(lower=V0)
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


def write_backtest_outputs(
    results: pd.DataFrame,
    summary: dict[str, float],
    partition_cols: list[str],
    base_dir: Path,
    existing_data_behavior: str = "overwrite_or_ignore",
    run_id: str | None = None,
    trades: pd.DataFrame | None = None,
    positions: pd.DataFrame | None = None,
    data_dir: Path | None = None,
) -> None:
    """Write backtest results to parquet and upsert summary in SQLite.

    Parameters
    ----------
    results : pd.DataFrame
        Backtest time series.
    summary : dict[str, float]
        Summary metrics to store in SQLite.
    partition_cols : list[str]
        Parquet partition columns (hive-style).
    base_dir : Path
        Output directory for the dataset.
    existing_data_behavior : str
        Behavior when data already exist.
    run_id : str | None
        Optional run identifier; defaults to a timestamp.
    """
    if results is None or results.empty:
        raise ValueError("results empty")
    if not summary:
        raise ValueError("summary empty")
    run_id = ensure_run_id(run_id, prefix="backtest")
    data_dir = data_dir or DATA_DIR
    from quant_portfolio.core.storage import write_partitioned_dataset

    df = results.copy()
    df["date"] = pd.to_datetime(df.index, errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    df["year"] = df["date"].dt.year.astype("int32")
    df["run_id"] = str(run_id)

    write_partitioned_dataset(
        df,
        base_dir,
        partition_cols,
        existing_data_behavior=existing_data_behavior,
    )
    if trades is not None and not trades.empty:
        trade_output = trades.copy()
        trade_output["date"] = pd.to_datetime(trade_output["execution_date"], errors="coerce")
        trade_output["run_id"] = str(run_id)
        trade_output["year"] = trade_output["date"].dt.year.astype("int32")
        write_partitioned_dataset(
            trade_output,
            data_dir / "parquet/trades",
            ["year", "run_id"],
            existing_data_behavior=existing_data_behavior,
        )
    if positions is not None and not positions.empty:
        position_output = positions.copy()
        position_output["date"] = pd.to_datetime(position_output["date"], errors="coerce")
        position_output["run_id"] = str(run_id)
        position_output["year"] = position_output["date"].dt.year.astype("int32")
        write_partitioned_dataset(
            position_output,
            data_dir / "parquet/positions",
            ["year", "run_id"],
            existing_data_behavior=existing_data_behavior,
        )
    conn = sqlite3.connect(data_dir / "_meta.db")
    init_backtest_db(conn)
    dates = df["date"].sort_values()
    date_start = dates.iloc[0].strftime("%Y-%m-%d")
    date_end = dates.iloc[-1].strftime("%Y-%m-%d")
    upsert_backtest_summary(conn, summary, run_id, date_start, date_end)
    conn.close()
    return None




def run_backtest_pipeline(run_id: str | None = None, plot: bool = False):
    """Run the backtest pipeline end-to-end.

    Parameters
    ----------
    run_id : str | None
        Optional run identifier to select weights.
    plot : bool
        If True, plot strategy vs baseline curves.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (results, baseline)
    """
    from quant_portfolio.core.storage import load_weights_dataset

    base = load_base_config()
    weights = load_weights_dataset(base.parquet_dir / "weights", run_id)
    effective_run_id = str(weights.attrs.get("run_id") or ensure_run_id(run_id, prefix="backtest"))
    snapshot_path = base.data_dir / "runs" / effective_run_id / "config.json"
    snapshot = json.loads(snapshot_path.read_text()) if snapshot_path.exists() else None
    if snapshot is None and (base.parquet_dir / "weights" / f"run_id={effective_run_id}").exists():
        raise ValueError("Incomplete optimization run: configuration snapshot is missing")
    cfg = BacktestConfig(**snapshot["backtest"]) if snapshot is not None else load_backtest_config()
    prices = load_reference_price_dataset(base)
    if snapshot is not None:
        from quant_portfolio.core.storage import load_regimes_dataset
        from quant_portfolio.pipeline.mc import regime_series
        from quant_portfolio.pipeline.optimize import input_fingerprint

        states = regime_series(load_regimes_dataset(base.parquet_dir / "regimes")) if snapshot["optimize"]["use_regimes"] else None
        fingerprint = input_fingerprint(build_returns_matrix(prices), states, pd.Timestamp(snapshot["decision_end"]))
        if fingerprint != snapshot["input_fingerprint"] or base.universe.fingerprint != snapshot["universe_fingerprint"]:
            raise ValueError("Backtest inputs differ from the optimization run; create a new run")
    returns = build_returns_matrix(prices, return_type=cfg.return_type)
    results, trades, positions = simulate_portfolio(
        returns,
        weights,
        cfg,
        return_details=True,
    )
    baseline = compute_baseline_hold(prices, cfg)
    summary = summarize_performance(results, risk_free_rate=cfg.cash_rate_annual)
    write_backtest_outputs(
        results,
        summary,
        partition_cols=["year", "run_id"],
        base_dir=base.parquet_dir / "backtests",
        run_id=effective_run_id,
        trades=trades,
        positions=positions,
        data_dir=base.data_dir,
    )
    if plot:
        plot_backtest_vs_baseline(results, baseline)
    return results, baseline



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run backtest pipeline.")
    parser.add_argument("--run-id", default=None, help="Optional run identifier to select weights.")
    parser.add_argument("--plot", action="store_true", help="Plot strategy vs baseline.")
    args = parser.parse_args()
    run_backtest_pipeline(run_id=args.run_id, plot=args.plot)
