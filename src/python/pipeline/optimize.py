from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
import sys

import numpy as np
import pandas as pd
try:
    import yaml
except Exception:
    yaml = None

PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from core.storage import (
    load_mc_dataset,
    load_prices_dataset,
    load_regimes_dataset,
    write_weights_dataset,
)


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
PRICES_DIR = DATA_DIR / "parquet/prices"
WEIGHTS_DIR = DATA_DIR / "parquet/weights"
CONFIG_PATH = ROOT / "config/optimize.yaml"
REGIME_DIR = DATA_DIR / "parquet/regimes"
MC_DIR = DATA_DIR / "parquet/mc"


@dataclass
class OptimizeConfig:
    """Configuration for the optimization pipeline.

    Attributes
    ----------
    rebal_freq : str
        Rebalance frequency ("D", "W", "2W", "M").
    max_weight : float
        Maximum per-asset weight cap.
    allow_cash : bool
        If True, allow weights to sum to less than 1.0.
    min_weight : float
        Minimum per-asset weight floor.
    lookback : int
        Lookback window length (in rows) for covariance estimation.
    use_regimes : bool
        If True, apply regime-based policy adjustments.
    use_mc : bool
        If True, apply MC risk overlay.
    mc_horizon : int
        Horizon used to select MC risk summary.
    risk_var_limit : float
        Absolute VaR limit for overlay scaling.
    risk_cvar_limit : float
        Absolute CVaR limit for overlay scaling.
    """
    rebal_freq: str
    max_weight: float
    allow_cash: bool
    min_weight: float
    lookback: int
    use_regimes: bool
    use_mc: bool
    mc_horizon: int
    risk_var_limit: float
    risk_cvar_limit: float


DEFAULT_CFG = OptimizeConfig(
    rebal_freq="W",
    max_weight=0.3,
    allow_cash=True,
    min_weight=0.01,
    lookback=10,
    use_regimes=True,
    use_mc=True,
    mc_horizon=5,
    risk_var_limit=0.05,
    risk_cvar_limit=0.08,
)


def load_optimize_config() -> OptimizeConfig:
    """Load optimizer configuration from YAML or defaults.

    Returns
    -------
    OptimizeConfig
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
    if cfg.lookback <= 1:
        raise ValueError("lookback must be > 1")
    if cfg.mc_horizon <= 0:
        raise ValueError("mc_horizon must be > 0")
    if cfg.risk_var_limit < 0 or cfg.risk_cvar_limit < 0:
        raise ValueError("risk_var_limit and risk_cvar_limit must be >= 0")
    return cfg





def build_returns_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot prices into a returns matrix indexed by date.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form prices with columns date, ticker, adj_close.

    Returns
    -------
    pd.DataFrame
        Log returns with dates as index and tickers as columns.
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


def get_regime_at_date(regimes: pd.DataFrame, date: pd.Timestamp) -> dict[str, float | int]:
    """Return the regime state/probabilities at or before a date (placeholder)."""
    if regimes is None or regimes.empty:
        raise ValueError("regimes is empty")
    df = regimes.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    if df.empty:
        raise ValueError("regimes has no valid dates")
    date = pd.to_datetime(date)
    row = df[df["date"] <= date].tail(1)
    if row.empty:
        raise ValueError("no regime available at or before date")
    r = row.iloc[0]
    return {
        "date": r["date"],
        "state": int(r["state"]),
        "p_state_0": float(r.get("p_state_0", np.nan)),
        "p_state_1": float(r.get("p_state_1", np.nan)),
        "p_state_2": float(r.get("p_state_2", np.nan)),
    }


def regime_policy_from_state(state: int, cfg: OptimizeConfig) -> dict[str, float]:
    """Map a regime state to optimization policy parameters (placeholder)."""
    if state is None:
        raise ValueError("state is None")
    if state == 0:
        risk_scale = 1.0
        max_weight_scale = 1.0
    elif state == 1:
        risk_scale = 0.7
        max_weight_scale = 0.8
    else : 
        risk_scale = 0.4
        max_weight_scale = 0.6
    return {
        "risk_scale": risk_scale,
        "max_weight_scale": max_weight_scale,
        "base_max_weight": float(cfg.max_weight),
    }



def apply_regime_policy(
    weights: np.ndarray,
    policy: dict[str, float],
    allow_cash: bool,
) -> np.ndarray:
    """Apply regime-based policy adjustments to weights (placeholder)."""
    w = np.asarray(weights, dtype=float)
    base_max_weight = float(policy.get("base_max_weight", np.max(w) if w.size else 0.0))
    w = w * float(policy.get("risk_scale", 1.0))
    w = np.clip(w, 0.0, float(policy.get("max_weight_scale", 1.0)) * base_max_weight)
    if not allow_cash:
        s = w.sum()
        if s > 0:
            w = w / s
    return w



def get_mc_risk_at_date(
    mc: pd.DataFrame,
    date: pd.Timestamp,
    horizon: int,
) -> dict[str, float]:
    """Return MC risk metrics at or before a date (placeholder)."""
    if mc is None or mc.empty:
        raise ValueError("mc is empty")
    df = mc.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    if df.empty:
        raise ValueError("mc has no valid dates")
    date = pd.to_datetime(date)
    df = df[df["horizon"] == int(horizon)]
    row = df[df["date"] <= date].tail(1)
    if row.empty:
        raise ValueError("no mc data available at or before date for this horizon")
    r = row.iloc[0]
    return {
        "date": r["date"],
        "state": int(r["state"]),
        "var": float(r["var"]),
        "cvar": float(r["cvar"]),
        "q95": float(r["q95"]),
    }


def apply_mc_overlay(
    weights: np.ndarray,
    mc_summary: dict[str, float],
    risk_limits: dict[str, float],
    allow_cash: bool,
) -> np.ndarray:
    """Apply an MC risk overlay to weights (placeholder)."""
    w = np.asarray(weights, dtype=float)
    scale = 1.0
    var = float(mc_summary.get("var", 0.0))
    cvar = float(mc_summary.get("cvar", 0.0))
    var_limit = float(risk_limits.get("var", np.inf))
    cvar_limit = float(risk_limits.get("cvar", np.inf))

    if var < -var_limit and var != 0.0:
        scale = min(scale, var_limit / abs(var))
    if cvar < -cvar_limit and cvar != 0.0:
        scale = min(scale, cvar_limit / abs(cvar))

    w = w * scale
    if not allow_cash and w.sum() > 0:
        w = w / w.sum()
    return w




def min_variance_weights(cov: np.ndarray, max_weight: float, min_weight: float) -> np.ndarray:
    """Compute a simple inverse-variance long-only weight vector.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix.
    max_weight : float
        Upper bound for each weight.
    min_weight : float
        Lower bound for each weight.

    Returns
    -------
    np.ndarray
        Normalized weight vector that sums to 1.
    """
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



def compute_covariance(returns_window: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Compute covariance from a return window and list of used assets.

    Parameters
    ----------
    returns_window : pd.DataFrame
        Returns subset used to compute covariance.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        Covariance matrix and list of asset columns kept.
    """
    if returns_window is None or returns_window.empty:
        raise ValueError("returns window empty")
    data = returns_window.copy()
    data = data.dropna(how="all")
    if data.empty:
        raise ValueError("returns window empty after dropna")
    # Keep only columns with at least 2 non-NaN points
    valid = data.columns[data.count() >= 2]
    data = data[valid]
    if data.shape[1] == 0:
        raise ValueError("no assets with sufficient history for covariance")
    cov = data.cov(min_periods=2).to_numpy()
    return cov, list(valid)



def optimize_over_time(
    returns: pd.DataFrame,
    rebal_dates: pd.DatetimeIndex,
    cfg: OptimizeConfig,
) -> pd.DataFrame:
    """Compute weights at each rebalance date using a rolling window.

    Parameters
    ----------
    returns : pd.DataFrame
        Returns matrix.
    rebal_dates : pd.DatetimeIndex
        Dates when the portfolio should be rebalanced.
    cfg : OptimizeConfig
        Optimization configuration.

    Returns
    -------
    pd.DataFrame
        Long-form weights with columns: date, ticker, weight.
    """
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
    regimes = (
        load_regimes_dataset(
            REGIME_DIR,
            columns=["date", "state", "p_state_0", "p_state_1", "p_state_2"],
        )
        if cfg.use_regimes
        else None
    )
    mc = (
        load_mc_dataset(
            MC_DIR,
            columns=["date", "state", "horizon", "var", "cvar", "q95"],
        )
        if cfg.use_mc
        else None
    )
    risk_limits = {"var": cfg.risk_var_limit, "cvar": cfg.risk_cvar_limit}

    for dt in pd.DatetimeIndex(rebal_dates):
        hist = returns.loc[:dt].tail(window)
        if hist.empty:
            continue
        try:
            cov, used_cols = compute_covariance(hist)
        except ValueError:
            continue
        w = min_variance_weights(cov, cfg.max_weight, cfg.min_weight)
        # Map back to full universe, zero for missing tickers
        w_full = np.zeros(len(tickers), dtype=float)
        col_idx = {c: i for i, c in enumerate(tickers)}
        for c, wc in zip(used_cols, w, strict=False):
            if c in col_idx:
                w_full[col_idx[c]] = float(wc)
        s = w_full.sum()
        if s > 0:
            w_full = w_full / s

        if cfg.use_regimes:
            try:
                reg = get_regime_at_date(regimes, dt)
                policy = regime_policy_from_state(reg["state"], cfg)
                w_full = apply_regime_policy(w_full, policy, cfg.allow_cash)
            except Exception:
                pass

        if cfg.use_mc:
            try:
                mc_summary = get_mc_risk_at_date(mc, dt, cfg.mc_horizon)
                w_full = apply_mc_overlay(w_full, mc_summary, risk_limits, cfg.allow_cash)
            except Exception:
                pass

        for t, wt in zip(tickers, w_full, strict=False):
            weights_rows.append({"date": dt, "ticker": t, "weight": float(wt)})

    if not weights_rows:
        raise ValueError("No weights computed (check lookback/rebal_dates).")
    return pd.DataFrame(weights_rows)



def run_optimize_pipeline(
    run_id: str | None = None,
    existing_data_behavior: str = "overwrite_or_ignore",
) -> str:
    """Run the optimize pipeline end-to-end and return the run id.

    Parameters
    ----------
    run_id : str | None
        Optional run identifier.
    existing_data_behavior : str
        Behavior when data already exist.

    Returns
    -------
    str
        The run identifier used for the output dataset.
    """
    config = load_optimize_config()
    prices = load_prices_dataset(PRICES_DIR, columns=["date", "ticker", "adj_close"])
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
    print(f"Optimize run_id: {out_run_id}")
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
