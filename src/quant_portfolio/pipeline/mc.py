"""Causal Monte-Carlo risk of a weighted buy-and-hold portfolio."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

from quant_portfolio.core.ids import validate_run_id
from quant_portfolio.core.settings import PROJECT_ROOT, load_base_config, load_yaml_mapping
from quant_portfolio.core.storage import (
    load_regimes_dataset,
    read_parquet_dataset,
    replace_partitioned_dataset,
)
from quant_portfolio.models.covariance import CovarianceConfig, estimate_covariance

LOGGER = logging.getLogger(__name__)
CONFIG_PATH = PROJECT_ROOT / "config/mc.yaml"


@dataclass(frozen=True)
class MCConfig:
    n_sims: int = 2000
    horizons: tuple[int, ...] = (5, 20)
    window: int = 252
    min_observations: int = 20
    random_seed: int = 42
    dist: str = "gaussian"
    compare_distributions: tuple[str, ...] = ("student_t",)
    student_df: float = 8.0
    loss_threshold: float = 0.05
    drawdown_threshold: float = 0.10
    sparse_regime_policy: str = "pooled"

    def __post_init__(self) -> None:
        for name in ("n_sims", "window", "min_observations", "random_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.n_sims < 100 or self.min_observations < 2 or self.window < self.min_observations:
            raise ValueError("MC requires >=100 simulations and window >= min_observations >= 2")
        if self.random_seed < 0 or not np.isfinite(self.student_df) or self.student_df <= 2:
            raise ValueError("MC seed must be >= 0 and student_df > 2")
        if not self.horizons or any(type(h) is not int or h <= 0 for h in self.horizons):
            raise ValueError("MC horizons must contain positive integers")
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError("MC horizons must be unique")
        if set((self.dist, *self.compare_distributions)) - {"gaussian", "student_t"}:
            raise ValueError("MC distribution must be gaussian or student_t")
        if not 0 < self.loss_threshold < 1 or not 0 < self.drawdown_threshold < 1:
            raise ValueError("MC loss/drawdown thresholds must be in (0, 1)")
        if self.sparse_regime_policy not in {"pooled", "error"}:
            raise ValueError("sparse_regime_policy must be pooled or error")


def load_mc_config(path: Path | None = None) -> MCConfig:
    data = load_yaml_mapping(path or CONFIG_PATH)
    unknown = set(data) - {field.name for field in fields(MCConfig)}
    if unknown:
        raise ValueError(f"Unknown MC fields: {sorted(unknown)}")
    data.setdefault("random_seed", load_base_config().random_seed)
    for name in ("horizons", "compare_distributions"):
        if name in data:
            data[name] = tuple(data[name])
    return MCConfig(**data)


class InsufficientHistory(ValueError):
    """No defensible historical calibration is available yet."""


def regime_series(regimes: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(regimes, pd.Series):
        states = regimes.copy()
    else:
        data = regimes.set_index("date") if "date" in regimes.columns else regimes
        states = data["state"].copy()
    states.index = pd.DatetimeIndex(states.index)
    if states.index.has_duplicates or not states.dropna().isin([0, 1, 2]).all():
        raise ValueError("Regimes require unique dates and semantic states 0/1/2")
    return states.dropna().sort_index()


def state_at_date(regimes: pd.Series, date: pd.Timestamp) -> int:
    available = regimes.loc[regimes.index <= date]
    if available.empty:
        raise InsufficientHistory("No causal regime available")
    return int(available.iloc[-1])


@dataclass(frozen=True)
class MCCalibration:
    columns: tuple[str, ...]
    mu: np.ndarray
    covariance: np.ndarray
    state: int | None
    calibration_start: pd.Timestamp
    calibration_end: pd.Timestamp
    observations: int
    calibration_status: str
    covariance_estimator: str


def calibrate_mc(
    returns: pd.DataFrame,
    date: pd.Timestamp,
    columns: list[str],
    cfg: MCConfig,
    covariance_cfg: CovarianceConfig,
    regimes: pd.Series | None = None,
) -> MCCalibration:
    """Fit log-return parameters on dates strictly before the decision.

    Inputs are simple returns. Sparse-regime fallback is recorded in every
    summary, never silently pooled. Weighted assets are never dropped.
    """
    if covariance_cfg.method == "sample":
        raise ValueError("MC requires a shrunk covariance estimator")
    if covariance_cfg.min_periods > cfg.window:
        raise ValueError("MC window is shorter than covariance min_periods")
    hist = returns.loc[returns.index < date, columns].sort_index()
    if np.isinf(hist.to_numpy()).any() or (hist <= -1).any().any():
        raise ValueError("MC log-return calibration requires returns > -1 and no infinities")
    hist = np.log1p(hist).dropna(how="all")
    state = state_at_date(regimes, date) if regimes is not None else None
    selected = hist
    if regimes is not None:
        labels = regimes.reindex(hist.index, method="ffill")
        selected = hist.loc[labels == state]
    status = "conditional" if regimes is not None else "unconditional"
    minimum = max(cfg.min_observations, covariance_cfg.min_periods)

    def fit(frame: pd.DataFrame):
        frame = frame.tail(cfg.window)
        if len(frame) < minimum or (frame.count() < minimum).any():
            raise InsufficientHistory("Insufficient MC observations for every weighted asset")
        try:
            estimate = estimate_covariance(frame, covariance_cfg)
        except ValueError as exc:
            raise InsufficientHistory(str(exc)) from exc
        return frame, estimate

    try:
        sample, estimate = fit(selected)
    except InsufficientHistory:
        if regimes is None or cfg.sparse_regime_policy != "pooled":
            raise
        sample, estimate = fit(hist)
        status = "pooled_sparse_regime"
        LOGGER.info("MC %s state=%s: explicit pooled-history fallback", date.date(), state)
    return MCCalibration(
        tuple(columns),
        sample.mean().to_numpy(),
        estimate.covariance,
        state,
        sample.index.min(),
        sample.index.max(),
        len(sample),
        status,
        estimate.estimator,
    )


def date_seed(seed: int, date: pd.Timestamp, distribution: str) -> int:
    """Stable across process order, appended data, and Python hash randomization."""
    payload = f"{seed}|{pd.Timestamp(date).isoformat()}|{distribution}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def simulate_paths(
    mu: np.ndarray,
    sigma: np.ndarray,
    n_sims: int,
    horizon: int,
    dist: str = "gaussian",
    *,
    seed: int = 42,
    student_df: float = 8.0,
) -> np.ndarray:
    """Correlated daily log returns, shape (simulation, day, asset).

    Student innovations have the same covariance as Gaussian innovations.
    A shared chi-square scale per scenario/day preserves tail dependence.
    """
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu, dtype=float)
    innovations = rng.standard_normal((horizon, n_sims, len(mu))) @ np.linalg.cholesky(sigma).T
    if dist == "student_t":
        if student_df <= 2:
            raise ValueError("student_df must exceed 2")
        innovations *= np.sqrt((student_df - 2) / rng.chisquare(student_df, (horizon, n_sims, 1)))
    elif dist != "gaussian":
        raise ValueError("Unknown MC distribution")
    return (innovations + mu).transpose(1, 0, 2)


def simulate_calibration(model: MCCalibration, date: pd.Timestamp, cfg: MCConfig) -> dict[str, np.ndarray]:
    return {
        dist: simulate_paths(
            model.mu,
            model.covariance,
            cfg.n_sims,
            max(cfg.horizons),
            dist,
            seed=date_seed(cfg.random_seed, date, dist),
            student_df=cfg.student_df,
        )
        for dist in dict.fromkeys((cfg.dist, *cfg.compare_distributions))
    }


def summarize_paths(
    paths: np.ndarray,
    weights: np.ndarray,
    *,
    cash_return: float = 0.0,
    loss_threshold: float = 0.05,
    drawdown_threshold: float = 0.10,
) -> dict[str, float]:
    """Risk as fraction of initial NAV; assets drift buy-and-hold through paths.

    VaR/CVaR are nonnegative losses, PnL quantiles signed; initial NAV is a peak.
    """
    weights = np.asarray(weights, dtype=float)
    if paths.ndim != 3 or paths.shape[2] != len(weights):
        raise ValueError("MC paths and portfolio weights must align")
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() > 1 + 1e-10:
        raise ValueError("MC weights must be finite, long-only, with exposure <= 1")
    with np.errstate(over="raise", invalid="raise"):
        levels = np.exp(paths.cumsum(axis=1)) @ weights
    days = np.arange(1, paths.shape[1] + 1)
    levels += max(0.0, 1 - weights.sum()) * (1 + cash_return) ** days
    if not np.isfinite(levels).all():
        raise ValueError("MC portfolio paths are non-finite")
    pnl = levels[:, -1] - 1.0
    peaks = np.maximum.accumulate(np.column_stack([np.ones(len(levels)), levels]), axis=1)[:, 1:]
    drawdown = (1 - levels / peaks).max(axis=1)
    result = {f"pnl_q{int(q * 100):02d}": float(np.quantile(pnl, q)) for q in (0.01, 0.05, 0.5, 0.95, 0.99)}
    for alpha, suffix in ((0.01, "01"), (0.05, "05")):
        quantile = np.quantile(pnl, alpha)
        result[f"var_{suffix}"] = max(0.0, float(-quantile))
        result[f"cvar_{suffix}"] = max(0.0, float(-pnl[pnl <= quantile].mean()))
    result.update(
        expected_pnl=float(pnl.mean()),
        probability_loss=float((pnl < -loss_threshold).mean()),
        probability_drawdown=float((drawdown > drawdown_threshold).mean()),
        loss_threshold=loss_threshold,
        drawdown_threshold=drawdown_threshold,
    )
    return result


def summarize_portfolio(
    model: MCCalibration,
    paths: dict[str, np.ndarray],
    weights: pd.Series,
    date: pd.Timestamp,
    role: str,
    cfg: MCConfig,
    cash_return: float = 0.0,
) -> list[dict]:
    if weights.drop(labels=list(model.columns), errors="ignore").abs().sum() > 1e-12:
        raise ValueError("MC calibration is missing a weighted asset")
    aligned = weights.reindex(model.columns, fill_value=0.0).to_numpy()
    weight_json = json.dumps({str(k): float(v) for k, v in weights.items()}, sort_keys=True)
    rows = []
    for dist, simulations in paths.items():
        for horizon in cfg.horizons:
            rows.append(
                {
                    "date": date,
                    "portfolio_role": role,
                    "state": model.state,
                    "horizon": horizon,
                    "distribution": dist,
                    "random_seed": cfg.random_seed,
                    "n_sims": cfg.n_sims,
                    "student_df": cfg.student_df,
                    "exposure": float(weights.sum()),
                    "weights_json": weight_json,
                    "cash_return": cash_return,
                    "calibration_start": model.calibration_start,
                    "calibration_end": model.calibration_end,
                    "observations": model.observations,
                    "calibration_status": model.calibration_status,
                    "covariance_estimator": model.covariance_estimator,
                    **summarize_paths(
                        simulations[:, :horizon],
                        aligned,
                        cash_return=cash_return,
                        loss_threshold=cfg.loss_threshold,
                        drawdown_threshold=cfg.drawdown_threshold,
                    ),
                }
            )
    return rows


def run_mc_pipeline(existing_data_behavior: str = "overwrite_or_ignore", run_id: str | None = None) -> str:
    """Re-evaluate saved decision-time weights; never feed a replay to optimization."""
    if run_id is None:
        raise ValueError("mc requires --run-id from an existing optimize run")
    run_id = validate_run_id(run_id)
    base = load_base_config()
    from quant_portfolio.pipeline.backtest import build_returns_matrix
    from quant_portfolio.pipeline.data_quality import load_reference_price_dataset
    from quant_portfolio.pipeline.optimize import input_fingerprint

    snapshot = json.loads((base.data_dir / "runs" / run_id / "config.json").read_text())
    cfg = load_mc_config()
    covariance_cfg = CovarianceConfig(**snapshot["covariance"])
    returns = build_returns_matrix(load_reference_price_dataset(base))
    inputs = read_parquet_dataset(base.parquet_dir / "risk_weights" / f"run_id={run_id}")
    regimes = None
    if snapshot["optimize"]["use_regimes"]:
        regimes = regime_series(load_regimes_dataset(base.parquet_dir / "regimes"))
    if (
        snapshot["universe_fingerprint"] != base.universe.fingerprint
        or input_fingerprint(returns, regimes, pd.Timestamp(snapshot["decision_end"]))
        != snapshot["input_fingerprint"]
    ):
        raise ValueError("MC replay inputs differ from the optimization run; create a new run")
    cash_return = (1 + snapshot["backtest"]["cash_rate_annual"]) ** (1 / 252) - 1
    rows = []
    for date, group in inputs.groupby("date", sort=True):
        columns = [
            ticker for ticker in returns.columns if group.loc[group.ticker == ticker, "in_calibration"].any()
        ]
        if not columns:
            continue
        model = calibrate_mc(returns, pd.Timestamp(date), columns, cfg, covariance_cfg, regimes)
        paths = simulate_calibration(model, pd.Timestamp(date), cfg)
        for role, portfolio in group.groupby("portfolio_role"):
            weights = portfolio.set_index("ticker")["weight"]
            rows.extend(summarize_portfolio(model, paths, weights, date, role, cfg, cash_return))
    if not rows:
        raise ValueError("No calibrated risk weights to replay")
    destination = base.parquet_dir / "mc_replay" / f"run_id={run_id}"
    if existing_data_behavior == "error" and destination.exists():
        raise FileExistsError(destination)
    replace_partitioned_dataset(pd.DataFrame(rows), destination, ["year"])
    (base.data_dir / "runs" / run_id / "mc_replay.json").write_text(
        json.dumps(
            {"mc": asdict(cfg), "source_input_fingerprint": snapshot["input_fingerprint"]},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    LOGGER.info("MC replay %s: %s", run_id, asdict(cfg))
    return run_id
