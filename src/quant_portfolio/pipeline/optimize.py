"""Walk-forward allocation and daily risk overlay using effective holdings."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import numpy as np
import pandas as pd

from quant_portfolio.core.ids import ensure_run_id
from quant_portfolio.core.portfolio import drift_weights, execution_weights
from quant_portfolio.core.provenance import collect_provenance
from quant_portfolio.core.settings import PROJECT_ROOT, load_base_config, load_yaml_mapping
from quant_portfolio.core.storage import load_regimes_dataset, replace_partitioned_dataset
from quant_portfolio.models.allocation import solve_allocation
from quant_portfolio.models.covariance import CovarianceConfig, estimate_covariance, load_covariance_config
from quant_portfolio.pipeline.backtest import (
    BacktestConfig,
    build_returns_matrix,
    compute_turnover,
    load_backtest_config,
)
from quant_portfolio.pipeline.data_quality import load_reference_price_dataset
from quant_portfolio.pipeline.mc import (
    InsufficientHistory,
    MCConfig,
    calibrate_mc,
    load_mc_config,
    regime_series,
    simulate_calibration,
    state_at_date,
    summarize_portfolio,
)

CONFIG_PATH = PROJECT_ROOT / "config/optimize.yaml"
LOGGER = logging.getLogger(__name__)
REGIME_NAMES = {0: "calm", 1: "choppy", 2: "stress"}


@dataclass(frozen=True)
class OptimizeConfig:
    rebal_freq: str = "W"
    max_weight: float = 0.30
    allow_cash: bool = True
    min_weight: float = 0.0
    lookback: int = 60
    use_regimes: bool = True
    use_mc: bool = True
    use_overlay: bool = True
    mc_horizon: int = 5
    risk_var_limit: float = 0.05
    risk_cvar_limit: float = 0.08
    target_vol: float = 0.15
    min_exposure: float = 0.0
    max_exposure: float = 1.0
    min_multiplier: float = 0.0
    max_multiplier: float = 1.0
    hysteresis: float = 0.02
    max_exposure_change: float | None = 0.10
    turnover_governor: float | None = 0.15
    turnover_penalty: float = 0.002
    distance_penalty: float = 0.01
    exposure_penalty: float = 100.0
    target_vol_penalty: float = 1.0
    solver: str = "CLARABEL"
    solver_fallback: str = "error"
    regime_exposure: dict[str, float] = field(
        default_factory=lambda: {"calm": 1.0, "choppy": 0.7, "stress": 0.4}
    )
    regime_cap_scale: dict[str, float] = field(
        default_factory=lambda: {"calm": 1.0, "choppy": 0.8, "stress": 0.6}
    )
    sector_caps: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("allow_cash", "use_regimes", "use_mc", "use_overlay"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.rebal_freq not in {"D", "W", "2W", "M"}:
            raise ValueError("rebal_freq must be D, W, 2W or M")
        if (
            type(self.lookback) is not int
            or self.lookback < 2
            or type(self.mc_horizon) is not int
            or self.mc_horizon < 1
        ):
            raise ValueError("lookback and mc_horizon must be positive integer windows")
        if not 0 <= self.min_weight <= self.max_weight <= 1 or self.max_weight == 0:
            raise ValueError("min_weight/max_weight must satisfy 0 <= min_weight <= max_weight <= 1")
        if not 0 <= self.min_exposure <= self.max_exposure <= 1:
            raise ValueError("Exposure bounds must satisfy 0 <= min <= max <= 1")
        if not 0 <= self.min_multiplier <= self.max_multiplier <= 1:
            raise ValueError("Multiplier bounds must satisfy 0 <= min <= max <= 1")
        for name in ("target_vol", "risk_var_limit", "risk_cvar_limit", "exposure_penalty"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and > 0")
        for name in ("hysteresis", "turnover_penalty", "distance_penalty", "target_vol_penalty"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and >= 0")
        for cap in (self.max_exposure_change, self.turnover_governor):
            if cap is not None and not 0 <= cap <= 1:
                raise ValueError("Exposure speed and turnover caps must be in [0, 1] or null")
        for policy in (self.regime_exposure, self.regime_cap_scale):
            if set(policy) != set(REGIME_NAMES.values()) or any(not 0 < x <= 1 for x in policy.values()):
                raise ValueError("Regime policies require calm/choppy/stress values in (0, 1]")
        if any(not isinstance(s, str) or not 0 <= cap <= 1 for s, cap in self.sector_caps.items()):
            raise ValueError("Sector caps must be a sector-to-weight mapping in [0, 1]")
        if self.solver_fallback not in {"error", "hold"}:
            raise ValueError("solver_fallback must be error or hold")
        if not self.allow_cash and (self.min_exposure > 1 or self.max_exposure < 1):
            raise ValueError("No-cash portfolios require total exposure 1")


DEFAULT_CFG = OptimizeConfig()


def resolve_execution_config(cfg: OptimizeConfig, backtest: BacktestConfig) -> BacktestConfig:
    """Apply the governors both to decision targets and to delayed executions."""
    turnover_caps = [cap for cap in (cfg.turnover_governor, backtest.turnover_cap) if cap is not None]
    strategy_speed = cfg.max_exposure_change if cfg.use_overlay and cfg.allow_cash else None
    speed_caps = [cap for cap in (strategy_speed, backtest.exposure_change_cap) if cap is not None]
    return replace(
        backtest,
        turnover_cap=min(turnover_caps) if turnover_caps else None,
        exposure_change_cap=min(speed_caps) if speed_caps else None,
    )


def load_optimize_config(path: Path | None = None) -> OptimizeConfig:
    data = load_yaml_mapping(path or CONFIG_PATH)
    unknown = set(data) - {f.name for f in fields(OptimizeConfig)}
    if unknown:
        raise ValueError(f"Unknown optimizer fields: {sorted(unknown)}")
    return OptimizeConfig(**data)


def build_rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """First observed session of a calendar period; invariant to appended rows."""
    dates = pd.DatetimeIndex(index).sort_values().unique()
    if freq == "D" or dates.empty:
        return dates
    if freq == "M":
        groups = dates.to_period("M").asi8
    elif freq in {"W", "2W"}:
        groups = dates.to_period("W-SUN").asi8
        if freq == "2W":
            groups = groups // 2  # Fixed epoch anchor, not length of this sample.
    else:
        raise ValueError("Frequency must be D, W, 2W or M")
    return dates[np.r_[True, groups[1:] != groups[:-1]]]


def overlay_exposure(
    forecast_vol: float,
    current_exposure: float,
    base_exposure: float,
    state: int | None,
    mc_risk: dict | None,
    cfg: OptimizeConfig,
) -> dict[str, float | str]:
    """Desired exposure before hard speed/turnover limits in the solver."""
    vol_scale = float(
        np.clip(cfg.target_vol / max(forecast_vol, 1e-12), cfg.min_multiplier, cfg.max_multiplier)
    )
    regime_scale = cfg.regime_exposure[REGIME_NAMES[state]] if cfg.use_regimes and state is not None else 1.0
    mc_scale = 1.0
    if cfg.use_mc and mc_risk is not None:
        mc_scale = min(
            1.0,
            cfg.risk_var_limit / max(mc_risk["var_05"], 1e-12),
            cfg.risk_cvar_limit / max(mc_risk["cvar_05"], 1e-12),
        )
    desired = float(
        np.clip(base_exposure * vol_scale * regime_scale * mc_scale, cfg.min_exposure, cfg.max_exposure)
    )
    reason = "risk_target"
    if not cfg.use_overlay:
        desired, reason = min(base_exposure, cfg.max_exposure), "overlay_disabled"
    if not cfg.allow_cash:
        desired, reason = 1.0, "cash_disabled"
    elif cfg.use_overlay and abs(desired - current_exposure) < cfg.hysteresis:
        desired, reason = current_exposure, "hysteresis"
    return dict(
        desired_exposure=desired,
        vol_multiplier=vol_scale,
        regime_multiplier=regime_scale,
        mc_multiplier=mc_scale,
        reason=reason,
    )


@dataclass
class StrategyResult:
    weights: pd.DataFrame
    decisions: pd.DataFrame
    risk: pd.DataFrame
    risk_weights: pd.DataFrame


def optimize_over_time(
    returns: pd.DataFrame,
    rebal_dates: pd.DatetimeIndex | None,
    cfg: OptimizeConfig,
    *,
    regimes: pd.DataFrame | pd.Series | None = None,
    mc_cfg: MCConfig | None = None,
    covariance_cfg: CovarianceConfig | None = None,
    backtest_cfg: BacktestConfig | None = None,
    sectors: dict[str, str | None] | None = None,
    return_details: bool = False,
) -> pd.DataFrame | StrategyResult:
    """Decide after each close; execute at t+lag using shared backtest accounting.

    Model histories exclude the decision day's return. Effective weights include
    that observed return; they never include the next execution day's return.
    """
    mc_cfg = mc_cfg or load_mc_config()
    covariance_cfg = covariance_cfg or load_covariance_config()
    backtest_cfg = resolve_execution_config(cfg, backtest_cfg or load_backtest_config())
    if cfg.lookback < covariance_cfg.min_periods:
        raise ValueError("lookback must cover covariance min_periods")
    if cfg.use_mc and cfg.mc_horizon not in mc_cfg.horizons:
        raise ValueError("mc_horizon is absent from configured MC horizons")
    if cfg.use_regimes and regimes is None:
        raise ValueError("use_regimes requires causal regime observations")
    states = regime_series(regimes) if cfg.use_regimes else None
    data = returns.sort_index().copy()
    data.index = pd.DatetimeIndex(data.index)
    if data.empty or data.index.has_duplicates or data.columns.has_duplicates:
        raise ValueError("Returns require nonempty, unique dates and tickers")
    if np.isinf(data.to_numpy()).any() or (data < -1).any().any():
        raise ValueError("Optimizer inputs must be simple returns >= -1 without infinities")
    tickers = list(data.columns)
    metadata = [sectors.get(ticker) for ticker in tickers] if sectors is not None else [None] * len(tickers)
    if cfg.sector_caps and (any(s is None for s in metadata) or set(cfg.sector_caps) - set(metadata)):
        raise ValueError("Sector caps require complete universe metadata and known sector names")
    dates = data.index
    if backtest_cfg.start_date:
        dates = dates[dates >= pd.Timestamp(backtest_cfg.start_date)]
    if backtest_cfg.end_date:
        dates = dates[dates <= pd.Timestamp(backtest_cfg.end_date)]
    if dates.empty:
        raise ValueError("No strategy dates in the configured backtest range")
    rebalances = set(build_rebalance_dates(dates, cfg.rebal_freq) if rebal_dates is None else rebal_dates)
    cash_return = (1 + backtest_cfg.cash_rate_annual) ** (1 / 252) - 1
    fee_rate = (backtest_cfg.transaction_bps + backtest_cfg.slippage_bps) / 10000
    caps = [c for c in (cfg.turnover_governor, backtest_cfg.turnover_cap) if c is not None]
    turnover_cap = min(caps) if caps else None
    current = np.zeros(len(tickers))
    pending: dict[int, np.ndarray] = {}
    previous_state, previous_eligible = None, None
    last_candidate = None
    weight_rows, decision_rows, risk_rows, risk_weight_rows = [], [], [], []

    for position, date in enumerate(dates):
        if position in pending:
            requested = pending.pop(position)
            current = execution_weights(
                current, requested, backtest_cfg.turnover_cap, backtest_cfg.exposure_change_cap
            )
        current, _, _ = drift_weights(
            current, data.loc[date].to_numpy(), cash_return, backtest_cfg.missing_return_policy
        )
        current_exposure = float(current.sum())
        log = dict(
            date=date,
            effective_exposure=current_exposure,
            effective_weights_json=json.dumps(dict(zip(tickers, current, strict=True)), sort_keys=True),
        )
        hist = data.loc[data.index < date].tail(cfg.lookback)
        eligible = hist.columns[hist.count() >= covariance_cfg.min_periods].tolist()
        try:
            if not eligible:
                raise InsufficientHistory("covariance warmup")
            if any(current[i] > 1e-12 and ticker not in eligible for i, ticker in enumerate(tickers)):
                raise InsufficientHistory("held asset lacks covariance history")
            state = state_at_date(states, date) if states is not None else None
            try:
                estimate = estimate_covariance(hist[eligible], covariance_cfg)
            except ValueError as exc:
                raise InsufficientHistory(str(exc)) from exc
            indices = [tickers.index(t) for t in eligible]
            covariance = np.eye(len(tickers)) * covariance_cfg.eps
            covariance[np.ix_(indices, indices)] = estimate.covariance
            cap_scale = cfg.regime_cap_scale[REGIME_NAMES[state]] if state is not None else 1.0
            maximum = np.zeros(len(tickers))
            maximum[indices] = cfg.max_weight * cap_scale
            minimum = np.zeros(len(tickers))
            minimum[indices] = cfg.min_weight
            capacity = float(maximum.sum())
            if cfg.sector_caps:
                capacity = sum(
                    min(float(maximum[np.array(metadata) == s].sum()), cfg.sector_caps.get(s, 1.0))
                    for s in set(metadata)
                )
            base_exposure = min(1.0, capacity)
            if base_exposure <= 0:
                raise ValueError("Constraints leave no risky allocation capacity")
            composition_day = (
                date in rebalances
                or last_candidate is None
                or state != previous_state
                or eligible != previous_eligible
            )
            candidate = (
                current / current_exposure * base_exposure if current_exposure > 1e-12 else last_candidate
            )
            if candidate is not None and (
                (candidate > maximum + 1e-8).any() or (candidate < minimum - 1e-8).any()
            ):
                composition_day = True
            if candidate is not None and any(
                candidate[np.array(metadata) == sector].sum() > cap + 1e-8
                for sector, cap in cfg.sector_caps.items()
            ):
                composition_day = True
            if composition_day:
                base = solve_allocation(
                    covariance,
                    current,
                    minimum=minimum,
                    maximum=maximum,
                    allow_cash=True,
                    fixed_exposure=base_exposure,
                    desired_exposure=base_exposure,
                    turnover_penalty=cfg.turnover_penalty,
                    transaction_cost_rate=fee_rate,
                    distance_penalty=cfg.distance_penalty,
                    exposure_penalty=0.0,
                    target_vol_penalty=0.0,
                    sectors=metadata,
                    sector_caps=cfg.sector_caps,
                    solver=cfg.solver,
                    fallback="error",
                )
                candidate = base.weights
                base_status = base.status
            else:
                base_status = "drifted_composition"
            if not cfg.use_overlay and not composition_day:
                decision_rows.append({**log, "status": "no_decision", "reason": "between_rebalances"})
                continue
            forecast_vol = float(np.sqrt(252 * candidate @ covariance @ candidate))
            portfolios = {
                "effective": pd.Series(current, index=tickers),
                "candidate": pd.Series(candidate, index=tickers),
            }
            model, simulations, candidate_risk = None, None, None
            day_risk = []
            if cfg.use_mc:
                model = calibrate_mc(data, date, eligible, mc_cfg, covariance_cfg, states)
                simulations = simulate_calibration(model, date, mc_cfg)
                for role, weights in portfolios.items():
                    day_risk.extend(
                        summarize_portfolio(model, simulations, weights, date, role, mc_cfg, cash_return)
                    )
                candidate_risk = next(
                    row
                    for row in day_risk
                    if row["portfolio_role"] == "candidate"
                    and row["horizon"] == cfg.mc_horizon
                    and row["distribution"] == mc_cfg.dist
                )
        except InsufficientHistory as exc:
            # No order is sent. Existing holdings keep drifting; this is not a
            # falsely "validated" new target, nor a silent model failure.
            LOGGER.info("No decision %s: %s", date.date(), exc)
            decision_rows.append({**log, "status": "history_hold", "reason": str(exc)})
            continue
        overlay = overlay_exposure(forecast_vol, current_exposure, base_exposure, state, candidate_risk, cfg)
        requested_exposure = float(overlay["desired_exposure"])
        # Exposure bounds are hard alongside per-name floors. Target-vol and
        # regime/MC exposure requests remain soft under speed/turnover limits.
        final = solve_allocation(
            covariance,
            current,
            minimum=minimum,
            maximum=maximum,
            desired_exposure=requested_exposure,
            fixed_exposure=current_exposure
            if overlay["reason"] == "hysteresis" and not composition_day
            else None,
            allow_cash=cfg.allow_cash,
            turnover_penalty=cfg.turnover_penalty,
            transaction_cost_rate=fee_rate,
            distance_penalty=cfg.distance_penalty,
            exposure_penalty=cfg.exposure_penalty,
            target_vol=cfg.target_vol,
            target_vol_penalty=cfg.target_vol_penalty if cfg.use_overlay else 0.0,
            turnover_cap=turnover_cap,
            exposure_change_cap=backtest_cfg.exposure_change_cap,
            sectors=metadata,
            sector_caps=cfg.sector_caps,
            composition=None if composition_day else candidate / candidate.sum(),
            solver=cfg.solver,
            fallback=cfg.solver_fallback,
            min_exposure=cfg.min_exposure,
            max_exposure=cfg.max_exposure,
        )
        target = final.weights
        exposure = float(target.sum())
        turn = compute_turnover(current, target, 1 - current_exposure, 1 - exposure)
        action = (
            "reduce"
            if exposure < current_exposure - 1e-6
            else "restore"
            if exposure > current_exposure + 1e-6
            else "hold"
        )
        decision_rows.append(
            {
                **log,
                **overlay,
                "information_date": hist.index.max(),
                "state": state,
                "status": final.status,
                "solver": final.solver,
                "base_solver_status": base_status,
                "constraint_violation": final.max_constraint_violation,
                "objective": final.objective,
                "covariance_estimator": estimate.estimator,
                "composition_rebalance": composition_day,
                "forecast_vol": forecast_vol,
                "effective_forecast_vol": float(np.sqrt(252 * current @ covariance @ current)),
                "target_forecast_vol": float(np.sqrt(252 * target @ covariance @ target)),
                "target_exposure": exposure,
                "target_cash": 1 - exposure,
                "turnover": turn,
                "estimated_cost": turn * fee_rate,
                "action": action,
                "governed": abs(exposure - requested_exposure) > 1e-3,
                "mc_status": model.calibration_status if model is not None else "disabled",
            }
        )
        LOGGER.info(
            "Overlay %s %s exposure=%.4f -> %.4f requested=%.4f turnover=%.4f solver=%s",
            date.date(),
            action,
            current_exposure,
            exposure,
            requested_exposure,
            turn,
            final.status,
        )
        portfolios["target"] = pd.Series(target, index=tickers)
        if model is not None:
            day_risk.extend(
                summarize_portfolio(
                    model, simulations, portfolios["target"], date, "target", mc_cfg, cash_return
                )
            )
        risk_rows.extend(day_risk)
        for role, portfolio in portfolios.items():
            risk_weight_rows.extend(
                dict(
                    date=date,
                    portfolio_role=role,
                    ticker=ticker,
                    weight=float(weight),
                    in_calibration=ticker in eligible,
                )
                for ticker, weight in portfolio.items()
            )
        weight_rows.extend(
            dict(date=date, ticker=ticker, weight=float(weight))
            for ticker, weight in zip(tickers, target, strict=True)
        )
        pending[position + backtest_cfg.execution_lag_days] = target.copy()
        last_candidate = candidate.copy()
        previous_state, previous_eligible = state, eligible
    if not weight_rows:
        raise InsufficientHistory("No allocation decisions: inspect history, regime warmup and date range")
    result = StrategyResult(
        pd.DataFrame(weight_rows),
        pd.DataFrame(decision_rows),
        pd.DataFrame(risk_rows),
        pd.DataFrame(risk_weight_rows),
    )
    return result if return_details else result.weights


def input_fingerprint(returns: pd.DataFrame, regimes: pd.Series | None, end: pd.Timestamp) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(list(returns.columns)).encode())
    digest.update(pd.util.hash_pandas_object(returns.loc[:end], index=True).values.tobytes())
    if regimes is not None:
        digest.update(pd.util.hash_pandas_object(regimes.loc[:end], index=True).values.tobytes())
    return digest.hexdigest()


def run_optimize_pipeline(
    run_id: str | None = None, existing_data_behavior: str = "overwrite_or_ignore"
) -> str:
    base = load_base_config()
    cfg, mc_cfg = load_optimize_config(), load_mc_config()
    covariance_cfg, backtest_cfg = load_covariance_config(), load_backtest_config()
    run_id = ensure_run_id(run_id, prefix="weights")
    run_dir = base.data_dir / "runs" / run_id
    names = ("weights", "decisions", "mc", "risk_weights")
    legacy_run_exists = any(
        (partition / f"run_id={run_id}").exists()
        for partition in (base.parquet_dir / "weights").glob("year=*")
    )
    # Runs are immutable: avoid mixing configurations, stale MC, or old targets.
    if (
        legacy_run_exists
        or run_dir.exists()
        or any((base.parquet_dir / name / f"run_id={run_id}").exists() for name in names)
    ):
        raise FileExistsError(f"Run already exists; choose another --run-id: {run_id}")
    returns = build_returns_matrix(load_reference_price_dataset(base))
    regimes = regime_series(load_regimes_dataset(base.parquet_dir / "regimes")) if cfg.use_regimes else None
    result = optimize_over_time(
        returns,
        None,
        cfg,
        regimes=regimes,
        mc_cfg=mc_cfg,
        covariance_cfg=covariance_cfg,
        backtest_cfg=backtest_cfg,
        sectors={asset.ticker: asset.sector for asset in base.universe.assets},
        return_details=True,
    )
    end = result.weights.date.max()
    for name, frame in zip(
        names, (result.weights, result.decisions, result.risk, result.risk_weights), strict=True
    ):
        if not frame.empty:
            replace_partitioned_dataset(frame, base.parquet_dir / name / f"run_id={run_id}", ["year"])
    execution_config = asdict(resolve_execution_config(cfg, backtest_cfg))
    execution_config["end_date"] = min(
        pd.Timestamp(backtest_cfg.end_date) if backtest_cfg.end_date else returns.index.max(),
        returns.index.max(),
    ).isoformat()
    snapshot = dict(
        schema_version=2,
        optimize=asdict(cfg),
        mc=asdict(mc_cfg),
        covariance=asdict(covariance_cfg),
        backtest=execution_config,
        backtest_requested=asdict(backtest_cfg),
        universe_fingerprint=base.universe.fingerprint,
        reference_currency=base.reference_currency,
        decision_end=end.isoformat(),
        input_fingerprint=input_fingerprint(returns, regimes, end),
        input_start=returns.index.min().isoformat(),
        input_end=returns.index.max().isoformat(),
        decision_start=result.weights.date.min().isoformat(),
        strategy_provenance=collect_provenance(),
    )
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    LOGGER.info("Optimization run %s: %d decisions", run_id, result.weights.date.nunique())
    return run_id
