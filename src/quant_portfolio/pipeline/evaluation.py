"""Reproducible baselines, ablations, realized metrics, and regime attribution."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from quant_portfolio.core.ids import validate_run_id
from quant_portfolio.core.settings import PROJECT_ROOT, load_yaml_mapping
from quant_portfolio.core.storage import read_parquet_dataset
from quant_portfolio.models.covariance import CovarianceConfig
from quant_portfolio.pipeline.backtest import BacktestConfig, simulate_portfolio
from quant_portfolio.pipeline.mc import MCConfig
from quant_portfolio.pipeline.optimize import OptimizeConfig, build_rebalance_dates, optimize_over_time

REPORT_CONFIG_PATH = PROJECT_ROOT / "config/report.yaml"
VARIANT_LABELS = {
    "equal_weight_buy_hold": "Equal-weight buy-and-hold",
    "equal_weight_rebalanced": "Equal-weight rebalanced",
    "minimum_variance_no_regimes": "Minimum variance — no regimes",
    "strategy_no_overlay": "Strategy — no overlay",
    "strategy_no_mc": "Strategy — no Monte-Carlo",
    "strategy_full": "Strategy — full",
}
VARIANT_ORDER = tuple(VARIANT_LABELS)


@dataclass(frozen=True)
class ReportConfig:
    trading_days: int = 252
    extreme_loss_threshold: float = 0.02
    capacity_adv_window: int = 20
    capacity_participation_rate: float = 0.10
    capacity_quantile: float = 0.05

    def __post_init__(self) -> None:
        if type(self.trading_days) is not int or self.trading_days < 2:
            raise ValueError("trading_days must be an integer >= 2")
        if type(self.capacity_adv_window) is not int or self.capacity_adv_window < 1:
            raise ValueError("capacity_adv_window must be a positive integer")
        for name in ("extreme_loss_threshold", "capacity_participation_rate", "capacity_quantile"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must be finite and in (0, 1)")


def load_report_config(path: Path | None = None) -> ReportConfig:
    data = load_yaml_mapping(path or REPORT_CONFIG_PATH)
    unknown = set(data) - {field.name for field in fields(ReportConfig)}
    if unknown:
        raise ValueError(f"Unknown report fields: {sorted(unknown)}")
    return ReportConfig(**data)


@dataclass
class EvaluationBundle:
    results: pd.DataFrame
    trades: pd.DataFrame
    positions: pd.DataFrame


def load_run_artifact(base_dir: Path, run_id: str) -> pd.DataFrame:
    """Load either run-scoped or year/run_id Hive layouts without mixing runs."""
    run_id = validate_run_id(run_id)
    scoped = base_dir / f"run_id={run_id}"
    legacy = sorted(base_dir.glob(f"year=*/run_id={run_id}"))
    if scoped.exists() and legacy:
        raise ValueError(f"Ambiguous artifact layouts for {base_dir.name}/{run_id}")
    if scoped.exists():
        frame = read_parquet_dataset(scoped)
    elif legacy:
        frame = pd.concat((pd.read_parquet(path) for path in legacy), ignore_index=True)
    else:
        raise FileNotFoundError(f"Artifact not found: {base_dir.name}/{run_id}")
    for column in ("date", "decision_date", "execution_date", "calibration_date"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def evaluation_boundary(full_results: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return first full-strategy decision and its strictly later execution."""
    required = {"is_rebalance", "decision_date"}
    missing = required - set(full_results)
    if missing:
        raise KeyError(f"Full backtest missing evaluation boundary columns: {sorted(missing)}")
    executed = full_results.loc[full_results["is_rebalance"].astype(bool)].copy()
    executed["decision_date"] = pd.to_datetime(executed["decision_date"], errors="coerce")
    executed = executed.dropna(subset=["decision_date"])
    if executed.empty:
        raise ValueError("Full strategy has no executed out-of-sample decision")
    execution_date = pd.Timestamp(executed.index.min())
    decision_date = pd.Timestamp(executed.loc[execution_date, "decision_date"])
    if decision_date >= execution_date:
        raise ValueError("Evaluation decision must precede execution")
    return decision_date, execution_date


def equal_weight_targets(tickers: list[str], decision_dates: pd.DatetimeIndex) -> pd.DataFrame:
    if not tickers or len(set(tickers)) != len(tickers):
        raise ValueError("Equal-weight baseline requires a unique nonempty universe")
    dates = pd.DatetimeIndex(decision_dates).sort_values().unique()
    if dates.empty:
        raise ValueError("Equal-weight baseline requires decision dates")
    weight = 1.0 / len(tickers)
    return pd.DataFrame(
        [{"date": date, "ticker": ticker, "weight": weight} for date in dates for ticker in tickers]
    )


def _simulate(returns: pd.DataFrame, targets: pd.DataFrame, config: BacktestConfig) -> EvaluationBundle:
    results, trades, positions = simulate_portfolio(returns, targets, config, return_details=True)
    return EvaluationBundle(results, trades, positions)


def run_ablation_suite(
    simple_returns: pd.DataFrame,
    simulation_returns: pd.DataFrame,
    full_weights: pd.DataFrame,
    full_bundle: EvaluationBundle,
    regimes: pd.DataFrame | pd.Series,
    optimize_config: OptimizeConfig,
    mc_config: MCConfig,
    covariance_config: CovarianceConfig,
    backtest_config: BacktestConfig,
    sectors: Mapping[str, str | None],
) -> tuple[dict[str, EvaluationBundle], pd.Timestamp, pd.Timestamp]:
    """Build six variants on identical dates, universe, return data, and fees."""
    if not simple_returns.index.equals(simulation_returns.index) or list(simple_returns) != list(
        simulation_returns
    ):
        raise ValueError("Optimizer and simulation return matrices must align")
    full_results = full_bundle.results.sort_index()
    decision_date, evaluation_start = evaluation_boundary(full_results)
    tickers = list(simple_returns.columns)
    weight_tickers = set(full_weights["ticker"].astype(str))
    if weight_tickers != set(tickers):
        raise ValueError("Full strategy and ablations must use exactly the same universe")

    baseline_config = replace(backtest_config, turnover_cap=None, exposure_change_cap=None)
    hold = equal_weight_targets(tickers, pd.DatetimeIndex([decision_date]))
    scheduled = build_rebalance_dates(simple_returns.index, optimize_config.rebal_freq)
    scheduled = scheduled[scheduled > decision_date].insert(0, decision_date)
    rebalanced = equal_weight_targets(tickers, scheduled)
    bundles: dict[str, EvaluationBundle] = {
        "equal_weight_buy_hold": _simulate(simulation_returns, hold, baseline_config),
        "equal_weight_rebalanced": _simulate(simulation_returns, rebalanced, baseline_config),
    }

    strategy_config = replace(backtest_config, start_date=decision_date.isoformat())
    variants = {
        "minimum_variance_no_regimes": replace(
            optimize_config, use_regimes=False, use_mc=False, use_overlay=False
        ),
        "strategy_no_overlay": replace(optimize_config, use_overlay=False),
        "strategy_no_mc": replace(optimize_config, use_mc=False),
    }
    for name, config in variants.items():
        targets = optimize_over_time(
            simple_returns,
            None,
            config,
            regimes=regimes if config.use_regimes else None,
            mc_cfg=mc_config,
            covariance_cfg=covariance_config,
            backtest_cfg=strategy_config,
            sectors=dict(sectors),
        )
        targets = targets.loc[pd.to_datetime(targets["date"]) >= decision_date]
        bundles[name] = _simulate(simulation_returns, targets, backtest_config)
    bundles["strategy_full"] = full_bundle

    reference_index = full_results.index
    reference_tickers = set(tickers) | {"__CASH__"}
    for name, bundle in bundles.items():
        if not bundle.results.index.equals(reference_index):
            raise ValueError(f"Ablation dates differ for {name}")
        if set(bundle.positions["ticker"]) != reference_tickers:
            raise ValueError(f"Ablation universe differs for {name}")
    return {name: bundles[name] for name in VARIANT_ORDER}, decision_date, evaluation_start


def enrich_trade_capacity(trades: pd.DataFrame, prices: pd.DataFrame, config: ReportConfig) -> pd.DataFrame:
    """Estimate NAV capacity from a configurable share of trailing ADV."""
    output = trades.copy()
    if output.empty:
        return pd.DataFrame(
            columns=[
                *output.columns,
                "date",
                "adv_reference_currency",
                "adv_observations",
                "capacity_limit_reference_currency",
            ]
        )
    output["date"] = pd.to_datetime(output["execution_date"], errors="coerce")
    required = {"date", "ticker", "adj_close", "volume"}
    if not required.issubset(prices):
        output["adv_reference_currency"] = np.nan
        output["adv_observations"] = 0
        output["capacity_limit_reference_currency"] = np.nan
        return output
    liquidity = prices[list(required)].copy()
    liquidity["date"] = pd.to_datetime(liquidity["date"], errors="coerce")
    liquidity["dollar_volume"] = pd.to_numeric(liquidity["adj_close"], errors="coerce") * pd.to_numeric(
        liquidity["volume"], errors="coerce"
    )
    liquidity = liquidity.sort_values(["ticker", "date"])
    groups = liquidity.groupby("ticker", sort=False)["dollar_volume"]
    liquidity["adv_reference_currency"] = groups.transform(
        lambda series: series.rolling(config.capacity_adv_window, min_periods=1).mean()
    )
    liquidity["adv_observations"] = groups.transform(
        lambda series: series.rolling(config.capacity_adv_window, min_periods=1).count()
    ).astype(int)
    output = output.merge(
        liquidity[["date", "ticker", "adv_reference_currency", "adv_observations"]],
        on=["date", "ticker"],
        how="left",
        validate="many_to_one",
    )
    risky = output["ticker"].ne("__CASH__") & output["delta_weight"].abs().gt(1e-14)
    output["capacity_limit_reference_currency"] = np.nan
    output.loc[risky, "capacity_limit_reference_currency"] = (
        config.capacity_participation_rate
        * output.loc[risky, "adv_reference_currency"]
        / output.loc[risky, "delta_weight"].abs()
    )
    return output


def _tail_loss(returns: pd.Series, alpha: float) -> tuple[float, float]:
    quantile = float(returns.quantile(alpha))
    tail = returns.loc[returns <= quantile]
    return max(0.0, -quantile), max(0.0, -float(tail.mean()))


def _drawdown_metrics(values: pd.Series, initial_capital: float) -> dict[str, object]:
    values = values.astype(float)
    running_peak = values.cummax().clip(lower=initial_capital)
    drawdown = values / running_peak - 1.0
    trough_date = pd.Timestamp(drawdown.idxmin())
    max_drawdown = abs(float(drawdown.loc[trough_date]))
    before = values.loc[:trough_date]
    peak_value = max(initial_capital, float(before.max()))
    if peak_value == initial_capital and (before < initial_capital).all():
        peak_date = pd.Timestamp(values.index[0])
    else:
        peak_date = pd.Timestamp(before.loc[before >= peak_value - 1e-12].index[-1])
    recovered = values.loc[values.index > trough_date]
    recovered = recovered.loc[recovered >= peak_value - 1e-12]
    recovery_date = pd.Timestamp(recovered.index[0]) if not recovered.empty else None
    recovery_days = (recovery_date - peak_date).days if recovery_date is not None else np.nan
    return {
        "max_drawdown": max_drawdown,
        "drawdown_peak_date": peak_date,
        "drawdown_trough_date": trough_date,
        "recovery_date": recovery_date,
        "recovery_days": recovery_days,
        "recovered": recovery_date is not None,
    }


def compute_metrics(
    results: pd.DataFrame,
    positions: pd.DataFrame,
    trades: pd.DataFrame,
    evaluation_start: pd.Timestamp,
    backtest_config: BacktestConfig,
    report_config: ReportConfig,
) -> dict[str, object]:
    """Compute every headline metric from persisted evaluation artifacts."""
    period = results.loc[results.index >= evaluation_start].copy()
    if period.empty:
        raise ValueError("No out-of-sample evaluation observations")
    daily = period["portfolio_return"].astype(float)
    days = max(1, int((period.index[-1] - period.index[0]).days) + 1)
    years = days / 365.25
    initial = float(backtest_config.initial_capital)
    final = float(period["portfolio_value"].iloc[-1])
    cagr = (final / initial) ** (1 / years) - 1 if initial > 0 else np.nan
    volatility = float(daily.std(ddof=1) * np.sqrt(report_config.trading_days))
    daily_rf = (1 + backtest_config.cash_rate_annual) ** (1 / report_config.trading_days) - 1
    excess = daily - daily_rf
    excess_std = float(excess.std(ddof=1))
    sharpe = (
        float(excess.mean() / excess_std * np.sqrt(report_config.trading_days)) if excess_std > 0 else np.nan
    )
    downside = np.minimum(excess.to_numpy(), 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside**2)) * np.sqrt(report_config.trading_days))
    sortino = (
        float(excess.mean() * report_config.trading_days / downside_deviation)
        if downside_deviation > 0
        else np.nan
    )
    var_01, cvar_01 = _tail_loss(daily, 0.01)
    var_05, cvar_05 = _tail_loss(daily, 0.05)
    drawdown = _drawdown_metrics(period["portfolio_value"], initial)

    period_positions = positions.loc[pd.to_datetime(positions["date"]) >= evaluation_start].copy()
    risky = period_positions.loc[~period_positions["is_cash"].astype(bool)]
    risky_pivot = risky.pivot(index="date", columns="ticker", values="weight").fillna(0.0)
    exposure = risky_pivot.sum(axis=1)
    hhi_risky = risky_pivot.pow(2).sum(axis=1).div(exposure.pow(2)).where(exposure > 1e-14)
    all_pivot = period_positions.pivot(index="date", columns="ticker", values="weight").fillna(0.0)
    capacity = pd.to_numeric(
        trades.get("capacity_limit_reference_currency", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    capacity_value = (
        float(capacity.quantile(report_config.capacity_quantile)) if not capacity.empty else np.nan
    )
    turnover = period["turnover"].astype(float)
    return {
        "date_start": pd.Timestamp(period.index[0]),
        "date_end": pd.Timestamp(period.index[-1]),
        "observations": int(len(period)),
        "CAGR": float(cagr),
        "volatility": volatility,
        "Sharpe": sharpe,
        "Sortino": sortino,
        **drawdown,
        "realized_var_01": var_01,
        "realized_cvar_01": cvar_01,
        "realized_var_05": var_05,
        "realized_cvar_05": cvar_05,
        "downside_deviation": downside_deviation,
        "extreme_loss_frequency": float((daily < -report_config.extreme_loss_threshold).mean()),
        "extreme_loss_threshold": report_config.extreme_loss_threshold,
        "turnover_total": float(turnover.sum()),
        "turnover_annualized": float(turnover.sum() / years),
        "cost_total_reference_currency": float(period["cost_amount"].sum()),
        "cost_fraction_initial_capital": float(period["cost_amount"].sum() / initial),
        "capacity_approx_reference_currency": capacity_value,
        "capacity_participation_rate": report_config.capacity_participation_rate,
        "capacity_quantile": report_config.capacity_quantile,
        "hhi_risky_mean": float(hhi_risky.mean()),
        "hhi_risky_max": float(hhi_risky.max()),
        "hhi_total_mean": float(all_pivot.pow(2).sum(axis=1).mean()),
        "max_asset_weight": float(risky_pivot.max(axis=1).max()),
        "cash_exposure_mean": float(1.0 - exposure.mean()),
        "cash_exposure_max": float(1.0 - exposure.min()),
    }


def regime_attribution(
    daily_artifacts: pd.DataFrame,
    regimes: pd.DataFrame,
    evaluation_start: pd.Timestamp,
    trading_days: int = 252,
) -> pd.DataFrame:
    """Attribute additive log return and squared-deviation risk to confirmed regimes."""
    required = {"date", "regime"}
    if not required.issubset(regimes):
        raise KeyError(f"Regime attribution requires {sorted(required)}")
    labels = regimes[["date", "regime"]].dropna().sort_values("date").drop_duplicates("date", keep="last")
    rows: list[dict[str, object]] = []
    for variant, frame in daily_artifacts.groupby("variant", sort=False):
        frame = frame.loc[frame["date"] >= evaluation_start].sort_values("date")
        aligned = pd.merge_asof(frame, labels, on="date", direction="backward").dropna(subset=["regime"])
        full_mean = float(aligned["portfolio_return"].mean())
        total_squared = float(((aligned["portfolio_return"] - full_mean) ** 2).sum())
        total_log = float(np.log1p(aligned["portfolio_return"]).sum())
        for regime, group in aligned.groupby("regime", sort=True):
            returns = group["portfolio_return"].astype(float)
            volatility = float(returns.std(ddof=1) * np.sqrt(trading_days)) if len(group) > 1 else np.nan
            rows.append(
                {
                    "variant": variant,
                    "regime": str(regime),
                    "days": int(len(group)),
                    "total_return": float((1 + returns).prod() - 1),
                    "annualized_return": float((1 + returns).prod() ** (trading_days / len(group)) - 1),
                    "annualized_volatility": volatility,
                    "Sharpe_zero_rate": float(returns.mean() / returns.std(ddof=1) * np.sqrt(trading_days))
                    if len(group) > 1 and returns.std(ddof=1) > 0
                    else np.nan,
                    "downside_deviation": float(
                        np.sqrt(np.mean(np.minimum(returns, 0) ** 2)) * np.sqrt(trading_days)
                    ),
                    "turnover_total": float(group["turnover"].sum()),
                    "turnover_mean": float(group["turnover"].mean()),
                    "cost_total_reference_currency": float(group["cost_amount"].sum()),
                    "exposure_mean": float(group["gross_exposure"].mean()),
                    "log_return_contribution": float(np.log1p(returns).sum()),
                    "log_return_share": float(np.log1p(returns).sum() / total_log)
                    if abs(total_log) > 1e-14
                    else np.nan,
                    "variance_share": float(((returns - full_mean) ** 2).sum() / total_squared)
                    if total_squared > 0
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def bundle_artifacts(
    bundles: Mapping[str, EvaluationBundle], evaluation_start: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_rows, position_rows, trade_rows = [], [], []
    for variant, bundle in bundles.items():
        daily = bundle.results.rename_axis("date").reset_index()
        daily["variant"] = variant
        daily["sample"] = np.where(daily["date"] < evaluation_start, "calibration_warmup", "out_of_sample")
        daily_rows.append(daily)
        positions = bundle.positions.copy()
        positions["variant"] = variant
        positions["sample"] = np.where(
            pd.to_datetime(positions["date"]) < evaluation_start, "calibration_warmup", "out_of_sample"
        )
        position_rows.append(positions)
        trades = bundle.trades.copy()
        if not trades.empty:
            trades["variant"] = variant
            trade_rows.append(trades)
    daily = pd.concat(daily_rows, ignore_index=True)
    positions = pd.concat(position_rows, ignore_index=True)
    trades = pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame()
    return daily, positions, trades
