"""Autonomous HTML report and persisted milestone-5 evaluation artifacts."""

from __future__ import annotations

import html
import json
import logging
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from quant_portfolio.core.ids import validate_run_id
from quant_portfolio.core.provenance import collect_provenance
from quant_portfolio.core.settings import PROJECT_ROOT, BaseConfig, load_base_config
from quant_portfolio.core.storage import (
    load_regimes_dataset,
    load_weights_dataset,
    replace_partitioned_dataset,
)
from quant_portfolio.models.covariance import CovarianceConfig
from quant_portfolio.pipeline.backtest import BacktestConfig, build_returns_matrix
from quant_portfolio.pipeline.data_quality import load_reference_price_dataset
from quant_portfolio.pipeline.evaluation import (
    VARIANT_LABELS,
    EvaluationBundle,
    bundle_artifacts,
    compute_metrics,
    enrich_trade_capacity,
    load_report_config,
    load_run_artifact,
    regime_attribution,
    run_ablation_suite,
)
from quant_portfolio.pipeline.mc import MCConfig, regime_series
from quant_portfolio.pipeline.optimize import OptimizeConfig, input_fingerprint

LOGGER = logging.getLogger(__name__)


def _resolve_run(config: BaseConfig, run_id: str | None) -> str:
    if run_id is not None:
        return validate_run_id(run_id)
    runs_dir = config.data_dir / "runs"
    candidates = sorted(path.parent.name for path in runs_dir.glob("*/config.json"))
    if not candidates:
        raise ValueError("No completed optimization run found")
    return validate_run_id(candidates[-1])


def _read_snapshot(config: BaseConfig, run_id: str) -> dict:
    path = config.data_dir / "runs" / run_id / "config.json"
    if not path.exists():
        raise ValueError(f"Run {run_id} has no immutable configuration snapshot")
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "optimize",
        "mc",
        "covariance",
        "backtest",
        "input_fingerprint",
        "universe_fingerprint",
        "decision_end",
    }
    missing = required - set(snapshot)
    if missing:
        raise ValueError(f"Run snapshot missing fields: {sorted(missing)}")
    return snapshot


def _full_bundle(config: BaseConfig, run_id: str, initial_capital: float) -> EvaluationBundle:
    results = load_run_artifact(config.parquet_dir / "backtests", run_id)
    if results["date"].duplicated().any():
        raise ValueError("Backtest artifact contains duplicate dates")
    results = results.set_index("date").sort_index()
    results.attrs["initial_capital"] = initial_capital
    positions = load_run_artifact(config.parquet_dir / "positions", run_id)
    try:
        trades = load_run_artifact(config.parquet_dir / "trades", run_id)
    except FileNotFoundError:
        trades = pd.DataFrame(
            columns=["decision_date", "execution_date", "ticker", "side", "delta_weight", "notional"]
        )
    return EvaluationBundle(results, trades, positions)


def _save_artifacts(
    config: BaseConfig,
    run_id: str,
    daily: pd.DataFrame,
    positions: pd.DataFrame,
    trades: pd.DataFrame,
    metrics: pd.DataFrame,
    attribution: pd.DataFrame,
) -> dict[str, str]:
    frames = {
        "evaluations": daily,
        "evaluation_positions": positions,
        "evaluation_metrics": metrics,
        "regime_attribution": attribution,
    }
    if not trades.empty:
        frames["evaluation_trades"] = trades
    paths: dict[str, str] = {}
    for name, frame in frames.items():
        destination = config.parquet_dir / name / f"run_id={run_id}"
        replace_partitioned_dataset(frame, destination, ["year"])
        paths[name] = str(destination.relative_to(config.data_dir))
    return paths


def _figure_svg(draw: Callable) -> str:
    import matplotlib

    matplotlib.use("Agg", force=True)
    matplotlib.rcParams["svg.hashsalt"] = "quant-portfolio-report"
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(11, 4.6))
    draw(figure, axis)
    figure.tight_layout()
    stream = StringIO()
    figure.savefig(stream, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(figure)
    svg = stream.getvalue()
    return svg[svg.find("<svg") :]


def _charts(
    daily: pd.DataFrame,
    regimes: pd.DataFrame,
    attribution: pd.DataFrame,
    evaluation_start: pd.Timestamp,
    initial_capital: float,
) -> dict[str, str]:
    evaluation = daily.loc[daily["date"] >= evaluation_start]

    def equity(_, axis):
        for variant, frame in evaluation.groupby("variant", sort=False):
            axis.plot(
                frame["date"], frame["portfolio_value"] / initial_capital, label=VARIANT_LABELS[variant]
            )
        axis.set(title="Equity curves — out-of-sample", ylabel="NAV / capital initial")
        axis.grid(alpha=0.25)
        axis.legend(ncol=2, fontsize=8)

    def drawdown(_, axis):
        for variant, frame in evaluation.groupby("variant", sort=False):
            values = frame["portfolio_value"].to_numpy(dtype=float)
            peaks = np.maximum.accumulate(np.r_[initial_capital, values])[1:]
            axis.plot(frame["date"], values / peaks - 1, label=VARIANT_LABELS[variant])
        axis.set(title="Drawdowns — out-of-sample", ylabel="Drawdown")
        axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        axis.grid(alpha=0.25)

    def exposure(_, axis):
        for variant, frame in evaluation.groupby("variant", sort=False):
            axis.plot(frame["date"], frame["gross_exposure"], label=VARIANT_LABELS[variant])
        axis.set(title="Risky exposure", ylabel="Exposure", ylim=(-0.02, 1.02))
        axis.grid(alpha=0.25)

    def regime_probabilities(_, axis):
        frame = regimes.loc[regimes["date"] >= daily["date"].min()].sort_values("date")
        colors = {"p_calm": "#2f9e73", "p_choppy": "#d69e2e", "p_stress": "#d1495b"}
        for column, color in colors.items():
            if column in frame:
                axis.plot(frame["date"], frame[column], label=column.removeprefix("p_"), color=color)
        axis.set(title="Filtered regime probabilities", ylabel="Probability", ylim=(0, 1))
        axis.grid(alpha=0.25)
        axis.legend(ncol=3)

    def regime_returns(_, axis):
        table = attribution.pivot(index="regime", columns="variant", values="total_return")
        table = table.reindex(columns=[name for name in VARIANT_LABELS if name in table])
        table.plot.bar(ax=axis, width=0.8)
        axis.set(title="Compounded return by confirmed regime", ylabel="Return", xlabel="Regime")
        axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        axis.legend([VARIANT_LABELS[name] for name in table.columns], fontsize=7, ncol=2)
        axis.grid(axis="y", alpha=0.25)

    return {
        "equity": _figure_svg(equity),
        "drawdown": _figure_svg(drawdown),
        "exposure": _figure_svg(exposure),
        "regimes": _figure_svg(regime_probabilities),
        "regime_returns": _figure_svg(regime_returns),
    }


PERCENT_COLUMNS = {
    "CAGR",
    "volatility",
    "max_drawdown",
    "realized_var_05",
    "realized_cvar_05",
    "downside_deviation",
    "extreme_loss_frequency",
    "turnover_annualized",
    "cost_fraction_initial_capital",
    "max_asset_weight",
    "cash_exposure_mean",
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "variance_share",
}


def _format_table(frame: pd.DataFrame, currency: str) -> str:
    display = frame.copy()
    for column in display:
        if column == "variant":
            display[column] = display[column].map(VARIANT_LABELS).fillna(display[column])
        elif column in PERCENT_COLUMNS:
            display[column] = display[column].map(
                lambda value: "—" if pd.isna(value) else f"{float(value):.2%}"
            )
        elif "capacity_approx" in column or "cost_total" in column:
            display[column] = display[column].map(
                lambda value: "—" if pd.isna(value) else f"{float(value):,.0f} {currency}"
            )
        elif pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "—" if pd.isna(value) else f"{float(value):.3f}"
            )
    return display.to_html(index=False, border=0, classes="data-table", escape=True)


def _conclusions(metrics: pd.DataFrame) -> list[str]:
    indexed = metrics.set_index("variant")
    best_return = str(indexed["CAGR"].astype(float).idxmax())
    lowest_drawdown = str(indexed["max_drawdown"].astype(float).idxmin())
    full = indexed.loc["strategy_full"]
    baseline = indexed.loc["equal_weight_buy_hold"]
    return [
        f"CAGR observé le plus élevé : {VARIANT_LABELS[best_return]} ({float(indexed.loc[best_return, 'CAGR']):.2%}).",
        f"Max drawdown observé le plus faible : {VARIANT_LABELS[lowest_drawdown]} ({float(indexed.loc[lowest_drawdown, 'max_drawdown']):.2%}).",
        f"Stratégie complète moins equal-weight buy-and-hold : CAGR {float(full['CAGR'] - baseline['CAGR']):+.2%}; différence de max drawdown {float(full['max_drawdown'] - baseline['max_drawdown']):+.2%}.",
        "Backtest descriptif : aucune causalité, aucune promesse future, aucun score de performance in-sample dans ce rapport.",
    ]


def _render_html(
    run_id: str,
    metrics: pd.DataFrame,
    attribution: pd.DataFrame,
    charts: dict[str, str],
    snapshot: dict,
    manifest: dict,
    reference_currency: str,
) -> str:
    metric_columns = [
        "variant",
        "CAGR",
        "volatility",
        "Sharpe",
        "Sortino",
        "max_drawdown",
        "recovery_days",
        "realized_var_05",
        "realized_cvar_05",
        "turnover_annualized",
        "cost_total_reference_currency",
        "hhi_risky_mean",
        "max_asset_weight",
        "cash_exposure_mean",
        "capacity_approx_reference_currency",
    ]
    regime_columns = [
        "variant",
        "regime",
        "days",
        "total_return",
        "annualized_volatility",
        "downside_deviation",
        "turnover_total",
        "cost_total_reference_currency",
        "exposure_mean",
        "variance_share",
    ]
    conclusions = "".join(f"<li>{html.escape(text)}</li>" for text in manifest["conclusions"])
    strategy_provenance = snapshot.get("strategy_provenance", {"status": "unavailable for pre-J5 run"})
    config_json = html.escape(json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=False))
    manifest_json = html.escape(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Quant portfolio — {html.escape(run_id)}</title>
<style>
:root{{--ink:#17202a;--muted:#5f6b76;--paper:#f5f3ee;--card:#fff;--accent:#245b78;--line:#d8d6cf}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1240px;margin:auto;padding:34px}} h1{{font-size:34px;margin-bottom:4px}} h2{{margin-top:38px;border-bottom:1px solid var(--line);padding-bottom:8px}}
.lede,.note{{color:var(--muted)}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:18px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px;overflow:auto}} svg{{max-width:100%;height:auto}}
.data-table{{border-collapse:collapse;width:100%;font-size:12px;background:var(--card)}} th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left}} th{{position:sticky;top:0;background:#edf1f3}}
code,pre{{font-family:ui-monospace,monospace}} pre{{white-space:pre-wrap;font-size:11px}} details{{margin:10px 0}} .tag{{display:inline-block;background:#e3edf2;padding:3px 8px;border-radius:20px;margin-right:6px}}
</style></head><body><main>
<h1>Regime-aware portfolio</h1><p class="lede">Run <code>{html.escape(run_id)}</code> · rapport autonome · devise {html.escape(reference_currency)}</p>
<p><span class="tag">Calibration/warmup: {html.escape(manifest["sample_periods"]["calibration_warmup"])}</span><span class="tag">Walk-forward out-of-sample: {html.escape(manifest["sample_periods"]["out_of_sample"])}</span></p>
<div class="card note"><strong>Séparation in/out-of-sample.</strong> Historique antérieur à première exécution sert calibration/warmup. Aucun score de performance in-sample. Tous tableaux comparatifs utilisent uniquement période walk-forward out-of-sample, mêmes dates, univers, rendements, frais et slippage.</div>
<h2>Conclusions descriptives</h2><div class="card"><ul>{conclusions}</ul></div>
<h2>Ablations</h2><div class="card">{_format_table(metrics[metric_columns], reference_currency)}</div>
<h2>Equity et drawdowns</h2><div class="grid"><div class="card">{charts["equity"]}</div><div class="card">{charts["drawdown"]}</div></div>
<h2>Exposition</h2><div class="card">{charts["exposure"]}</div>
<h2>Régimes</h2><div class="grid"><div class="card">{charts["regimes"]}</div><div class="card">{charts["regime_returns"]}</div></div>
<div class="card">{_format_table(attribution[regime_columns], reference_currency)}</div>
<h2>Risque réalisé et praticabilité</h2><p class="note">VaR/CVaR empiriques sur rendements journaliers nets. Capacité = quantile configuré des NAV maximales par trade sous participation ADV; approximation, pas garantie d'exécution.</p>
<div class="card">{_format_table(metrics[["variant", "realized_var_01", "realized_cvar_01", "realized_var_05", "realized_cvar_05", "downside_deviation", "extreme_loss_frequency", "turnover_total", "cost_fraction_initial_capital", "capacity_approx_reference_currency"]], reference_currency)}</div>
<h2>Provenance et recalcul</h2><p>Git stratégie: <code>{html.escape(str(strategy_provenance.get("git_revision")))}</code> · source: <code>{html.escape(str(strategy_provenance.get("source_fingerprint")))}</code> · dirty: <code>{html.escape(str(strategy_provenance.get("git_dirty")))}</code></p>
<p>Chaque mesure se recalcule depuis artefacts Parquet listés dans manifeste. HTML ne charge aucune ressource externe.</p>
<details><summary>Manifeste rapport</summary><pre>{manifest_json}</pre></details>
<details><summary>Configuration stratégie</summary><pre>{config_json}</pre></details>
</main></body></html>"""


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def run_report_pipeline(run_id: str | None = None, output_path: Path | None = None) -> Path:
    """Build all baselines/ablations, persist metrics, and write one standalone HTML."""
    config = load_base_config()
    effective_run_id = _resolve_run(config, run_id)
    snapshot = _read_snapshot(config, effective_run_id)
    report_config = load_report_config()
    optimize_config = OptimizeConfig(**snapshot["optimize"])
    mc_config = MCConfig(**snapshot["mc"])
    covariance_config = CovarianceConfig(**snapshot["covariance"])
    backtest_config = BacktestConfig(**snapshot["backtest"])

    prices = load_reference_price_dataset(config)
    simple_returns = build_returns_matrix(prices, return_type="simple")
    simulation_returns = build_returns_matrix(prices, return_type=backtest_config.return_type)
    frozen_end_raw = backtest_config.end_date or snapshot.get("input_end")
    if frozen_end_raw is not None:
        frozen_end = pd.Timestamp(frozen_end_raw)
        simple_returns = simple_returns.loc[simple_returns.index <= frozen_end]
        simulation_returns = simulation_returns.loc[simulation_returns.index <= frozen_end]
    regimes = load_regimes_dataset(config.parquet_dir / "regimes")
    states = regime_series(regimes) if optimize_config.use_regimes else None
    fingerprint = input_fingerprint(simple_returns, states, pd.Timestamp(snapshot["decision_end"]))
    if (
        fingerprint != snapshot["input_fingerprint"]
        or snapshot.get("universe_fingerprint") != config.universe.fingerprint
    ):
        raise ValueError("Report inputs differ from optimization run; create a new run")

    full_weights = load_weights_dataset(config.parquet_dir / "weights", effective_run_id)
    full = _full_bundle(config, effective_run_id, backtest_config.initial_capital)
    bundles, decision_date, evaluation_start = run_ablation_suite(
        simple_returns,
        simulation_returns,
        full_weights,
        full,
        regimes,
        optimize_config,
        mc_config,
        covariance_config,
        backtest_config,
        {asset.ticker: asset.sector for asset in config.universe.assets},
    )
    for bundle in bundles.values():
        bundle.trades = enrich_trade_capacity(bundle.trades, prices, report_config)
    daily, positions, trades = bundle_artifacts(bundles, evaluation_start)
    metrics = pd.DataFrame(
        [
            {
                "variant": variant,
                **compute_metrics(
                    bundle.results,
                    bundle.positions,
                    bundle.trades,
                    evaluation_start,
                    backtest_config,
                    report_config,
                ),
            }
            for variant, bundle in bundles.items()
        ]
    )
    attribution = regime_attribution(daily, regimes, evaluation_start, report_config.trading_days)
    evaluation_end = pd.Timestamp(daily["date"].max())
    for frame in (daily, positions, trades):
        if not frame.empty:
            frame["run_id"] = effective_run_id
    metrics["date"] = evaluation_end
    metrics["run_id"] = effective_run_id
    attribution["date"] = evaluation_end
    attribution["run_id"] = effective_run_id
    artifact_paths = _save_artifacts(config, effective_run_id, daily, positions, trades, metrics, attribution)

    conclusions = _conclusions(metrics)
    provenance = collect_provenance()
    manifest = {
        "schema_version": 1,
        "run_id": effective_run_id,
        "reference_currency": config.reference_currency,
        "input_dates": {
            "start": simple_returns.index.min().isoformat(),
            "end": simple_returns.index.max().isoformat(),
        },
        "sample_periods": {
            "calibration_warmup": f"{simple_returns.index.min().date()} → {decision_date.date()}",
            "out_of_sample": f"{evaluation_start.date()} → {evaluation_end.date()}",
        },
        "same_inputs_contract": {
            "dates": [evaluation_start.isoformat(), evaluation_end.isoformat()],
            "universe": list(simple_returns.columns),
            "transaction_bps": backtest_config.transaction_bps,
            "slippage_bps": backtest_config.slippage_bps,
            "return_type": backtest_config.return_type,
            "baseline_governors_disabled": True,
        },
        "variants": list(bundles),
        "variant_configuration": {
            "equal_weight_buy_hold": {
                "allocation": "equal_weight_once",
                "turnover_cap": None,
                "exposure_change_cap": None,
            },
            "equal_weight_rebalanced": {
                "allocation": "equal_weight",
                "rebal_freq": optimize_config.rebal_freq,
                "turnover_cap": None,
                "exposure_change_cap": None,
            },
            "minimum_variance_no_regimes": {
                "use_regimes": False,
                "use_mc": False,
                "use_overlay": False,
            },
            "strategy_no_overlay": {"use_overlay": False},
            "strategy_no_mc": {"use_mc": False},
            "strategy_full": asdict(optimize_config),
        },
        "report_config": asdict(report_config),
        "strategy_provenance": snapshot.get("strategy_provenance"),
        "report_provenance": provenance,
        "artifacts": artifact_paths,
        "conclusions": conclusions,
    }
    charts = _charts(daily, regimes, attribution, evaluation_start, backtest_config.initial_capital)
    target = output_path or PROJECT_ROOT / "artifacts/reports" / f"{effective_run_id}.html"
    if target.suffix.lower() not in {".html", ".htm"}:
        raise ValueError("Report output must use .html or .htm")
    rendered = _render_html(
        effective_run_id,
        metrics,
        attribution,
        charts,
        snapshot,
        manifest,
        config.reference_currency,
    )
    _atomic_text(target, rendered)
    manifest["html_report"] = str(target)
    _atomic_text(
        config.data_dir / "runs" / effective_run_id / "report.json",
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    LOGGER.info("Report %s: %s", effective_run_id, target)
    return target
