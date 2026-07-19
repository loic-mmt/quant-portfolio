from __future__ import annotations

from pathlib import Path
import sqlite3

from quant_portfolio.core.ids import validate_run_id
from quant_portfolio.core.settings import PROJECT_ROOT, load_base_config


def _load_summary(conn: sqlite3.Connection, run_id: str | None) -> tuple[str, sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    if run_id is None:
        row = conn.execute("SELECT * FROM backtests ORDER BY rowid DESC LIMIT 1").fetchone()
    else:
        run_id = validate_run_id(run_id)
        row = conn.execute("SELECT * FROM backtests WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise ValueError("No backtest summary found for the requested run")
    return str(row["run_id"]), row


def run_report_pipeline(run_id: str | None = None, output_path: Path | None = None) -> Path:
    """Write a small auditable Markdown summary for an existing backtest run.

    The full graphical/ablation report remains part of roadmap milestone 5.
    """
    config = load_base_config()
    with sqlite3.connect(config.metadata_db) as conn:
        effective_run_id, row = _load_summary(conn, run_id)

    target = output_path or PROJECT_ROOT / "artifacts/reports" / f"{effective_run_id}.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("CAGR", row["CAGR"]),
        ("Volatilité annualisée", row["volatility"]),
        ("Sharpe", row["Sharpe"]),
        ("Max drawdown", row["max_drawdown"]),
        ("Turnover annualisé", row["turnover_annualised"]),
    ]
    lines = [
        f"# Backtest `{effective_run_id}`",
        "",
        f"- Période : `{row['date_debut']}` → `{row['date_fin']}`",
        "- Statut : résumé technique ; le rapport d'ablation complet relève du jalon 5.",
        "",
        "## Métriques",
        "",
        "| Mesure | Valeur |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {float(value):.6f} |" for name, value in metrics)
    lines.append("")
    target.write_text("\n".join(lines), encoding="utf-8")
    return target
