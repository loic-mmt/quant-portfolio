from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path
from typing import Sequence

from quant_portfolio.core.ids import ensure_run_id

LOGGER = logging.getLogger(__name__)
EXISTING_DATA_BEHAVIORS = ["overwrite_or_ignore", "overwrite", "error", "delete_matching"]


def _add_existing_data_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--existing-data-behavior",
        default="overwrite_or_ignore",
        choices=EXISTING_DATA_BEHAVIORS,
        help="How to handle existing Parquet partitions.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quant-portfolio",
        description="Regime-aware portfolio research pipeline.",
    )
    parser.add_argument("--log-level", default=None, help="Override the configured log level.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest", help="Download missing market prices.")

    for command, help_text in [
        ("features", "Build market and per-asset features."),
        ("regimes", "Estimate market regimes."),
        ("mc", "Compute regime-conditional Monte-Carlo summaries."),
    ]:
        command_parser = subparsers.add_parser(command, help=help_text)
        _add_existing_data_argument(command_parser)
        if command == "mc":
            command_parser.add_argument(
                "--run-id", required=True, help="Existing optimization run to re-evaluate."
            )

    optimize = subparsers.add_parser("optimize", help="Compute target portfolio weights.")
    optimize.add_argument("--run-id", default=None)
    _add_existing_data_argument(optimize)

    backtest = subparsers.add_parser("backtest", help="Run a temporally lagged portfolio simulation.")
    backtest.add_argument("--run-id", default=None, help="Weight run to simulate; latest when omitted.")
    backtest.add_argument("--plot", action="store_true")

    report = subparsers.add_parser("report", help="Build ablations and a standalone HTML report.")
    report.add_argument("--run-id", default=None)
    report.add_argument("--output", type=Path, default=None)

    run_all = subparsers.add_parser("run-all", help="Execute the complete research pipeline.")
    run_all.add_argument("--run-id", default=None)
    run_all.add_argument("--skip-ingest", action="store_true")
    run_all.add_argument("--with-report", action="store_true")
    _add_existing_data_argument(run_all)
    return parser


def _run_command(args: argparse.Namespace) -> None:
    if args.command == "ingest":
        from quant_portfolio.pipeline.ingest import run_ingest_pipeline

        run_ingest_pipeline()
    elif args.command == "features":
        from quant_portfolio.pipeline.features import run_features_pipeline

        run_features_pipeline(args.existing_data_behavior)
    elif args.command == "regimes":
        from quant_portfolio.pipeline.regimes import run_regime_pipeline

        run_regime_pipeline(args.existing_data_behavior)
    elif args.command == "mc":
        from quant_portfolio.pipeline.mc import run_mc_pipeline

        run_mc_pipeline(args.existing_data_behavior, run_id=args.run_id)
    elif args.command == "optimize":
        from quant_portfolio.pipeline.optimize import run_optimize_pipeline

        run_optimize_pipeline(args.run_id, args.existing_data_behavior)
    elif args.command == "backtest":
        from quant_portfolio.pipeline.backtest import run_backtest_pipeline

        run_backtest_pipeline(args.run_id, plot=args.plot)
    elif args.command == "report":
        from quant_portfolio.pipeline.report import run_report_pipeline

        run_report_pipeline(args.run_id, output_path=args.output)
    elif args.command == "run-all":
        _run_all(args)
    else:  # pragma: no cover - argparse enforces the command choices.
        raise ValueError(f"Unknown command: {args.command}")


def _run_all(args: argparse.Namespace) -> None:
    from quant_portfolio.pipeline.backtest import run_backtest_pipeline
    from quant_portfolio.pipeline.features import run_features_pipeline
    from quant_portfolio.pipeline.optimize import run_optimize_pipeline
    from quant_portfolio.pipeline.regimes import run_regime_pipeline

    if not args.skip_ingest:
        from quant_portfolio.pipeline.ingest import run_ingest_pipeline

        run_ingest_pipeline()
    run_features_pipeline(args.existing_data_behavior)
    run_regime_pipeline(args.existing_data_behavior)
    run_id = ensure_run_id(args.run_id, prefix="portfolio")
    run_optimize_pipeline(run_id, args.existing_data_behavior)
    run_backtest_pipeline(run_id)
    if args.with_report:
        from quant_portfolio.pipeline.report import run_report_pipeline

        run_report_pipeline(run_id)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        from quant_portfolio.core.db import initialize_metadata_db
        from quant_portfolio.core.settings import configure_logging, load_base_config

        base_config = load_base_config()
        configure_logging(args.log_level or base_config.log_level, verbose=args.verbose)
        base_config.data_dir.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(base_config.metadata_db) as connection:
            initialize_metadata_db(connection)
        _run_command(args)
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user")
        return 130
    except Exception:
        LOGGER.exception("Command failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
