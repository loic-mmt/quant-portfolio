import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from quant_portfolio.core.settings import load_base_config
from quant_portfolio.core.storage import (
    load_weights_dataset,
    read_parquet_dataset,
    replace_partitioned_dataset,
    write_weights_dataset,
)
from quant_portfolio.core.universe import AssetDefinition
from quant_portfolio.main import _run_all, build_parser
from quant_portfolio.models.covariance import CovarianceConfig
from quant_portfolio.pipeline.backtest import DEFAULT_CFG as BACKTEST_DEFAULT
from quant_portfolio.pipeline.backtest import run_backtest_pipeline
from quant_portfolio.pipeline.mc import MCConfig, run_mc_pipeline
from quant_portfolio.pipeline.optimize import OptimizeConfig, run_optimize_pipeline


class RiskPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = load_base_config()
        self.base = replace(
            base,
            data_dir=Path(self.temp.name),
            universe=replace(
                base.universe,
                assets=tuple(AssetDefinition(ticker, "EUR", sector="equity") for ticker in "ABCD"),
                fx_rates=(),
            ),
        )
        dates = pd.bdate_range("2024-01-01", periods=24)
        rng = np.random.default_rng(33)
        levels = 100 * np.cumprod(1 + rng.normal(0, 0.01, (24, 4)), axis=0)
        prices = pd.DataFrame(levels, index=dates, columns=list("ABCD")).rename_axis("date").reset_index()
        self.prices = prices.melt(id_vars="date", var_name="ticker", value_name="adj_close")
        regimes = pd.DataFrame({"date": dates, "state": np.where(np.arange(24) > 15, 1, 0)})
        replace_partitioned_dataset(regimes, self.base.parquet_dir / "regimes", ["year"])
        self.cfg = OptimizeConfig(max_weight=0.8, lookback=10, target_vol=0.12, sector_caps={"equity": 1.0})
        self.mc = MCConfig(n_sims=100, window=15, min_observations=5)
        self.covariance = CovarianceConfig("shrink_diag", 0.2, 5, 1e-8)
        self.backtest = replace(
            BACKTEST_DEFAULT, cash_rate_annual=0.03, execution_lag_days=2, turnover_cap=0.08
        )
        self.patch("quant_portfolio.pipeline.optimize.load_base_config", return_value=self.base)
        self.patch("quant_portfolio.pipeline.backtest.load_base_config", return_value=self.base)
        self.patch("quant_portfolio.pipeline.mc.load_base_config", return_value=self.base)
        self.patch("quant_portfolio.pipeline.optimize.load_optimize_config", return_value=self.cfg)
        self.patch("quant_portfolio.pipeline.optimize.load_mc_config", return_value=self.mc)
        self.patch("quant_portfolio.pipeline.mc.load_mc_config", return_value=self.mc)
        self.patch("quant_portfolio.pipeline.optimize.load_covariance_config", return_value=self.covariance)
        self.patch("quant_portfolio.pipeline.optimize.load_backtest_config", return_value=self.backtest)
        self.patch("quant_portfolio.pipeline.optimize.load_reference_price_dataset", return_value=self.prices)
        self.patch("quant_portfolio.pipeline.backtest.load_reference_price_dataset", return_value=self.prices)
        self.patch(
            "quant_portfolio.pipeline.data_quality.load_reference_price_dataset", return_value=self.prices
        )

    def patch(self, *args, **kwargs):
        patcher = patch(*args, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def test_optimize_backtest_and_mc_replay_share_run_and_effective_weights(self):
        run_id = run_optimize_pipeline("fixture-full")
        weights = load_weights_dataset(self.base.parquet_dir / "weights", run_id)
        self.assertEqual(weights.attrs["run_id"], run_id)
        # Backtest must use the saved execution/cash/fee configuration.
        self.patch(
            "quant_portfolio.pipeline.backtest.load_backtest_config",
            side_effect=AssertionError("live config used"),
        )
        results, _ = run_backtest_pipeline(run_id)
        risk = read_parquet_dataset(self.base.parquet_dir / "mc" / f"run_id={run_id}")
        effective = risk[risk.portfolio_role == "effective"].groupby("date").exposure.first()
        np.testing.assert_allclose(effective, results.loc[effective.index, "gross_exposure"], atol=1e-12)
        run_mc_pipeline(run_id=run_id)
        replay = read_parquet_dataset(self.base.parquet_dir / "mc_replay" / f"run_id={run_id}")
        keys = ["date", "portfolio_role", "horizon", "distribution"]
        assert_frame_equal(
            risk.sort_values(keys).reset_index(drop=True), replay.sort_values(keys).reset_index(drop=True)
        )
        for dataset in ("decisions", "risk_weights", "backtests", "trades", "positions"):
            self.assertTrue((self.base.parquet_dir / dataset).exists())
        self.assertTrue(self.base.metadata_db.exists())
        with self.assertRaises(FileExistsError):
            run_optimize_pipeline(run_id)

    def test_replay_rejects_changed_history_and_missing_run(self):
        run_id = run_optimize_pipeline("fixture-history")
        self.prices.loc[0, "adj_close"] *= 1.05
        with self.assertRaisesRegex(ValueError, "inputs differ"):
            run_mc_pipeline(run_id=run_id)
        with self.assertRaisesRegex(ValueError, "inputs differ"):
            run_backtest_pipeline(run_id)
        with self.assertRaisesRegex(ValueError, "requires --run-id"):
            run_mc_pipeline()

    def test_cli_mc_requires_run_id(self):
        args = build_parser().parse_args(["mc", "--run-id", "experiment-001"])
        self.assertEqual(args.run_id, "experiment-001")

    def test_legacy_weight_run_id_cannot_be_reused(self):
        weights = pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "ticker": ["A"], "weight": [0.5]})
        write_weights_dataset(weights, self.base.parquet_dir / "weights", ["year", "run_id"], run_id="legacy")
        with self.assertRaises(FileExistsError):
            run_optimize_pipeline("legacy")

    def test_run_all_uses_inline_mc_without_preliminary_mc_stage(self):
        order = []
        self.patch(
            "quant_portfolio.pipeline.features.run_features_pipeline",
            side_effect=lambda *a: order.append("features"),
        )
        self.patch(
            "quant_portfolio.pipeline.regimes.run_regime_pipeline",
            side_effect=lambda *a: order.append("regimes"),
        )
        self.patch(
            "quant_portfolio.pipeline.optimize.run_optimize_pipeline",
            side_effect=lambda *a: order.append("optimize"),
        )
        self.patch(
            "quant_portfolio.pipeline.backtest.run_backtest_pipeline",
            side_effect=lambda *a: order.append("backtest"),
        )
        self.patch(
            "quant_portfolio.pipeline.mc.run_mc_pipeline", side_effect=AssertionError("preliminary MC called")
        )
        _run_all(build_parser().parse_args(["run-all", "--run-id", "order", "--skip-ingest"]))
        self.assertEqual(order, ["features", "regimes", "optimize", "backtest"])


if __name__ == "__main__":
    unittest.main()
