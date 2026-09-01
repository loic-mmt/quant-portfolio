import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from quant_portfolio.pipeline.backtest import DEFAULT_CFG
from quant_portfolio.pipeline.evaluation import (
    ReportConfig,
    compute_metrics,
    enrich_trade_capacity,
    equal_weight_targets,
    load_run_artifact,
    regime_attribution,
)


class EvaluationTests(unittest.TestCase):
    def test_equal_weight_targets_are_exact(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-09"])
        targets = equal_weight_targets(["A", "B", "C"], dates)
        self.assertEqual(len(targets), 6)
        np.testing.assert_allclose(targets.groupby("date").weight.sum(), 1.0)
        self.assertTrue((targets.weight == 1 / 3).all())

    def test_capacity_uses_reference_currency_adv_and_trade_weight(self):
        dates = pd.bdate_range("2024-01-01", periods=3)
        prices = pd.DataFrame(
            {"date": dates, "ticker": ["A"] * 3, "adj_close": [100.0] * 3, "volume": [1000] * 3}
        )
        trades = pd.DataFrame(
            {
                "execution_date": [dates[-1], dates[-1]],
                "ticker": ["A", "__CASH__"],
                "delta_weight": [0.2, -0.2],
            }
        )
        enriched = enrich_trade_capacity(
            trades, prices, ReportConfig(capacity_adv_window=2, capacity_participation_rate=0.1)
        )
        risky = enriched.loc[enriched.ticker == "A"].iloc[0]
        self.assertAlmostEqual(risky.adv_reference_currency, 100_000)
        self.assertAlmostEqual(risky.capacity_limit_reference_currency, 50_000)
        self.assertTrue(
            enriched.loc[enriched.ticker == "__CASH__", "capacity_limit_reference_currency"].isna().all()
        )

    def test_metrics_cover_recovery_tail_costs_and_concentration(self):
        dates = pd.date_range("2024-01-01", periods=5)
        daily_returns = pd.Series([0.10, -0.20, 0.25, -0.10, 0.05], index=dates)
        values = 100_000 * (1 + daily_returns).cumprod()
        results = pd.DataFrame(
            {
                "portfolio_value": values,
                "portfolio_return": daily_returns,
                "turnover": [1.0, 0, 0, 0.1, 0],
                "cost_amount": [100.0, 0, 0, 10.0, 0],
            },
            index=dates,
        )
        positions = pd.DataFrame(
            [
                {"date": date, "ticker": ticker, "weight": weight, "is_cash": ticker == "__CASH__"}
                for date in dates
                for ticker, weight in (("A", 0.4), ("B", 0.4), ("__CASH__", 0.2))
            ]
        )
        trades = pd.DataFrame({"capacity_limit_reference_currency": [1_000_000.0, 2_000_000.0]})
        metrics = compute_metrics(
            results,
            positions,
            trades,
            dates[0],
            replace(DEFAULT_CFG, initial_capital=100_000),
            ReportConfig(capacity_quantile=0.5),
        )
        self.assertAlmostEqual(metrics["max_drawdown"], 0.20)
        self.assertEqual(metrics["recovery_days"], 2)
        self.assertTrue(metrics["recovered"])
        self.assertGreaterEqual(metrics["realized_cvar_05"], metrics["realized_var_05"])
        self.assertGreater(metrics["downside_deviation"], 0)
        self.assertAlmostEqual(metrics["hhi_risky_mean"], 0.5)
        self.assertAlmostEqual(metrics["max_asset_weight"], 0.4)
        self.assertAlmostEqual(metrics["cash_exposure_mean"], 0.2)
        self.assertAlmostEqual(metrics["cost_total_reference_currency"], 110)
        self.assertAlmostEqual(metrics["capacity_approx_reference_currency"], 1_500_000)

    def test_regime_attribution_is_additive_in_log_return_and_risk(self):
        dates = pd.bdate_range("2024-01-01", periods=6)
        daily = pd.DataFrame(
            {
                "date": list(dates) * 2,
                "variant": ["one"] * 6 + ["two"] * 6,
                "portfolio_return": [0.01, -0.02, 0.03, 0.01, -0.01, 0.02] * 2,
                "turnover": [0.1] * 12,
                "cost_amount": [1.0] * 12,
                "gross_exposure": [0.8] * 12,
            }
        )
        regimes = pd.DataFrame(
            {"date": dates, "regime": ["calm", "calm", "calm", "stress", "stress", "stress"]}
        )
        result = regime_attribution(daily, regimes, dates[0])
        for _, group in result.groupby("variant"):
            expected = np.log1p(daily.loc[daily.variant == group.variant.iloc[0], "portfolio_return"]).sum()
            self.assertAlmostEqual(group.log_return_contribution.sum(), expected)
            self.assertAlmostEqual(group.variance_share.sum(), 1.0)

    def test_run_artifact_loader_scopes_run_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            target = root / "run_id=test" / "year=2024"
            target.mkdir(parents=True)
            pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "value": [1]}).to_parquet(
                target / "part.parquet"
            )
            loaded = load_run_artifact(root, "test")
            self.assertEqual(loaded.value.tolist(), [1])
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(loaded.date))

    def test_report_config_rejects_invalid_probabilities(self):
        for changes in ({"capacity_quantile": 0}, {"extreme_loss_threshold": 1}, {"trading_days": 1}):
            with self.assertRaises(ValueError):
                ReportConfig(**changes)


if __name__ == "__main__":
    unittest.main()
