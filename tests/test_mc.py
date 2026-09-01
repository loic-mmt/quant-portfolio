import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from quant_portfolio.models.covariance import CovarianceConfig, estimate_covariance
from quant_portfolio.pipeline.mc import (
    InsufficientHistory,
    MCConfig,
    calibrate_mc,
    date_seed,
    simulate_calibration,
    simulate_paths,
    summarize_paths,
    summarize_portfolio,
)

COV = CovarianceConfig("shrink_diag", 0.2, 5, 1e-8)
MC = MCConfig(n_sims=200, window=30, min_observations=5, random_seed=7)


class MonteCarloTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2024-01-01", periods=70)
        rng = np.random.default_rng(10)
        self.returns = pd.DataFrame(rng.normal(0, 0.01, (70, 2)), index=self.dates, columns=["A", "B"])
        self.states = pd.Series(np.arange(70) % 2, index=self.dates)

    def test_calibration_uses_only_previous_rows_of_current_regime(self):
        date = self.dates[40]
        model = calibrate_mc(self.returns, date, ["A", "B"], MC, COV, self.states)
        selected = self.returns.iloc[:40].loc[self.states.iloc[:40] == self.states.loc[date]].tail(30)
        np.testing.assert_allclose(model.mu, np.log1p(selected).mean())
        self.assertEqual(model.calibration_end, selected.index.max())
        self.assertLess(model.calibration_end, date)
        self.assertEqual(model.calibration_status, "conditional")

    def test_future_changes_and_append_do_not_change_mc(self):
        date = self.dates[40]
        past = calibrate_mc(self.returns.iloc[:41], date, ["A", "B"], MC, COV, self.states.iloc[:41])
        changed = self.returns.copy()
        changed.loc[date:] = 0.8  # Includes today's return, excluded from calibration.
        full = calibrate_mc(changed, date, ["A", "B"], MC, COV, self.states)
        np.testing.assert_array_equal(past.mu, full.mu)
        np.testing.assert_array_equal(past.covariance, full.covariance)
        for dist, paths in simulate_calibration(past, date, MC).items():
            np.testing.assert_array_equal(paths, simulate_calibration(full, date, MC)[dist])

    def test_sparse_regime_policy_is_explicit(self):
        states = self.states * 0
        states.iloc[40:] = 2
        model = calibrate_mc(self.returns, self.dates[40], ["A", "B"], MC, COV, states)
        self.assertEqual(model.calibration_status, "pooled_sparse_regime")
        with self.assertRaises(InsufficientHistory):
            calibrate_mc(
                self.returns,
                self.dates[40],
                ["A", "B"],
                replace(MC, sparse_regime_policy="error"),
                COV,
                states,
            )

    def test_portfolio_is_weighted_buy_and_hold_not_sum_of_log_returns(self):
        paths = np.log1p(np.array([[[0.1, -0.2], [0.1, 0.3]], [[-0.1, 0.1], [-0.1, -0.1]]]))
        weights = np.array([0.2, 0.3])
        expected = np.array(
            [0.2 * 1.1**2 + 0.3 * 0.8 * 1.3 + 0.5 - 1, 0.2 * 0.9**2 + 0.3 * 1.1 * 0.9 + 0.5 - 1]
        )
        summary = summarize_paths(paths, weights)
        self.assertAlmostEqual(summary["expected_pnl"], expected.mean())
        self.assertAlmostEqual(summary["pnl_q05"], np.quantile(expected, 0.05))

    def test_cash_and_zero_weight_assets(self):
        paths = np.zeros((200, 5, 2))
        paths[:, :, 1] = -0.9
        summary = summarize_paths(paths, np.array([0.5, 0.0]), cash_return=0.001)
        self.assertAlmostEqual(summary["expected_pnl"], 0.5 * (1.001**5 - 1))
        all_cash = summarize_paths(paths, np.zeros(2))
        self.assertEqual(all_cash["var_01"], 0.0)
        self.assertEqual(all_cash["probability_loss"], 0.0)

    def test_drawdown_includes_initial_nav_and_intrahorizon_trough(self):
        paths = np.log1p(np.array([[[-0.2], [0.25]]] * 200))
        result = summarize_paths(paths, np.ones(1), drawdown_threshold=0.1)
        self.assertAlmostEqual(result["expected_pnl"], 0.0)
        self.assertEqual(result["probability_drawdown"], 1.0)

    def test_tail_metrics_are_losses_and_cash_scales_risk(self):
        paths = np.log1p(np.array([[[-0.1]], [[-0.2]], [[0.1]], [[-0.3]]] * 50))
        full = summarize_paths(paths, np.ones(1))
        half = summarize_paths(paths, np.array([0.5]))
        for key in ("var_01", "var_05", "cvar_01", "cvar_05", "pnl_q05"):
            self.assertAlmostEqual(half[key], 0.5 * full[key])
        self.assertGreaterEqual(full["cvar_01"], full["var_01"])

    def test_seed_is_local_and_student_matches_covariance_with_heavier_tails(self):
        np.random.seed(123)
        expected_global = np.random.random()
        np.random.seed(123)
        cov = np.array([[0.0004, 0.0001], [0.0001, 0.0009]])
        normal = simulate_paths(np.zeros(2), cov, 100000, 1, seed=5)
        student = simulate_paths(np.zeros(2), cov, 100000, 1, "student_t", seed=5, student_df=5)
        self.assertEqual(np.random.random(), expected_global)
        np.testing.assert_allclose(np.cov(student[:, 0].T), cov, rtol=0.08, atol=1e-5)
        self.assertLess(np.quantile(student[:, 0, 0], 0.001), np.quantile(normal[:, 0, 0], 0.001))
        self.assertNotEqual(date_seed(7, self.dates[0], "gaussian"), date_seed(8, self.dates[0], "gaussian"))

    def test_summaries_contain_both_horizons_distributions_and_actual_weights(self):
        date = self.dates[40]
        model = calibrate_mc(self.returns, date, ["A", "B"], MC, COV, self.states)
        rows = summarize_portfolio(
            model,
            simulate_calibration(model, date, MC),
            pd.Series({"A": 0.2, "B": 0.3}),
            date,
            "effective",
            MC,
        )
        self.assertEqual({r["horizon"] for r in rows}, {5, 20})
        self.assertEqual({r["distribution"] for r in rows}, {"gaussian", "student_t"})
        self.assertTrue(all(r["exposure"] == 0.5 and r["portfolio_role"] == "effective" for r in rows))

    def test_missing_held_asset_never_silently_dropped(self):
        self.returns["B"] = np.nan
        with self.assertRaises(InsufficientHistory):
            calibrate_mc(self.returns, self.dates[40], ["A", "B"], MC, COV, self.states)

    def test_pairwise_covariance_is_shrunk_positive_definite_and_explicit(self):
        data = self.returns.iloc[:30].copy()
        data.iloc[0, 0] = np.nan
        result = estimate_covariance(data, replace(COV, method="ledoit_wolf"))
        self.assertEqual(result.estimator, "shrink_diag_pairwise")
        self.assertGreater(np.linalg.eigvalsh(result.covariance).min(), 0)
        data.iloc[:25, 0] = np.nan
        data.iloc[25:, 1] = np.nan
        with self.assertRaises(ValueError):
            estimate_covariance(data, COV)

    def test_configuration_rejects_invalid_values(self):
        for values in (
            {"student_df": 2},
            {"dist": "bad"},
            {"random_seed": -1},
            {"n_sims": 1},
            {"horizons": (0,)},
            {"sparse_regime_policy": "silent"},
        ):
            with self.assertRaises(ValueError):
                MCConfig(**values)


if __name__ == "__main__":
    unittest.main()
