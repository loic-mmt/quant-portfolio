import json
import unittest
from dataclasses import replace

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from quant_portfolio.models.covariance import CovarianceConfig
from quant_portfolio.pipeline.backtest import DEFAULT_CFG as BACKTEST_DEFAULT
from quant_portfolio.pipeline.backtest import simulate_portfolio
from quant_portfolio.pipeline.mc import MCConfig
from quant_portfolio.pipeline.optimize import (
    OptimizeConfig,
    build_rebalance_dates,
    optimize_over_time,
    overlay_exposure,
    resolve_execution_config,
)

COV = CovarianceConfig("shrink_diag", 0.2, 5, 1e-8)
MC = MCConfig(n_sims=200, window=30, min_observations=5, compare_distributions=())
BACKTEST = replace(BACKTEST_DEFAULT, transaction_bps=5.0, slippage_bps=3.0)
CONFIG = OptimizeConfig(
    max_weight=0.8,
    lookback=20,
    use_regimes=False,
    use_mc=False,
    target_vol=0.12,
    max_exposure_change=0.15,
    turnover_governor=0.20,
)


class StrategyTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2024-01-01", periods=65)
        rng = np.random.default_rng(20)
        self.returns = pd.DataFrame(rng.normal(0, 0.012, (65, 4)), index=self.dates, columns=list("ABCD"))
        self.regimes = pd.Series(np.where(np.arange(65) < 35, 0, 2), index=self.dates)

    def run_strategy(self, returns=None, cfg=CONFIG, backtest=BACKTEST, regimes=None):
        return optimize_over_time(
            self.returns if returns is None else returns,
            None,
            cfg,
            covariance_cfg=COV,
            mc_cfg=MC,
            backtest_cfg=backtest,
            regimes=regimes,
            return_details=True,
        )

    def test_rebalance_calendar_is_prefix_stable_for_every_frequency(self):
        for frequency in ("D", "W", "2W", "M"):
            for count in (8, 17, 35):
                short = build_rebalance_dates(self.dates[:count], frequency)
                full = build_rebalance_dates(self.dates, frequency)
                self.assertTrue(short.equals(full[full <= self.dates[count - 1]]))

    def test_actual_holdings_match_backtest_with_lag_cash_fees_and_execution_governor(self):
        cfg = replace(
            BACKTEST,
            cash_rate_annual=0.03,
            turnover_cap=0.08,
            execution_lag_days=2,
            start_date=str(self.dates[10].date()),
            end_date=str(self.dates[-2].date()),
        )
        result = self.run_strategy(backtest=cfg)
        simulation, _, positions = simulate_portfolio(self.returns, result.weights, cfg, return_details=True)
        wide = positions.pivot(index="date", columns="ticker", values="weight")
        for row in result.decisions.itertuples():
            effective = pd.Series(json.loads(row.effective_weights_json)).reindex(self.returns.columns)
            np.testing.assert_allclose(effective, wide.loc[row.date, self.returns.columns], atol=1e-12)
        self.assertLessEqual(simulation.turnover.max(), 0.08 + 1e-9)
        self.assertGreater(simulation.cost.sum(), 0.0)

    def test_log_backtest_conversion_matches_canonical_simple_strategy(self):
        result = self.run_strategy()
        simple = simulate_portfolio(self.returns, result.weights, BACKTEST)
        logs = simulate_portfolio(
            np.log1p(self.returns), result.weights, replace(BACKTEST, return_type="log")
        )
        np.testing.assert_allclose(simple.portfolio_value, logs.portfolio_value)

    def test_governors_apply_to_delayed_execution_without_separate_backtest_cap(self):
        backtest = replace(BACKTEST, execution_lag_days=3, turnover_cap=None)
        result = self.run_strategy(backtest=backtest)
        execution = resolve_execution_config(CONFIG, backtest)
        simulation, trades, positions = simulate_portfolio(
            self.returns, result.weights, execution, return_details=True
        )
        self.assertLessEqual(simulation.turnover.max(), CONFIG.turnover_governor + 1e-12)
        cash_trades = trades.loc[trades.ticker == "__CASH__", "delta_weight"]
        self.assertLessEqual(cash_trades.abs().max(), CONFIG.max_exposure_change + 1e-12)
        wide = positions.pivot(index="date", columns="ticker", values="weight")
        for row in result.decisions.itertuples():
            effective = pd.Series(json.loads(row.effective_weights_json)).reindex(self.returns.columns)
            np.testing.assert_allclose(effective, wide.loc[row.date, self.returns.columns], atol=1e-12)

    def test_execution_lag_cannot_be_fractional_or_same_day(self):
        for lag in (0, 1.5, True):
            with self.assertRaises(ValueError):
                replace(BACKTEST, execution_lag_days=lag)

    def test_all_targets_obey_bounds_daily_speed_turnover_and_cash(self):
        result = self.run_strategy()
        matrix = result.weights.pivot(index="date", columns="ticker", values="weight")
        self.assertTrue((matrix >= 0).all().all())
        self.assertLessEqual(matrix.to_numpy().max(), CONFIG.max_weight + 1e-7)
        self.assertLessEqual(matrix.sum(axis=1).max(), 1 + 1e-10)
        decisions = result.decisions.dropna(subset=["target_exposure"])
        self.assertLessEqual(
            (decisions.target_exposure - decisions.effective_exposure).abs().max(), 0.15 + 1e-7
        )
        self.assertLessEqual(decisions.turnover.max(), 0.20 + 1e-7)
        self.assertLessEqual(decisions.constraint_violation.max(), 1e-7)
        self.assertEqual(len(decisions), len(self.dates) - COV.min_periods)
        stable = decisions[(decisions.reason == "hysteresis") & ~decisions.composition_rebalance.astype(bool)]
        self.assertFalse(stable.empty)
        np.testing.assert_allclose(stable.target_exposure, stable.effective_exposure, atol=1e-9)

    def test_volatility_shock_reduces_exposure_with_cash(self):
        rng = np.random.default_rng(123)
        data = pd.DataFrame(rng.normal(0, 0.002, (65, 4)), index=self.dates, columns=list("ABCD"))
        data.iloc[35:] = rng.normal(0, 0.09, (30, 4))
        result = self.run_strategy(returns=data)
        decisions = result.decisions.set_index("date")
        self.assertGreater(decisions.target_exposure.iloc[25:34].mean(), 0.9)
        self.assertLess(decisions.target_exposure.iloc[-5:].mean(), 0.4)
        self.assertIn("reduce", decisions.action.values)

    def test_future_changes_leave_weights_risk_and_decisions_unchanged(self):
        cfg = replace(CONFIG, use_regimes=True, use_mc=True)
        cutoff = self.dates[43]
        prefix = self.run_strategy(
            returns=self.returns.loc[:cutoff], cfg=cfg, regimes=self.regimes.loc[:cutoff]
        )
        changed = self.returns.copy()
        changed.loc[changed.index > cutoff] = 0.3
        changed_states = self.regimes.copy()
        changed_states.loc[changed_states.index > cutoff] = 1
        extended = self.run_strategy(returns=changed, cfg=cfg, regimes=changed_states)
        for name in ("weights", "risk", "decisions", "risk_weights"):
            past = getattr(prefix, name)
            full = getattr(extended, name)
            assert_frame_equal(
                past.reset_index(drop=True), full.loc[full.date <= cutoff].reset_index(drop=True)
            )

    def test_regime_and_mc_ablations_are_distinct_and_reproducible(self):
        cfg = replace(CONFIG, use_regimes=True, use_mc=True, risk_var_limit=0.003, risk_cvar_limit=0.004)
        full = self.run_strategy(cfg=cfg, regimes=self.regimes)
        repeated = self.run_strategy(cfg=cfg, regimes=self.regimes)
        no_regime = self.run_strategy(cfg=replace(cfg, use_regimes=False), regimes=None)
        no_mc = self.run_strategy(cfg=replace(cfg, use_mc=False), regimes=self.regimes)
        assert_frame_equal(full.weights, repeated.weights)
        assert_frame_equal(full.risk, repeated.risk)
        for variant in (no_regime, no_mc):
            self.assertGreater(np.linalg.norm(full.weights.weight - variant.weights.weight), 0.01)
        self.assertTrue(no_mc.risk.empty)
        self.assertTrue((no_regime.risk.calibration_status == "unconditional").all())
        self.assertEqual(set(full.risk.portfolio_role), {"effective", "candidate", "target"})
        self.assertTrue((full.risk.calibration_end < full.risk.date).all())
        effective = full.risk[full.risk.portfolio_role == "effective"].groupby("date").exposure.first()
        expected = full.decisions.set_index("date").loc[effective.index, "effective_exposure"]
        np.testing.assert_allclose(effective, expected)

    def test_overlay_hysteresis_and_cash_disabled_are_explicit(self):
        overlay = overlay_exposure(CONFIG.target_vol / 0.801, 0.8, 1.0, None, None, CONFIG)
        self.assertEqual(overlay["reason"], "hysteresis")
        self.assertEqual(overlay["desired_exposure"], 0.8)
        no_cash = overlay_exposure(2.0, 1.0, 1.0, None, None, replace(CONFIG, allow_cash=False))
        self.assertEqual(no_cash["desired_exposure"], 1.0)
        self.assertEqual(no_cash["reason"], "cash_disabled")

    def test_disabled_overlay_only_rebalances_composition_periodically(self):
        result = self.run_strategy(cfg=replace(CONFIG, use_overlay=False, turnover_governor=None))
        decisions = result.decisions.dropna(subset=["target_exposure"])
        self.assertLess(len(decisions), len(self.dates) / 2)
        self.assertTrue(decisions.composition_rebalance.all())

    def test_unknown_sector_metadata_or_missing_regimes_are_errors(self):
        with self.assertRaisesRegex(ValueError, "metadata"):
            self.run_strategy(cfg=replace(CONFIG, sector_caps={"tech": 0.2}))
        with self.assertRaisesRegex(ValueError, "regime"):
            self.run_strategy(cfg=replace(CONFIG, use_regimes=True))


if __name__ == "__main__":
    unittest.main()
