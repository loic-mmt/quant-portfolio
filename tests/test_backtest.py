from dataclasses import replace
import unittest

import numpy as np
import pandas as pd

from quant_portfolio.pipeline.backtest import (
    DEFAULT_CFG,
    build_execution_schedule,
    build_returns_matrix,
    compute_turnover,
    simulate_portfolio,
)


def _weights(decision_date: str, values: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": pd.Timestamp(decision_date), "ticker": ticker, "weight": weight}
            for ticker, weight in values.items()
        ]
    )


def _config(**changes):
    values = {
        "transaction_bps": 0.0,
        "slippage_bps": 0.0,
        "cash_rate_annual": 0.0,
        "execution_lag_days": 1,
        "return_type": "simple",
        "missing_return_policy": "cash",
    }
    values.update(changes)
    return replace(DEFAULT_CFG, **values)


class BacktestTests(unittest.TestCase):
    def test_simple_returns_are_not_log_returns(self):
        prices = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "ticker": ["A", "A"],
                "adj_close": [100.0, 110.0],
            }
        )
        returns = build_returns_matrix(prices, return_type="simple")
        self.assertAlmostEqual(returns.iloc[0, 0], 0.10)

    def test_decision_executes_on_next_trading_day(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05"])
        decisions = pd.DataFrame([[1.0]], index=[pd.Timestamp("2024-01-02")], columns=["A"])
        schedule = build_execution_schedule(decisions, dates, lag_days=1)
        self.assertEqual(list(schedule), [pd.Timestamp("2024-01-03")])
        self.assertEqual(schedule[pd.Timestamp("2024-01-03")][0], pd.Timestamp("2024-01-02"))

    def test_same_day_return_is_not_captured(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        returns = pd.DataFrame({"A": [0.50, 0.0]}, index=dates)
        results = simulate_portfolio(returns, _weights("2024-01-02", {"A": 1.0}), _config())
        self.assertAlmostEqual(results.loc[pd.Timestamp("2024-01-02"), "portfolio_value"], 100_000.0)
        self.assertAlmostEqual(results.loc[pd.Timestamp("2024-01-03"), "portfolio_value"], 100_000.0)

    def test_weights_drift_between_rebalances(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        returns = pd.DataFrame({"A": [0.0, 0.10], "B": [0.0, 0.0]}, index=dates)
        _, _, positions = simulate_portfolio(
            returns,
            _weights("2024-01-02", {"A": 0.5, "B": 0.5}),
            _config(),
            return_details=True,
        )
        end = positions[positions["date"] == pd.Timestamp("2024-01-03")].set_index("ticker")
        self.assertAlmostEqual(end.loc["A", "weight"], 0.55 / 1.05)
        self.assertAlmostEqual(end.loc["B", "weight"], 0.50 / 1.05)
        self.assertAlmostEqual(end["value"].sum(), 105_000.0)

    def test_initial_allocation_is_charged_and_cash_is_explicit(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        returns = pd.DataFrame({"A": [0.0, 0.0]}, index=dates)
        cfg = _config(transaction_bps=100.0)
        results, trades, positions = simulate_portfolio(
            returns,
            _weights("2024-01-02", {"A": 0.6}),
            cfg,
            return_details=True,
        )
        execution = results.loc[pd.Timestamp("2024-01-03")]
        self.assertAlmostEqual(execution["turnover"], 0.6)
        self.assertAlmostEqual(execution["cost"], 0.006)
        self.assertAlmostEqual(execution["portfolio_value"], 99_400.0)
        end = positions[positions["date"] == pd.Timestamp("2024-01-03")].set_index("ticker")
        self.assertAlmostEqual(end.loc["__CASH__", "weight"], 0.4)
        self.assertEqual(set(trades["ticker"]), {"A", "__CASH__"})

    def test_single_asset_reconciles_with_buy_and_hold(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        returns = pd.DataFrame({"A": [0.10, -0.05, 0.02]}, index=dates)
        results = simulate_portfolio(
            returns,
            _weights("2023-12-29", {"A": 1.0}),
            _config(),
        )
        expected = 100_000.0 * np.prod(1.0 + returns["A"].to_numpy())
        self.assertAlmostEqual(results["portfolio_value"].iloc[-1], expected)

    def test_future_data_does_not_change_past_results(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        base = pd.DataFrame({"A": [0.01, 0.02, 0.03]}, index=dates)
        weights = _weights("2023-12-29", {"A": 1.0})
        expected = simulate_portfolio(base.iloc[:2], weights, _config())
        mutated = base.copy()
        mutated.loc[pd.Timestamp("2024-01-04"), "A"] = -0.99
        actual = simulate_portfolio(mutated, weights, _config()).iloc[:2]
        pd.testing.assert_frame_equal(expected, actual)

    def test_turnover_includes_implicit_cash_leg(self):
        prev = np.array([0.0, 0.0])
        nxt = np.array([0.4, 0.2])
        self.assertAlmostEqual(compute_turnover(prev, nxt, prev_cash=1.0, next_cash=0.4), 0.6)

    def test_rejects_leveraged_target(self):
        returns = pd.DataFrame(
            {"A": [0.0, 0.0], "B": [0.0, 0.0]},
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        )
        with self.assertRaisesRegex(ValueError, "exposure exceeds"):
            simulate_portfolio(
                returns,
                _weights("2024-01-02", {"A": 0.8, "B": 0.8}),
                _config(),
            )


if __name__ == "__main__":
    unittest.main()
