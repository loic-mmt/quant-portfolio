import unittest

import numpy as np

from quant_portfolio.models.allocation import OptimizationError, constraint_violations, solve_allocation


class AllocationTests(unittest.TestCase):
    def test_full_covariance_minimum_variance_matches_analytic_solution(self):
        cov = np.array([[0.04, 0.01], [0.01, 0.09]]) / 252
        expected = np.linalg.solve(cov, np.ones(2))
        expected /= expected.sum()
        result = solve_allocation(
            cov,
            np.array([0.5, 0.5]),
            minimum=np.zeros(2),
            maximum=np.ones(2),
            allow_cash=False,
            turnover_penalty=0,
            distance_penalty=0,
            transaction_cost_rate=0,
            exposure_penalty=0,
            target_vol_penalty=0,
        )
        np.testing.assert_allclose(result.weights, expected, atol=1e-6)
        inverse_variance = 1 / cov.diagonal()
        inverse_variance /= inverse_variance.sum()
        self.assertGreater(np.linalg.norm(result.weights - inverse_variance), 0.01)

    def test_bounds_sector_exposure_turnover_and_speed_all_hold(self):
        previous = np.array([0.2, 0.2, 0.2, 0.2])
        minimum, maximum = np.full(4, 0.05), np.full(4, 0.3)
        sectors, sector_caps = ["tech", "tech", "other", "other"], {"tech": 0.35}
        result = solve_allocation(
            np.diag([0.0001, 0.0002, 0.0003, 0.0004]),
            previous,
            minimum=minimum,
            maximum=maximum,
            desired_exposure=0.4,
            turnover_cap=0.12,
            exposure_change_cap=0.1,
            sectors=sectors,
            sector_caps=sector_caps,
            min_exposure=0.6,
            max_exposure=0.9,
        )
        violations = constraint_violations(
            result.weights,
            previous,
            minimum,
            maximum,
            allow_cash=True,
            turnover_cap=0.12,
            exposure_change_cap=0.1,
            sectors=sectors,
            sector_caps=sector_caps,
            min_exposure=0.6,
            max_exposure=0.9,
        )
        self.assertLessEqual(max(violations.values()), 1e-7)
        self.assertLess(result.weights.sum(), previous.sum())

    def test_no_renormalization_can_break_caps(self):
        result = solve_allocation(
            np.diag([0.00001, 0.0001, 0.0005]),
            np.zeros(3),
            minimum=np.full(3, 0.1),
            maximum=np.full(3, 0.4),
            allow_cash=False,
        )
        self.assertAlmostEqual(result.weights.sum(), 1.0)
        self.assertLessEqual(result.weights.max(), 0.4 + 1e-7)
        self.assertGreaterEqual(result.weights.min(), 0.1 - 1e-7)
        with self.assertRaises(OptimizationError):
            solve_allocation(
                np.eye(2), np.zeros(2), minimum=np.zeros(2), maximum=np.full(2, 0.3), allow_cash=False
            )

    def test_costs_turnover_and_distance_discourage_trades(self):
        arguments = dict(
            covariance=np.diag([0.0001, 0.002]),
            previous=np.array([0.1, 0.9]),
            minimum=np.zeros(2),
            maximum=np.ones(2),
            allow_cash=False,
            target_vol_penalty=0,
            exposure_penalty=0,
        )
        cheap = solve_allocation(**arguments, turnover_penalty=0, transaction_cost_rate=0, distance_penalty=0)
        for penalty in (
            dict(turnover_penalty=1.0, transaction_cost_rate=0, distance_penalty=0),
            dict(turnover_penalty=0, transaction_cost_rate=1.0, distance_penalty=0),
            dict(turnover_penalty=0, transaction_cost_rate=0, distance_penalty=10.0),
        ):
            costly = solve_allocation(**arguments, **penalty)
            self.assertLess(
                np.linalg.norm(costly.weights - arguments["previous"]),
                np.linalg.norm(cheap.weights - arguments["previous"]),
            )

    def test_solver_failure_hold_only_when_previous_is_feasible(self):
        arguments = dict(covariance=np.eye(2), minimum=np.zeros(2), maximum=np.full(2, 0.5), solver="MISSING")
        result = solve_allocation(**arguments, previous=np.array([0.4, 0.4]), fallback="hold")
        self.assertIn("fallback_hold", result.status)
        with self.assertRaises(OptimizationError):
            solve_allocation(**arguments, previous=np.array([0.6, 0.4]), fallback="hold")
        with self.assertRaises(OptimizationError):
            solve_allocation(**arguments, previous=np.array([0.4, 0.4]), fallback="error")

    def test_sector_caps_require_metadata(self):
        with self.assertRaises(ValueError):
            solve_allocation(
                np.eye(2), np.zeros(2), minimum=np.zeros(2), maximum=np.ones(2), sector_caps={"tech": 0.3}
            )

    def test_target_vol_penalty_reduces_high_volatility(self):
        arguments = dict(
            covariance=np.diag([0.01, 0.02]),
            previous=np.array([0.5, 0.5]),
            minimum=np.zeros(2),
            maximum=np.ones(2),
            desired_exposure=1.0,
            target_vol=0.1,
            exposure_penalty=1.0,
        )
        unpenalized = solve_allocation(**arguments, target_vol_penalty=0)
        penalized = solve_allocation(**arguments, target_vol_penalty=100)
        self.assertLess(penalized.weights.sum(), unpenalized.weights.sum())


if __name__ == "__main__":
    unittest.main()
