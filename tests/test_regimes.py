import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from quant_portfolio.core.db import init_regime_last_dates_db, upsert_regime_last_dates
from quant_portfolio.pipeline.regimes import (
    RegimeConfig,
    apply_regime_confirmation,
    compute_regime_diagnostics,
    load_regime_config,
    walk_forward_regimes,
    write_calibration_artifacts,
)


def regime_config() -> RegimeConfig:
    return RegimeConfig(
        n_states=3,
        random_seed=17,
        covariance_type="diag",
        n_iter=60,
        tol=1e-3,
        min_covar=1e-5,
        min_train_size=90,
        train_window=120,
        recalibration_frequency="40B",
        confidence_threshold=0.55,
        short_confirmation_days=20,
        short_confirmation_share=0.60,
        long_confirmation_days=60,
        long_confirmation_probability=0.40,
        regime_features=(
            "mom_mkt_20",
            "vol_mkt_20",
            "avg_corr_20",
            "breadth_up_20",
        ),
    )


def synthetic_regime_features(periods: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(2024)
    dates = pd.bdate_range("2020-01-01", periods=periods)
    profiles = np.asarray(
        [
            [1.5, 0.3, 0.2, 0.8],
            [0.0, 1.0, 0.5, 0.5],
            [-1.5, 2.0, 0.8, 0.2],
        ]
    )
    state = (np.arange(periods) // 30) % 3
    values = profiles[state] + rng.normal(0.0, 0.08, size=(periods, 4))
    return pd.DataFrame(values, index=dates, columns=regime_config().regime_features)


class RegimeTests(unittest.TestCase):
    def test_repository_regime_configuration_is_valid(self):
        config = load_regime_config()
        self.assertEqual(config.n_states, 3)
        self.assertEqual(config.short_confirmation_days, 20)
        self.assertEqual(config.long_confirmation_days, 60)
        self.assertGreaterEqual(config.train_window, config.min_train_size)

    def test_walk_forward_is_deterministic_and_future_invariant(self):
        config = regime_config()
        full_features = synthetic_regime_features()
        prefix_features = full_features.iloc[:190]

        prefix, prefix_metadata, _ = walk_forward_regimes(prefix_features, config)
        full, full_metadata, _ = walk_forward_regimes(full_features, config)
        repeated, _, _ = walk_forward_regimes(full_features, config)

        assert_frame_equal(prefix, full.loc[prefix.index], rtol=1e-11, atol=1e-11)
        assert_frame_equal(full, repeated, rtol=1e-11, atol=1e-11)
        self.assertTrue(
            all(
                pd.Timestamp(record["train_end"]) < pd.Timestamp(record["prediction_start"])
                for record in prefix_metadata
            )
        )
        self.assertGreater(len(full_metadata), len(prefix_metadata))
        self.assertEqual(set(full["regime"].unique()) <= {"calm", "choppy", "stress"}, True)
        diagnostics = compute_regime_diagnostics(full, full_features, full_metadata)
        self.assertEqual(diagnostics["recalibrations"], len(full_metadata))
        self.assertEqual(set(diagnostics["transition_probabilities"]), {"calm", "choppy", "stress"})

    def test_confirmation_requires_short_and_long_evidence(self):
        config = regime_config()
        dates = pd.bdate_range("2024-01-01", periods=120)
        probabilities = pd.DataFrame(
            {
                "p_calm": [0.90] * 60 + [0.05] * 60,
                "p_choppy": [0.05] * 120,
                "p_stress": [0.05] * 60 + [0.90] * 60,
            },
            index=dates,
        )

        confirmed = apply_regime_confirmation(probabilities, config)
        first_stress = confirmed.index[confirmed["regime"].eq("stress")][0]

        self.assertGreaterEqual(first_stress, dates[79])
        self.assertEqual(confirmed.loc[first_stress, "transition_reason"], "confirmed_20_60")
        self.assertEqual(int(confirmed.loc[first_stress, "state"]), 2)

    def test_calibration_artifacts_are_replaced_idempotently(self):
        records = [
            {"calibration_date": "2024-01-01", "model": {"means": [[0.0]]}},
            {"calibration_date": "2024-02-01", "model": {"means": [[1.0]]}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "calibrations"
            write_calibration_artifacts(records, target)
            write_calibration_artifacts(records, target)
            files = sorted(path.name for path in target.glob("*.json"))

        self.assertEqual(files, ["2024-01-01.json", "2024-02-01.json", "index.json"])

    def test_freshness_cache_keeps_latest_semantic_state_and_probability(self):
        rows = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "ticker": ["__MARKET__", "__MARKET__"],
                "state": [0, 2],
                "proba": [0.70, 0.85],
            }
        )
        with sqlite3.connect(":memory:") as connection:
            init_regime_last_dates_db(connection)
            upsert_regime_last_dates(connection, "regime", rows)
            result = connection.execute("SELECT date, state, proba FROM regimes_last_dates").fetchone()

        self.assertEqual(result, ("2024-01-02", 2, 0.85))


if __name__ == "__main__":
    unittest.main()
