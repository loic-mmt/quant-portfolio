import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from quant_portfolio.models.hmm import (
    economic_state_mapping,
    filtered_hmm_probabilities,
    fit_hmm_features,
)


class HmmTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)
        values = np.vstack(
            [
                rng.normal([-1.5, 0.3], [0.25, 0.10], size=(60, 2)),
                rng.normal([0.0, 1.0], [0.25, 0.10], size=(60, 2)),
                rng.normal([1.5, 2.0], [0.25, 0.10], size=(60, 2)),
            ]
        )
        self.features = pd.DataFrame(values, columns=["momentum", "volatility"])

    def test_filtered_probabilities_do_not_use_future_observations(self):
        model = fit_hmm_features(self.features, random_seed=7, n_iter=80)
        prefix = filtered_hmm_probabilities(model, self.features.iloc[:80])
        extended = filtered_hmm_probabilities(model, self.features.iloc[:160]).iloc[:80]

        assert_frame_equal(prefix, extended, rtol=1e-12, atol=1e-12)

    def test_same_seed_produces_same_model_and_probabilities(self):
        first = fit_hmm_features(self.features, random_seed=11, n_iter=80)
        second = fit_hmm_features(self.features, random_seed=11, n_iter=80)

        np.testing.assert_allclose(first.startprob_, second.startprob_)
        np.testing.assert_allclose(first.transmat_, second.transmat_)
        np.testing.assert_allclose(first.means_, second.means_)
        assert_frame_equal(
            filtered_hmm_probabilities(first, self.features),
            filtered_hmm_probabilities(second, self.features),
        )

    def test_filter_supports_every_configured_covariance_type(self):
        for covariance_type in ("diag", "full", "tied", "spherical"):
            with self.subTest(covariance_type=covariance_type):
                model = fit_hmm_features(
                    self.features,
                    covariance_type=covariance_type,
                    random_seed=5,
                    n_iter=40,
                )
                probabilities = filtered_hmm_probabilities(model, self.features)
                np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)

    def test_economic_mapping_is_independent_from_raw_state_number(self):
        features = pd.DataFrame(
            {
                "mom_mkt_20": [2.0, 2.1, 1.9, 0.1, 0.0, -0.1, -2.0, -2.1, -1.9],
                "vol_mkt_20": [0.2, 0.3, 0.2, 1.0, 1.1, 0.9, 2.0, 2.1, 1.9],
                "breadth_up_20": [0.9, 0.8, 0.9, 0.5, 0.6, 0.5, 0.1, 0.2, 0.1],
            }
        )
        original = pd.DataFrame(
            np.repeat(np.eye(3), 3, axis=0),
            columns=["p_raw_0", "p_raw_1", "p_raw_2"],
        )
        original_mapping, _ = economic_state_mapping(features, original)

        permuted = original.rename(columns={"p_raw_0": "tmp", "p_raw_2": "p_raw_0"}).rename(
            columns={"tmp": "p_raw_2"}
        )[["p_raw_0", "p_raw_1", "p_raw_2"]]
        permuted_mapping, _ = economic_state_mapping(features, permuted)

        self.assertEqual(original_mapping[0], "calm")
        self.assertEqual(original_mapping[2], "stress")
        self.assertEqual(permuted_mapping[2], "calm")
        self.assertEqual(permuted_mapping[0], "stress")


if __name__ == "__main__":
    unittest.main()
