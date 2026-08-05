from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from quant_portfolio.core.storage import read_parquet_dataset, replace_partitioned_dataset
from quant_portfolio.pipeline.features import (
    breadth,
    build_market_index,
    compute_feature_datasets,
    merge_feature_snapshots,
)


def synthetic_prices(periods: int = 280) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=periods)
    rows = []
    for index, ticker in enumerate(["AAA", "BBB"]):
        returns = 0.0004 + 0.0002 * np.sin(np.arange(periods) / (9.0 + index))
        levels = (100.0 + 20.0 * index) * np.cumprod(1.0 + returns)
        for date, level in zip(dates, levels):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "adj_close": level,
                    "volume": 1_000 + index,
                }
            )
    return pd.DataFrame(rows)


class FeatureTests(unittest.TestCase):
    def test_market_proxy_is_invariant_to_nominal_price_levels(self):
        prices = synthetic_prices(30)
        scaled = prices.copy()
        scaled.loc[scaled["ticker"].eq("AAA"), "adj_close"] *= 1_000.0
        scaled.loc[scaled["ticker"].eq("BBB"), "adj_close"] *= 0.01

        original = build_market_index(prices)["adj_close"]
        changed = build_market_index(scaled)["adj_close"]

        assert_series_equal(original, changed, check_names=False, rtol=1e-12, atol=1e-12)

    def test_breadth_excludes_missing_assets_from_denominator(self):
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
                "ticker": ["AAA", "BBB", "AAA"],
                "adj_close": [100.0, 100.0, 101.0],
            }
        )
        result = breadth(frame)
        self.assertEqual(result.loc[pd.Timestamp("2024-01-02"), "breadth_up"], 1.0)

    def test_incremental_merge_equals_full_recomputation(self):
        prices = synthetic_prices()
        first_prices = prices[prices["date"] <= prices["date"].sort_values().unique()[259]]
        regime_first, assets_first = compute_feature_datasets(first_prices, ["AAA", "BBB"])
        regime_full, assets_full = compute_feature_datasets(prices, ["AAA", "BBB"])

        regime_incremental = merge_feature_snapshots(
            regime_first,
            regime_full,
            ["date"],
            full_recompute=False,
        )
        assets_incremental = merge_feature_snapshots(
            assets_first,
            assets_full,
            ["ticker", "date"],
            full_recompute=False,
        )

        assert_frame_equal(
            regime_incremental.reset_index(drop=True),
            regime_full.reset_index(drop=True),
        )
        assert_frame_equal(
            assets_incremental.reset_index(drop=True),
            assets_full.reset_index(drop=True),
        )

    def test_snapshot_replacement_is_idempotent(self):
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "ticker": ["AAA", "AAA"],
                "value": [1.0, 2.0],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "features"
            replace_partitioned_dataset(frame, target, ["ticker", "year"])
            replace_partitioned_dataset(frame, target, ["ticker", "year"])
            loaded = read_parquet_dataset(target).sort_values("date")

        self.assertEqual(len(loaded), 2)
        self.assertFalse(loaded.duplicated(["ticker", "date"]).any())


if __name__ == "__main__":
    unittest.main()
