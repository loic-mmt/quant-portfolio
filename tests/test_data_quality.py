import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from quant_portfolio.core.settings import load_base_config
from quant_portfolio.pipeline.data_quality import (
    build_price_quality_report,
    build_feature_quality_report,
    clean_price_data,
    convert_prices_to_reference_currency,
    write_json_report,
)


class DataQualityTests(unittest.TestCase):
    def test_clean_prices_removes_duplicates_and_non_positive_rows(self):
        raw = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
                "ticker": ["AAA", "AAA", "AAA"],
                "adj_close": [10.0, 11.0, 0.0],
            }
        )
        clean = clean_price_data(raw)

        self.assertEqual(len(clean), 1)
        self.assertEqual(clean.iloc[0]["adj_close"], 11.0)

        report = build_price_quality_report(raw, ["AAA"])
        self.assertEqual(report["summary"]["duplicate_rows"], 2)
        self.assertEqual(report["summary"]["non_positive_price_rows"], 1)

    def test_report_marks_missing_and_delisted_tickers_explicitly(self):
        raw = pd.DataFrame(
            {"date": ["2024-01-01"], "ticker": ["AAA"], "adj_close": [10.0]}
        )
        report = build_price_quality_report(
            raw,
            ["AAA", "BBB", "CCC"],
            configured_statuses={"CCC": "delisted"},
        )

        self.assertEqual(report["tickers"]["BBB"]["status"], "missing")
        self.assertEqual(report["tickers"]["CCC"]["status"], "delisted")

    def test_fx_conversion_uses_latest_past_rate_and_price_multiplier(self):
        universe = load_base_config().universe
        prices = pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03"],
                "ticker": ["MONY.L", "MONY.L"],
                "adj_close": [10_000.0, 11_000.0],
            }
        )
        fx = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-04"],
                "ticker": ["EURGBP=X", "EURGBP=X"],
                "adj_close": [0.80, 0.90],
            }
        )

        converted = convert_prices_to_reference_currency(prices, fx, universe)

        self.assertAlmostEqual(converted.iloc[0]["adj_close"], 125.0)
        self.assertAlmostEqual(converted.iloc[1]["adj_close"], 137.5)
        self.assertEqual(converted.iloc[0]["fx_rate"], 0.80)

    def test_feature_quality_report_is_json_serializable(self):
        universe = load_base_config().universe
        regime = pd.DataFrame(
            {"date": pd.to_datetime(["2024-01-01", "2024-01-02"]), "feature": [1.0, 2.0]}
        )
        assets = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "ticker": ["PDD", "PDD"],
                "feature": [1.0, float("nan")],
            }
        )
        report = build_feature_quality_report(regime, assets, universe=universe)

        self.assertEqual(report["assets"]["features"]["feature"]["coverage"], 0.5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quality.json"
            write_json_report(report, path)
            loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["universe"]["id"], universe.universe_id)


if __name__ == "__main__":
    unittest.main()
