from pathlib import Path
import tempfile
import unittest

from quant_portfolio.core.settings import load_base_config
from quant_portfolio.core.universe import load_universe


class UniverseTests(unittest.TestCase):
    def test_default_universe_is_versioned_and_fx_complete(self):
        config = load_base_config()
        universe = config.universe

        self.assertEqual(universe.version, 1)
        self.assertEqual(universe.reference_currency, "EUR")
        self.assertEqual(len(universe.fingerprint), 64)
        self.assertEqual(len(universe.tickers), len(set(universe.tickers)))

        foreign = set(universe.currencies) - {universe.reference_currency}
        self.assertEqual(foreign, set(universe.fx_by_currency))

    def test_universe_rejects_missing_fx_rate(self):
        content = """
universe_id: test-v1
version: 1
as_of: "2024-01-01"
constituent_source: fixture
survivorship_bias: known
reference_currency: EUR
fx_policy: convert_to_reference
assets:
  - {ticker: USD_ASSET, currency: USD}
fx_rates: {}
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "universe.yaml"
            path.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing FX rates"):
                load_universe(path)


if __name__ == "__main__":
    unittest.main()
