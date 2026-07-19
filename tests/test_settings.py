from pathlib import Path
import tempfile
import unittest

from quant_portfolio.core.settings import PROJECT_ROOT, load_base_config
from quant_portfolio.pipeline.backtest import load_backtest_config


class SettingsTests(unittest.TestCase):
    def test_repository_configuration_is_valid(self):
        config = load_base_config()
        self.assertEqual(config.data_dir, PROJECT_ROOT / "data")
        self.assertGreaterEqual(len(config.tickers), 40)
        self.assertEqual(len(config.tickers), len(set(config.tickers)))

        backtest = load_backtest_config()
        self.assertGreaterEqual(backtest.execution_lag_days, 1)
        self.assertEqual(backtest.return_type, "simple")

    def test_invalid_base_configuration_fails_early(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.yaml"
            path.write_text("tickers: [A, A]\nunknown: true\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_base_config(path)


if __name__ == "__main__":
    unittest.main()
