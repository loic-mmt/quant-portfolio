from datetime import datetime, timezone
import unittest

from quant_portfolio.core.ids import create_run_id, ensure_run_id, validate_run_id


class RunIdTests(unittest.TestCase):
    def test_run_id_is_sortable_and_valid(self):
        run_id = create_run_id("backtest", datetime(2026, 7, 19, tzinfo=timezone.utc))
        self.assertTrue(run_id.startswith("backtest-20260719T000000"))
        self.assertEqual(validate_run_id(run_id), run_id)

    def test_caller_run_id_is_preserved(self):
        self.assertEqual(ensure_run_id("experiment-001"), "experiment-001")

    def test_unsafe_run_ids_are_rejected(self):
        for value in ["", "../escape", "has space", "/absolute"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_run_id(value)


if __name__ == "__main__":
    unittest.main()
