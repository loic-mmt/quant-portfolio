import sqlite3
import unittest

from quant_portfolio.core.db import initialize_metadata_db


class SqlSchemaTests(unittest.TestCase):
    def test_schema_and_views_apply_to_empty_database(self):
        with sqlite3.connect(":memory:") as connection:
            initialize_metadata_db(connection)
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            self.assertTrue({"last_dates", "feature_last_dates", "regimes_last_dates", "backtests"} <= tables)


if __name__ == "__main__":
    unittest.main()
