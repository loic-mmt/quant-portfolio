import unittest

from quant_portfolio.main import build_parser


class CliTests(unittest.TestCase):
    def test_cli_exposes_all_pipeline_commands(self):
        parser = build_parser()
        subparser_action = next(action for action in parser._actions if action.dest == "command")
        self.assertEqual(
            set(subparser_action.choices),
            {"ingest", "features", "regimes", "mc", "optimize", "backtest", "report", "run-all"},
        )

    def test_backtest_cli_parses_run_id(self):
        args = build_parser().parse_args(["backtest", "--run-id", "experiment-001"])
        self.assertEqual(args.command, "backtest")
        self.assertEqual(args.run_id, "experiment-001")


if __name__ == "__main__":
    unittest.main()
