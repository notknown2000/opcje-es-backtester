import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from taleb_kurtosis.digital_tail import Config, build_candidate, expiry_pnl, funding_count, mark_package, run, write_report


def chain(when="2020-02-19"):
    rows = []
    for strike, bid, ask, delta, iv in [
        (2200, 2.29, 2.40, -0.0104, 0.33115),
        (2825, 9.50, 10.0, -0.08, 0.20),
        (2850, 11.11, 11.50, -0.09, 0.19),
        (3375, 80.0, 81.0, -0.49, 0.11912),
    ]:
        rows.append(dict(QUOTE_DATE=pd.Timestamp(when), EXPIRE_DATE=pd.Timestamp("2020-06-19"),
                         UNDERLYING_LAST=3386.16, STRIKE=strike, P_BID=bid, P_ASK=ask,
                         P_DELTA=delta, P_IV=iv, DTE=(pd.Timestamp("2020-06-19") - pd.Timestamp(when)).days))
    return pd.DataFrame(rows)


class DigitalTailTests(unittest.TestCase):
    def test_funding_covers_round_trip_costs_and_is_minimal(self):
        n = funding_count(2.4, 1.11, 1)
        self.assertEqual(n, 3)
        self.assertGreaterEqual(n * 111 - 240 - 2 * (1 + 2*n), 0)
        self.assertLess((n-1) * 111 - 240 - 2 * (1 + 2*(n-1)), 0)
        self.assertIsNone(funding_count(2.4, 0.04, 1))

    def test_atm_sigma_actual_term_and_adjacent_spread(self):
        row, reason = build_candidate(chain(), pd.Timestamp("2020-02-19"), Config())
        self.assertEqual(reason, "candidate")
        self.assertEqual(row["spread_count"], 3)
        self.assertEqual(row["short_strike"], 2850)
        self.assertEqual(row["hedge_strike"], 2825)
        self.assertEqual(row["tail_strike"], 2200)
        self.assertAlmostEqual(row["sigma_horizon"], 0.11912 * np.sqrt(121/365))
        self.assertAlmostEqual(row["net_credit_after_costs"], 79)
        self.assertAlmostEqual(row["risk_budget"], 7421)

    def test_payoff_kinks_and_breakevens(self):
        row, _ = build_candidate(chain(), pd.Timestamp("2020-02-19"), Config())
        spots = np.array([3400, 2850, 2825, 2500, 2200, 2000, row["lower_breakeven"], row["upper_breakeven"]])
        np.testing.assert_allclose(expiry_pnl(row, spots), [79,79,-7421,-7421,-7421,12579,0,0], atol=1e-7)

    def test_missing_delta_and_crossed_quotes_rejected(self):
        frame = chain()
        frame.loc[0,"P_DELTA"] = -0.04
        self.assertEqual(build_candidate(frame, pd.Timestamp("2020-02-19"), Config())[1], "tail_delta_outside_tolerance")
        frame = chain()
        frame.loc[1,"P_BID"] = 30
        self.assertEqual(build_candidate(frame, pd.Timestamp("2020-02-19"), Config())[1], "invalid_entry_quotes")

    def test_missing_marks_are_not_filled_with_intrinsic(self):
        frame = chain()
        row, _ = build_candidate(frame, pd.Timestamp("2020-02-19"), Config())
        self.assertIsNone(mark_package(row, frame.iloc[1:].set_index(["EXPIRE_DATE", "STRIKE"]), Config()))

    def test_overlapping_expiry_quotes_allow_exit_without_negative_spread_mark(self):
        frame = chain()
        row, _ = build_candidate(frame, pd.Timestamp("2020-02-19"), Config())
        frame.loc[1, ["P_BID", "P_ASK"]] = [0, .06]
        frame.loc[2, ["P_BID", "P_ASK"]] = [0, .04]
        marked = mark_package(row, frame.set_index(["EXPIRE_DATE", "STRIKE"]), Config())
        self.assertTrue(marked["mark_projected"])
        self.assertAlmostEqual(marked["spread_pnl"], row["spread_entry_cash"] - 6)
        self.assertAlmostEqual(marked["spread_liq"], row["spread_entry_cash"] - 12 - 6)

    def test_unreliable_atm_and_thursday_series_are_rejected(self):
        frame = chain()
        frame.loc[3, "P_IV"] = .001
        self.assertEqual(build_candidate(frame, pd.Timestamp("2020-02-19"), Config())[1], "invalid_atm_iv")
        frame = chain()
        frame["EXPIRE_DATE"] = pd.Timestamp("2020-06-18")
        self.assertEqual(build_candidate(frame, pd.Timestamp("2020-02-19"), Config())[1], "no_standard_monthly_friday_expiry")

    def test_execution_costs_and_decomposition(self):
        frame = chain()
        row, _ = build_candidate(frame, pd.Timestamp("2020-02-19"), Config())
        mark = mark_package(row, frame.set_index(["EXPIRE_DATE", "STRIKE"]), Config())
        self.assertAlmostEqual(mark["pnl"], mark["tail_pnl"] + mark["spread_pnl"])
        self.assertLess(mark["liquidation_pnl"], mark["pnl"])
        self.assertLess(mark["pnl"], 0)

    def test_liquidation_is_not_clipped_to_terminal_loss_budget(self):
        frame = chain()
        row, _ = build_candidate(frame, pd.Timestamp("2020-02-19"), Config())
        frame.loc[0, ["P_BID", "P_ASK"]] = [0, .20]
        frame.loc[1, ["P_BID", "P_ASK"]] = [135.90, 148.11]
        frame.loc[2, ["P_BID", "P_ASK"]] = [161.00, 171.30]
        mark = mark_package(row, frame.set_index(["EXPIRE_DATE", "STRIKE"]), Config())
        self.assertLess(mark["liquidation_pnl"], -row["risk_budget"])
        self.assertGreaterEqual(mark["pnl"], -row["risk_budget"])
        self.assertAlmostEqual(mark["spread_liq"], row["spread_entry_cash"] - 3 * 35.4 * 100 - 6)

    def test_runner_monthly_exits_cashflow_and_risk_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            pd.concat([chain("2020-02-19"), chain("2020-02-20")]).to_parquet(data / "spx_eod_202002.parquet")
            pd.concat([chain("2020-06-18"), chain("2020-06-19")]).to_parquet(data / "spx_eod_202006.parquet")
            cfg = Config(initial_capital=1_000_000)
            with patch("taleb_kurtosis.digital_tail.write_report"):
                manifest = run(data, root / "run", cfg)
                blocked = run(data, root / "blocked", Config(initial_capital=1000))
            self.assertEqual(manifest["summary"]["trades"], 1)
            self.assertEqual(blocked["summary"]["trades"], 0)
            trades = pd.read_parquet(root / "run/trades.parquet")
            equity = pd.read_parquet(root / "run/daily_equity.parquet")
            self.assertEqual(trades.iloc[0]["exit_date"], pd.Timestamp("2020-06-18"))
            self.assertAlmostEqual(trades["pnl"].sum(), equity["equity_pnl"].iloc[-1])
            np.testing.assert_allclose(equity["equity_pnl"], equity["tail_pnl"] + equity["spread_pnl"])
            self.assertEqual(manifest["summary"]["unresolved_packages"], 0)
            self.assertEqual(json.loads((root / "run/manifest.json").read_text())["config"]["tail_delta"], .01)
            write_report(root / "run", trades, equity, manifest)
            report = (root / "run/report.html").read_text()
            self.assertEqual(report.count('class="plotly-graph-div"'), 6)
            self.assertIn("Mala proba transakcji", report)
            self.assertIn("Miesiace z wejsciem: 1/2", report)


if __name__ == "__main__":
    unittest.main()
