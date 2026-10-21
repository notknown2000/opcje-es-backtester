import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from taleb_kurtosis.digital_tail import Config, MonthlyExpiryMap, build_candidate, run
from taleb_kurtosis.digital_tail_comparison import DigitalTailPortfolio, exit_signal, run_comparison
from test_digital_tail import chain


class ExitTests(unittest.TestCase):
    def setUp(self):
        self.pos = {"short_strike": 2850, "risk_budget": 1000, "planned_exit_date": pd.Timestamp("2020-06-18")}
        self.when = pd.Timestamp("2020-03-18")
        self.cfg = Config(exit_policy="barrier_close_all")

    def test_barrier_is_inclusive_and_does_not_require_a_profit(self):
        self.assertEqual(exit_signal(self.pos, self.when, 2850, {"liquidation_pnl": -500}, self.cfg), "barrier_close_all")
        self.assertIsNone(exit_signal(self.pos, self.when, 2850.01, {"liquidation_pnl": -500}, self.cfg))

    def test_hold_ignores_barrier_and_uses_expiry_fallback(self):
        self.assertIsNone(exit_signal(self.pos, self.when, 2000, {}, Config()))
        self.assertEqual(exit_signal(self.pos, self.pos["planned_exit_date"], 3500, {}, Config()), "pre_expiry_bid_ask")

    def test_legacy_runner_does_not_silently_ignore_new_policies(self):
        with self.assertRaisesRegex(ValueError, "run_comparison"):
            run(Path("unused"), Path("unused"), self.cfg)

    def test_profit_uses_whole_executable_package_after_costs(self):
        cfg = replace(self.cfg, profit_target_risk=.05)
        self.assertIsNone(exit_signal(self.pos, self.when, 3000, {"liquidation_pnl": -10, "tail_liq": 1000}, cfg))
        self.assertEqual(exit_signal(self.pos, self.when, 3000, {"liquidation_pnl": 50}, cfg), "package_profit_target")
        self.assertIsNone(exit_signal(self.pos, self.when, 3000, None, cfg))


class ExpiryTests(unittest.TestCase):
    def setUp(self):
        self.mapper = MonthlyExpiryMap(pd.Timestamp("2020-01-01"), pd.Timestamp("2022-12-31"))

    def test_last_trade_settlement_and_legacy_saturday_share_model_expiry(self):
        for date in ["2020-06-18", "2020-06-19", "2020-06-20"]:
            self.assertEqual(self.mapper.mapping[pd.Timestamp(date)], pd.Timestamp("2020-06-19"))

    def test_good_friday_holiday_has_wednesday_last_trade(self):
        self.assertEqual(self.mapper.mapping[pd.Timestamp("2022-04-13")], pd.Timestamp("2022-04-14"))
        self.assertNotIn(pd.Timestamp("2022-04-15"), self.mapper.sessions)

    def test_sole_alias_switches_without_losing_mark(self):
        frame = chain()
        frame["EXPIRE_DATE"] = pd.Timestamp("2020-06-18")
        day, _ = self.mapper.normalize(frame)
        cfg = Config(expiry_mode="monthly_aliases")
        pos, _ = build_candidate(self.mapper.entry_frame(day), pd.Timestamp("2020-02-19"), cfg)
        self.assertEqual(pos["source_expire_date"], pd.Timestamp("2020-06-18"))
        new_day, _ = self.mapper.normalize(chain("2020-02-20"))
        self.mapper.entry_frame(new_day)
        mark = self.mapper.mark(pos, new_day, new_day.set_index(["EXPIRE_DATE", "STRIKE"]), cfg)
        self.assertIsNotNone(mark)
        self.assertEqual(pos["source_label_switches"], 1)
        self.assertEqual(pos["last_source_expire_date"], pd.Timestamp("2020-06-19"))

    def test_simultaneous_roots_not_combined_but_existing_exact_root_kept(self):
        friday = chain()
        thursday = friday.copy()
        thursday["EXPIRE_DATE"] = pd.Timestamp("2020-06-18")
        cfg = Config(expiry_mode="monthly_aliases")
        pos, _ = build_candidate(friday, pd.Timestamp("2020-02-19"), cfg)
        day, audit = self.mapper.normalize(pd.concat([friday, thursday]))
        self.assertTrue(audit.ambiguous.all())
        self.assertTrue(self.mapper.entry_frame(day).empty)
        self.assertIsNotNone(self.mapper.mark(pos, day, day.set_index(["EXPIRE_DATE", "STRIKE"]), cfg))
        later, _ = self.mapper.normalize(thursday)
        self.assertIsNone(self.mapper.mark(pos, later, later.set_index(["EXPIRE_DATE", "STRIKE"]), cfg))


class ComparisonTests(unittest.TestCase):
    def test_profit_target_closes_every_leg_without_barrier(self):
        dates = pd.to_datetime(["2020-02-19", "2020-02-20", "2020-06-19"])
        mapper = MonthlyExpiryMap(dates[0], dates[-1])
        cfg = Config(initial_capital=1e6, exit_policy="barrier_close_all", profit_target_risk=.05, expiry_mode="monthly_aliases")
        engine = DigitalTailPortfolio(cfg, dates, mapper)
        for i, when in enumerate(dates[:2]):
            frame = chain(str(when.date()))
            if i:
                frame.loc[0, ["P_BID", "P_ASK"]] = [16, 16.1]
            day, _ = mapper.normalize(frame)
            engine.step(when, day, mapper.entry_frame(day), day.set_index(["EXPIRE_DATE", "STRIKE"]), 3386.16, {})
        self.assertEqual(len(engine.closed), 1)
        self.assertFalse(engine.active)
        trade = engine.closed[0]
        self.assertEqual(trade["exit_reason"], "package_profit_target")
        self.assertGreaterEqual(trade["pnl"], .05 * trade["risk_budget"])
        self.assertAlmostEqual(trade["pnl"], engine.realized_tail + engine.realized_spread)

    def test_terminal_missing_quote_stays_unresolved(self):
        dates = pd.to_datetime(["2020-02-19", "2020-06-18", "2020-06-19"])
        mapper = MonthlyExpiryMap(dates[0], dates[-1])
        cfg = Config(initial_capital=1e6, expiry_mode="monthly_aliases")
        engine = DigitalTailPortfolio(cfg, dates, mapper)
        for i, when in enumerate(dates):
            frame = chain(str(when.date()))
            if i:
                frame = frame.iloc[1:]
            day, _ = mapper.normalize(frame)
            engine.step(when, day, mapper.entry_frame(day), day.set_index(["EXPIRE_DATE", "STRIKE"]), 3386.16, {})
        summary, trades, _ = engine.finish()
        self.assertEqual(summary["unresolved_packages"], 1)
        self.assertEqual(summary["trades"], 0)
        self.assertTrue(pd.isna(trades.iloc[0].exit_date))

    def test_pending_barrier_closes_all_legs_when_quotes_recover(self):
        dates = pd.to_datetime(["2020-02-19", "2020-03-18", "2020-03-19", "2020-06-18", "2020-06-19"])
        mapper = MonthlyExpiryMap(dates[0], dates[-1])
        cfg = Config(initial_capital=1e6, exit_policy="barrier_close_all", expiry_mode="monthly_aliases")
        engine = DigitalTailPortfolio(cfg, dates, mapper)
        for when, spot, missing in [(dates[0], 3386.16, False), (dates[1], 2800, True), (dates[2], 3000, False)]:
            frame = chain(str(when.date()))
            frame["UNDERLYING_LAST"] = spot
            if missing:
                frame = frame.iloc[1:]
            day, _ = mapper.normalize(frame)
            engine.step(when, day, mapper.entry_frame(day), day.set_index(["EXPIRE_DATE", "STRIKE"]), spot, {})
        self.assertEqual(len(engine.closed), 1)
        trade = engine.closed[0]
        self.assertEqual(trade["exit_reason"], "barrier_close_all")
        self.assertEqual(trade["exit_signal_date"], dates[1])
        self.assertEqual(trade["exit_date"], dates[2])
        self.assertAlmostEqual(trade["pnl"], trade["tail_liq"] + trade["spread_liq"])
        self.assertEqual(trade["missing_mark_days"], 1)

    def test_batch_outputs_all_variants_and_reconciles_cashflows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry = chain()
            touch = chain("2020-03-18")
            touch["UNDERLYING_LAST"] = 2800
            entry.to_parquet(root / "spx_eod_202002.parquet")
            touch.to_parquet(root / "spx_eod_202003.parquet")
            pd.concat([chain("2020-06-18"), chain("2020-06-19")]).to_parquet(root / "spx_eod_202006.parquet")
            with patch("taleb_kurtosis.digital_tail_comparison.write_comparison_report"):
                result = run_comparison(root, root / "out", Config(initial_capital=1e6))
            self.assertEqual(len(result["summary"]), 12)
            summary = pd.read_csv(root / "out/summary.csv")
            trades = pd.read_parquet(root / "out/trades.parquet")
            equity = pd.read_parquet(root / "out/daily_equity.parquet")
            self.assertEqual(summary.unresolved_packages.sum(), 0)
            self.assertEqual(summary.strategy_id.nunique(), 12)
            np.testing.assert_allclose(equity.equity_pnl, equity.tail_pnl + equity.spread_pnl)
            for _, row in summary.iterrows():
                selected = trades[trades.strategy_id.eq(row.strategy_id)]
                self.assertAlmostEqual(row.total_pnl, selected.pnl.sum())
            barrier = trades[trades.exit_reason.eq("barrier_close_all")]
            self.assertTrue(len(barrier) > 0)
            self.assertTrue(barrier.exit_date.eq(pd.Timestamp("2020-03-18")).all())
            paired = pd.read_csv(root / "out/paired_exits.csv")
            self.assertGreater(len(paired), 0)


if __name__ == "__main__":
    unittest.main()
