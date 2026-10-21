from __future__ import annotations

import unittest

from datetime import date

from taleb_kurtosis.backtest import (
    Leg,
    OptionQuote,
    Trade,
    calculate_cashflows,
    determine_ratio_qty,
    long_put_is_itm,
    mark_trade,
    make_scaleout_component_trades,
    make_ratio_tail_candidates,
    nearest_atm_put_iv,
    should_market_drop_exit,
    should_close_scaleout_short_core,
    should_scaleout_component_exit,
    should_skip_scaleout_entry,
    should_exit_combined,
    should_exit_at_dte,
    sigma_from_annual_iv,
    sigma_horizon,
    target_strike,
)


class RatioCreditTests(unittest.TestCase):
    def test_ratio_qty_allows_zero_cost(self) -> None:
        self.assertEqual(determine_ratio_qty(21.0, 1.0, min_net_credit=0.0), 21)

    def test_ratio_qty_reduces_for_strict_credit(self) -> None:
        self.assertEqual(determine_ratio_qty(21.0, 1.0, min_net_credit=0.01), 20)

    def test_ratio_qty_rejects_invalid_long_price(self) -> None:
        self.assertEqual(determine_ratio_qty(21.0, 0.0), 0)

    def test_cashflow_signs(self) -> None:
        legs = [Leg("put", 90.0, 20, 1.0), Leg("put", 102.0, -1, 21.0)]
        credit, debit, net, long_debit, short_credit = calculate_cashflows(legs)
        self.assertEqual(credit, 2100.0)
        self.assertEqual(debit, 2000.0)
        self.assertEqual(net, 100.0)
        self.assertEqual(long_debit, 2000.0)
        self.assertEqual(short_credit, 2100.0)

    def test_calendar_dte_to_session_sigma_horizon(self) -> None:
        self.assertAlmostEqual(sigma_horizon(0.01, 60), 0.06436198997408842)

    def test_tail_put_strikes_are_monotonic(self) -> None:
        horizon = sigma_horizon(0.01, 60)
        strikes = [target_strike(4000.0, horizon, multiple, "down") for multiple in [3, 4, 5]]
        self.assertGreater(strikes[0], strikes[1])
        self.assertGreater(strikes[1], strikes[2])

    def test_sigma_from_annual_iv_uses_60_day_scaling(self) -> None:
        self.assertAlmostEqual(sigma_from_annual_iv(0.20, 60), 0.08108848540793832)

    def test_nearest_atm_put_iv_uses_nearest_put_strike(self) -> None:
        quotes = [
            OptionQuote(strike=3900.0, call_mid=None, put_mid=10.0, put_iv=0.24),
            OptionQuote(strike=4005.0, call_mid=None, put_mid=20.0, put_iv=0.21),
            OptionQuote(strike=4100.0, call_mid=None, put_mid=30.0, put_iv=0.19),
        ]
        self.assertAlmostEqual(nearest_atm_put_iv(quotes, 4000.0) or 0.0, 0.21)

    def test_ratio_tail_skips_unfunded_target(self) -> None:
        quotes = [
            OptionQuote(strike=3300.0, call_mid=None, put_mid=2.0),
            OptionQuote(strike=4050.0, call_mid=None, put_mid=21.0),
        ]
        candidates, skipped = make_ratio_tail_candidates(
            quotes=quotes,
            underlying=4000.0,
            horizon_sigma=0.05,
            short_sigma=0.4,
            long_sigma=3.0,
            ratio_targets=[10, 20],
            min_net_credit=0.0,
            max_long_qty=None,
        )
        target_ratios = {meta["ratio_target"]: legs[0].qty for legs, meta in candidates if meta["ratio_policy"] == "target"}
        skipped_targets = {row["ratio_target"] for row in skipped if row.get("skip_reason") == "insufficient_credit"}
        self.assertEqual(target_ratios[10], 10)
        self.assertIn(20, skipped_targets)

    def test_long_put_itm_exit_trigger(self) -> None:
        class DummyTrade:
            legs = [Leg("put", 3200.0, 20, 1.0), Leg("put", 4050.0, -1, 21.0)]

        self.assertFalse(long_put_is_itm(DummyTrade(), 3201.0))
        self.assertTrue(long_put_is_itm(DummyTrade(), 3200.0))
        self.assertTrue(long_put_is_itm(DummyTrade(), 3000.0))

    def test_exit_at_dte_threshold(self) -> None:
        class DummyTrade:
            expire_date = date(2020, 3, 20)

        self.assertFalse(should_exit_at_dte(DummyTrade(), date(2020, 2, 18), 30))
        self.assertTrue(should_exit_at_dte(DummyTrade(), date(2020, 2, 19), 30))
        self.assertTrue(should_exit_at_dte(DummyTrade(), date(2020, 3, 1), 30))

    def test_combined_exit_triggers_on_long_itm_or_dte(self) -> None:
        class DummyTrade:
            expire_date = date(2020, 3, 20)
            legs = [Leg("put", 3200.0, 20, 1.0), Leg("put", 4050.0, -1, 21.0)]

        self.assertFalse(should_exit_combined(DummyTrade(), date(2020, 2, 18), 3300.0, 30))
        self.assertTrue(should_exit_combined(DummyTrade(), date(2020, 2, 18), 3200.0, 30))
        self.assertTrue(should_exit_combined(DummyTrade(), date(2020, 2, 19), 3300.0, 30))

    def test_market_drop_exit_triggers_below_five_percent(self) -> None:
        self.assertTrue(should_market_drop_exit(-0.051, 0.05))
        self.assertTrue(should_market_drop_exit(-0.05, 0.05))

    def test_market_drop_exit_ignores_smaller_moves(self) -> None:
        self.assertFalse(should_market_drop_exit(-0.049, 0.05))
        self.assertFalse(should_market_drop_exit(0.01, 0.05))
        self.assertFalse(should_market_drop_exit(None, 0.05))

    def test_scaleout_entry_blocks_only_above_50_iv(self) -> None:
        self.assertTrue(should_skip_scaleout_entry(0.501, 0.50))
        self.assertFalse(should_skip_scaleout_entry(0.50, 0.50))
        self.assertFalse(should_skip_scaleout_entry(None, 0.50))

    def test_scaleout_component_builder_creates_five_components(self) -> None:
        base_trade = Trade(
            trade_id=1,
            construction="ratio_tail_put",
            variant="left_put",
            exit_policy="scaleout_longs_5pct_iv40_itm",
            sigma_window=252,
            entry_frequency="weekly",
            short_sigma=0.4,
            long_sigma=4.0,
            entry_date=date(2020, 1, 1),
            expire_date=date(2020, 3, 1),
            dte_entry=60.0,
            underlying_entry=100.0,
            sigma_daily=0.01,
            sigma_t=0.05,
            legs=[],
            entry_credit=0.0,
            entry_debit=0.0,
            net_entry_cashflow=0.0,
            long_debit=0.0,
            short_credit=0.0,
            strategy_id="strategy",
            long_put_moneyness=0.8,
            short_put_moneyness=1.02,
            long_mid=1.0,
            short_mid=21.0,
            long_put_iv_entry=0.25,
        )
        long_leg = Leg("put", 80.0, 20, 1.0)
        short_leg = Leg("put", 102.0, -1, 21.0)
        quotes = [OptionQuote(strike=80.0, call_mid=None, put_mid=1.0, put_iv=0.25)]
        components = make_scaleout_component_trades("pkg-1", 10, date(2020, 1, 1), base_trade, long_leg, short_leg, quotes)
        self.assertEqual(len(components), 5)
        self.assertEqual({trade.component_type for trade in components}, {"short_core", "long_drop5", "long_iv40", "long_itm", "long_residual"})
        self.assertEqual(sum(trade.total_long_qty for trade in components), 20)
        self.assertEqual(sum(trade.total_short_qty for trade in components), 1)

    def test_scaleout_component_exits_only_matching_bucket(self) -> None:
        class Args:
            market_drop_exit_pct = 0.05
            scaleout_exit_iv_threshold = 0.50

        trade = Trade(
            trade_id=1,
            construction="ratio_tail_put",
            variant="left_put",
            exit_policy="scaleout_longs_5pct_iv40_itm",
            sigma_window=252,
            entry_frequency="weekly",
            short_sigma=0.4,
            long_sigma=4.0,
            entry_date=date(2020, 1, 1),
            expire_date=date(2020, 3, 1),
            dte_entry=60.0,
            underlying_entry=100.0,
            sigma_daily=0.01,
            sigma_t=0.05,
            legs=[Leg("put", 80.0, 5, 1.0)],
            entry_credit=0.0,
            entry_debit=500.0,
            net_entry_cashflow=-500.0,
            long_debit=500.0,
            short_credit=0.0,
            component_type="long_iv40",
            last_long_put_iv=0.51,
        )
        self.assertEqual(
            should_scaleout_component_exit(trade, date(2020, 1, 2), 90.0, -0.01, Args()),
            "scaleout_iv40",
        )
        trade.component_type = "long_drop5"
        trade.last_long_put_iv = 0.10
        self.assertEqual(
            should_scaleout_component_exit(trade, date(2020, 1, 2), 90.0, -0.051, Args()),
            "scaleout_drop5",
        )
        trade.component_type = "long_itm"
        self.assertEqual(
            should_scaleout_component_exit(trade, date(2020, 1, 2), 80.0, -0.01, Args()),
            "scaleout_long_itm",
        )

    def test_close_short_scaleout_closes_short_core_after_first_trigger(self) -> None:
        trade = Trade(
            trade_id=1,
            construction="ratio_tail_put",
            variant="left_put",
            exit_policy="scaleout_longs_close_short_on_first_trigger",
            sigma_window=252,
            entry_frequency="weekly",
            short_sigma=0.4,
            long_sigma=4.0,
            entry_date=date(2020, 1, 1),
            expire_date=date(2020, 3, 1),
            dte_entry=60.0,
            underlying_entry=100.0,
            sigma_daily=0.01,
            sigma_t=0.05,
            legs=[Leg("put", 102.0, -1, 21.0)],
            entry_credit=2100.0,
            entry_debit=0.0,
            net_entry_cashflow=2100.0,
            long_debit=0.0,
            short_credit=2100.0,
            package_id="pkg-1",
            component_type="short_core",
        )
        self.assertFalse(should_close_scaleout_short_core(trade, set()))
        self.assertTrue(should_close_scaleout_short_core(trade, {"pkg-1"}))

    def test_original_scaleout_keeps_short_core_after_long_trigger(self) -> None:
        trade = Trade(
            trade_id=1,
            construction="ratio_tail_put",
            variant="left_put",
            exit_policy="scaleout_longs_5pct_iv40_itm",
            sigma_window=252,
            entry_frequency="weekly",
            short_sigma=0.4,
            long_sigma=4.0,
            entry_date=date(2020, 1, 1),
            expire_date=date(2020, 3, 1),
            dte_entry=60.0,
            underlying_entry=100.0,
            sigma_daily=0.01,
            sigma_t=0.05,
            legs=[Leg("put", 102.0, -1, 21.0)],
            entry_credit=2100.0,
            entry_debit=0.0,
            net_entry_cashflow=2100.0,
            long_debit=0.0,
            short_credit=2100.0,
            package_id="pkg-1",
            component_type="short_core",
        )
        self.assertFalse(should_close_scaleout_short_core(trade, {"pkg-1"}))

    def test_mark_trade_supports_staggered_expiries(self) -> None:
        trade = Trade(
            trade_id=1,
            construction="ratio_tail_put_staggered",
            variant="left_put",
            exit_policy="hold_to_expiry",
            sigma_window=252,
            entry_frequency="every_n_days",
            short_sigma=0.4,
            long_sigma=4.0,
            entry_date=date(2020, 1, 1),
            expire_date=date(2020, 3, 1),
            dte_entry=60.0,
            underlying_entry=100.0,
            sigma_daily=0.01,
            sigma_t=0.05,
            legs=[
                Leg("put", 80.0, 20, 1.0, expire_date=date(2020, 3, 1)),
                Leg("put", 103.0, -1, 21.0, expire_date=date(2020, 1, 31)),
            ],
            entry_credit=2100.0,
            entry_debit=2000.0,
            net_entry_cashflow=100.0,
            long_debit=2000.0,
            short_credit=2100.0,
        )
        chains = {
            date(2020, 3, 1): (30.0, [OptionQuote(strike=80.0, call_mid=None, put_mid=4.0)]),
        }
        self.assertTrue(mark_trade(trade, chains, date(2020, 2, 15), 90.0))
        self.assertIsNotNone(trade.last_pnl)


if __name__ == "__main__":
    unittest.main()
