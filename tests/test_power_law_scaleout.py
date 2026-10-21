import math
import unittest

import pandas as pd

from taleb_kurtosis.power_law_scaleout import ExitBook, crossed_level, sell_half, entry_parameters, update_atm_cache


class ScaleoutTests(unittest.TestCase):
    def setUp(self):
        self.entry = pd.Timestamp('2020-02-04')
        self.expiry = pd.Timestamp('2020-06-19')
        self.ticket = dict(package_id='test-1', entry_date=self.entry, signal_date=self.entry-pd.Timedelta(days=1),
                           expire_date=self.expiry, source_expire_date=self.expiry, strike=50.,
                           qty=8, entry_ask=1., entry_bid=.9, entry_cost=808., entry_spot=100.,
                           entry_atm_iv=.2, entry_dte=136, sigma_horizon=.1)

    def raw(self, spot=100., bid=.9, strike=50.):
        return pd.DataFrame(dict(EXPIRE_DATE=[self.expiry], SOURCE_EXPIRE_DATE=[self.expiry],
                                 MONTHLY_EXPIRY=[self.expiry], STRIKE=[strike], P_BID=[bid],
                                 P_ASK=[bid+.1], UNDERLYING_LAST=[spot]))

    def book(self, policy='scaleout_integer', qty=8):
        book = ExitBook('power_law', policy)
        ticket = dict(self.ticket, qty=qty, entry_cost=qty*101.)
        book.step(self.entry, self.raw(), set(), [ticket])
        return book

    def test_sigma_is_log_cumulative_frozen_horizon(self):
        self.assertEqual(crossed_level(100*math.exp(-.3), 100, .1), 3)
        self.assertEqual(crossed_level(100*math.exp(-.5), 100, .1), 5)
        self.assertEqual(crossed_level(110, 100, .1), 2)
        self.assertIsNone(crossed_level(0, 100, .1))
        self.assertIsNone(crossed_level(80, 100, float('nan')))

    def test_integer_floor_retains_one_contract(self):
        self.assertEqual(sell_half(7), 3)
        self.assertEqual(sell_half(1), 0)
        self.assertEqual(sell_half(1, fractional=True), .5)
        with self.assertRaises(ValueError):
            sell_half(1.5)

    def test_atm_cache_fallback_is_causal_bounded_and_audited(self):
        cache = {}
        chain = self.raw(strike=100).assign(P_IV=.2, P_DELTA=-.5)
        prior = self.entry - pd.Timedelta(days=1)
        update_atm_cache(chain, prior, cache)
        broken = chain.assign(P_IV=.00023, P_DELTA=-.75)
        update_atm_cache(broken, self.entry, cache)
        params = entry_parameters(self.ticket, pd.concat([self.raw(), broken]), set(), cache)
        self.assertEqual(params['entry_atm_iv'], .2)
        self.assertEqual(params['atm_quote_date'], prior)
        self.assertTrue(params['atm_iv_carried'])
        for day in [self.entry + pd.Timedelta(days=5), prior - pd.Timedelta(days=1)]:
            with self.assertRaises(ValueError):
                entry_parameters(dict(self.ticket, entry_date=day), self.raw(), set(), cache)
        update_atm_cache(chain.assign(P_IV=.25), self.entry, cache)
        params = entry_parameters(self.ticket, self.raw(), set(), cache)
        self.assertFalse(params['atm_iv_carried'])
        self.assertEqual(params['entry_atm_iv'], .25)

    def test_trigger_next_session_bid_and_rebound(self):
        b = self.book()
        b.step(self.entry+pd.Timedelta(days=1), self.raw(100*math.exp(-.3), 6), set())
        self.assertEqual(len(b.fills), 0)
        b.step(self.entry+pd.Timedelta(days=2), self.raw(100, 4), set())
        self.assertEqual(b.fills[0]['qty'], 4)
        self.assertEqual(b.fills[0]['bid'], 4)
        self.assertGreater(b.fills[0]['exit_date'], b.fills[0]['signal_date'])
        self.assertEqual(b.active[0]['remaining'], 4)

    def test_gap_crosses_all_levels_once(self):
        b = self.book()
        gap = self.raw(100*math.exp(-.5), 6)
        b.step(self.entry+pd.Timedelta(days=1), gap, set())
        b.step(self.entry+pd.Timedelta(days=2), gap, set())
        self.assertEqual([f['level'] for f in b.fills], [3,4,5])
        self.assertEqual([f['qty'] for f in b.fills], [4,2,1])
        self.assertEqual(b.active[0]['remaining'], 1)
        b.step(self.entry+pd.Timedelta(days=3), self.raw(), set())
        b.step(self.entry+pd.Timedelta(days=4), gap, set())
        self.assertEqual(len(b.signals), 3)
        self.assertEqual(len(b.fills), 3)

    def test_fractional_exact_percentages(self):
        b = self.book('scaleout_fractional', qty=1)
        gap = self.raw(100*math.exp(-.5), 6)
        b.step(self.entry+pd.Timedelta(days=1), gap, set())
        b.step(self.entry+pd.Timedelta(days=2), gap, set())
        self.assertEqual([f['qty'] for f in b.fills], [.5,.25,.125])
        self.assertEqual(b.active[0]['remaining'], .125)

    def test_single_lot_cannot_scaleout_but_triggers_audited(self):
        b = self.book(qty=1)
        gap = self.raw(100*math.exp(-.5), 6)
        b.step(self.entry+pd.Timedelta(days=1), gap, set())
        b.step(self.entry+pd.Timedelta(days=2), gap, set())
        self.assertEqual(b.active[0]['remaining'], 1)
        self.assertEqual(len(b.fills), 0)
        self.assertEqual(len(b.signals), 3)

    def test_missing_option_quote_still_detects_spot_trigger(self):
        b = self.book()
        b.step(self.entry+pd.Timedelta(days=1), self.raw(100*math.exp(-.3), strike=60), set())
        self.assertEqual(len(b.signals), 1)
        self.assertEqual(len(b.fills), 0)
        b.step(self.entry+pd.Timedelta(days=2), self.raw(strike=60), set())
        self.assertEqual(len(b.fills), 0)
        b.step(self.entry+pd.Timedelta(days=3), self.raw(bid=3), set())
        self.assertEqual(b.fills[0]['qty'], 4)
        self.assertEqual(b.fills[0]['bid'], 3)

    def test_cash_and_allocation_reconcile_on_residual_exit(self):
        b = self.book()
        b.step(self.entry+pd.Timedelta(days=1), self.raw(100*math.exp(-.3), 6), set())
        b.step(self.entry+pd.Timedelta(days=2), self.raw(bid=4), set())
        b.step(pd.Timestamp('2020-05-20'), self.raw(bid=.2), set())
        summary, trades, equity = b.finish()
        self.assertEqual(summary['unresolved'], 0)
        self.assertEqual(sum(f['qty'] for f in b.fills), 8)
        self.assertAlmostEqual(sum(f['pnl'] for f in b.fills), trades.pnl.sum())
        self.assertAlmostEqual(equity.equity_pnl.iloc[-1], trades.pnl.sum())
        self.assertEqual(trades.exit_reason.iloc[0], '30dte')

    def test_scheduled_exit_overrides_pending_levels_without_overselling(self):
        b = self.book()
        b.step(pd.Timestamp('2020-05-19'), self.raw(100*math.exp(-.5), 6), set())
        b.step(pd.Timestamp('2020-05-20'), self.raw(bid=4), set())
        self.assertEqual(len(b.fills), 1)
        self.assertEqual(b.fills[0]['reason'], '30dte')
        self.assertEqual(b.fills[0]['qty'], 8)

    def test_no_trigger_and_controls_hold_original_position(self):
        for policy in ['scaleout_integer','30dte','tp5']:
            b = self.book(policy)
            b.step(self.entry+pd.Timedelta(days=1), self.raw(100*math.exp(-.29), 4), set())
            b.step(self.entry+pd.Timedelta(days=2), self.raw(bid=2), set())
            self.assertFalse(b.fills)
            self.assertEqual(b.active[0]['remaining'], 8)

    def test_tp_control_sells_all_next_session(self):
        b = self.book('tp5')
        b.step(self.entry+pd.Timedelta(days=1), self.raw(bid=5), set())
        self.assertFalse(b.fills)
        b.step(self.entry+pd.Timedelta(days=2), self.raw(bid=3), set())
        self.assertEqual(b.fills[0]['reason'], 'tp5_next_session')
        self.assertEqual(b.fills[0]['qty'], 8)
        self.assertEqual(b.fills[0]['bid'], 3)

    def test_missing_final_quote_remains_unresolved(self):
        b = self.book()
        b.step(pd.Timestamp('2020-05-20'), self.raw(strike=60), set(), final=True)
        summary, trades, _ = b.finish()
        self.assertEqual(summary['unresolved'], 1)
        self.assertTrue(trades.pnl.isna().all())
        self.assertFalse(b.fills)


if __name__ == '__main__':
    unittest.main()
