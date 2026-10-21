import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from taleb_kurtosis.power_law import PricingConfig, put_shape, relative_put_price, score_chain
from taleb_kurtosis.power_law_backtest import BacktestConfig, LongBook, aggregate_frequencies


def chain():
    strikes = np.arange(40., 101., 2.)
    prices = .1 * put_shape(np.minimum(strikes, 98), 100, 2.75)
    return pd.DataFrame(dict(STRIKE=strikes, P_BID=prices, P_ASK=prices + .02,
                             P_IV=.2, P_DELTA=np.where(strikes == 100, -.5, -.01)))


class PricingTests(unittest.TestCase):
    def test_integrated_bounded_pareto(self):
        for alpha in [1.01, 2, 2.75, 3.5]:
            integrals = []
            for strike in [30, 60, 80]:
                x = 1 - strike / 100
                r = np.linspace(x, 1, 100001)
                integrals.append(np.trapezoid((r-x) * r**(-alpha-1), r))
            np.testing.assert_allclose(put_shape([30, 60, 80], 100, alpha) / put_shape(80, 100, alpha),
                                       np.array(integrals) / integrals[-1], rtol=1e-7)

    def test_anchor_scale_zero_monotonic_convex(self):
        k = np.linspace(0, 80, 500)
        prices = relative_put_price(k, 100, 80, 2, 2.75)
        self.assertEqual(prices[0], 0)
        self.assertAlmostEqual(prices[-1], 2)
        self.assertTrue((np.diff(prices) > 0).all())
        self.assertTrue((np.diff(prices, 2) > 0).all())
        np.testing.assert_allclose(relative_put_price(k*50, 5000, 4000, 100, 2.75), prices*50)

    def test_invalid_domain_and_no_inward_extrapolation(self):
        for k in [-1, 100, float('nan')]:
            with self.assertRaises(ValueError):
                put_shape(k, 100, 2.75)
        with self.assertRaises(ValueError):
            relative_put_price([81], 100, 80, 1, 2.75)
        with self.assertRaises(ValueError):
            PricingConfig(alpha=1)

    def test_anchor_bid_and_purchase_costs(self):
        result, why = score_chain(chain(), 100, 120)
        self.assertEqual(why, 'ok')
        row = result.iloc[0]
        self.assertAlmostEqual(row.model_price, relative_put_price(row.STRIKE, 100, row.anchor_strike, row.anchor_bid, 2.75))
        self.assertAlmostEqual(row.all_in_ask, row.P_ASK + .02)
        self.assertLess(row.STRIKE, row.anchor_strike)

    def test_bad_quotes_excluded(self):
        q = chain()
        q.loc[0, 'P_BID'] = 0
        q.loc[1, 'P_ASK'] = .001
        q.loc[2, 'P_ASK'] = 100
        out, _ = score_chain(q, 100, 120)
        self.assertFalse(out.STRIKE.isin(q.STRIKE.iloc[:3]).any())

    def test_missing_anchor_not_extrapolated(self):
        out, why = score_chain(chain().query('STRIKE >= 90'), 100, 120)
        self.assertTrue(out.empty)
        self.assertEqual(why, 'no_liquid_anchor')

    def test_equal_day_frequencies(self):
        daily = pd.DataFrame(dict(QUOTE_DATE=['a','b'], sigma_bin=['3-3.5']*2, dte_bin=['105-135']*2,
                                  quotes=[1,100], frequency=[1.,0.], median_edge=[1.,0.],
                                  median_delta=[-.01]*2, robust_frequency=[1.,0.]))
        self.assertEqual(aggregate_frequencies(daily).frequency.iloc[0], .5)


class BookTests(unittest.TestCase):
    def setUp(self):
        self.cfg = BacktestConfig()
        self.book = LongBook('power_law', 5, self.cfg)
        self.expiry = pd.Timestamp('2020-06-19')
        self.signal = dict(signal_date=pd.Timestamp('2020-02-03'), expire_date=self.expiry,
                           source_expire_date=self.expiry, strike=100.)
        self.raw = pd.DataFrame(dict(EXPIRE_DATE=[self.expiry], SOURCE_EXPIRE_DATE=[self.expiry],
                                    MONTHLY_EXPIRY=[self.expiry], STRIKE=[100.], P_BID=[1.], P_ASK=[1.1]))

    def enter(self):
        self.book.pending = self.signal.copy()
        self.book.step(pd.Timestamp('2020-02-04'), self.raw, pd.DataFrame(), set())

    def test_ask_fill_integer_budget_and_cash(self):
        self.enter()
        p = self.book.active[0]
        self.assertLessEqual(p['entry_cost'] + p['qty'] * self.cfg.fee, 625)
        self.assertEqual(p['entry_ask'], 1.1)
        self.assertGreater(p['entry_date'], p['signal_date'])
        self.assertAlmostEqual(self.book.cash + p['entry_cost'], self.cfg.initial_capital)

    def test_profit_signal_exits_next_day_at_observed_bid(self):
        self.enter()
        jump = self.raw.assign(P_BID=6., P_ASK=6.1)
        self.book.step(pd.Timestamp('2020-02-05'), jump, pd.DataFrame(), set())
        self.assertEqual(len(self.book.closed), 0)
        self.book.step(pd.Timestamp('2020-02-06'), jump.assign(P_BID=4., P_ASK=4.1), pd.DataFrame(), set())
        self.assertEqual(self.book.closed[0]['exit_bid'], 4.)
        self.assertEqual(self.book.closed[0]['exit_reason'], 'tp5_next_session')

    def test_missing_exit_not_filled_or_settled(self):
        self.enter()
        self.book.step(pd.Timestamp('2020-05-22'), self.raw.iloc[0:0], pd.DataFrame(), set(), final=True)
        summary, trades, eq = self.book.finish()
        self.assertEqual(summary['unresolved'], 1)
        self.assertTrue(pd.isna(trades.pnl.iloc[0]))
        self.assertEqual(eq.stale_positions.iloc[-1], 1)

    def test_scheduled_exit_cash_reconciles(self):
        self.enter()
        self.book.step(pd.Timestamp('2020-05-22'), self.raw, pd.DataFrame(), set())
        summary, trades, eq = self.book.finish()
        self.assertEqual(summary['unresolved'], 0)
        self.assertAlmostEqual(eq.equity_pnl.iloc[-1], trades.pnl.sum())

    def test_no_buy_on_signal_day_without_pending(self):
        self.book.step(pd.Timestamp('2020-02-03'), self.raw, pd.DataFrame(), set())
        self.assertFalse(self.book.active)

    def test_no_entry_on_final_day(self):
        self.book.pending = self.signal.copy()
        self.book.step(pd.Timestamp('2020-02-04'), self.raw, pd.DataFrame(), set(), final=True)
        self.assertFalse(self.book.active)

    def test_execution_day_wide_spread_not_bought(self):
        self.book.pending = self.signal.copy()
        self.book.step(pd.Timestamp('2020-02-04'), self.raw.assign(P_ASK=10.), pd.DataFrame(), set())
        self.assertFalse(self.book.active)
        self.assertEqual(self.book.audit[0]['reason'], 'invalid_execution_spread')


if __name__ == '__main__':
    unittest.main()
