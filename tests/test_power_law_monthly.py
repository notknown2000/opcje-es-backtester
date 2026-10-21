import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from taleb_kurtosis.power_law import PricingConfig, put_shape
from taleb_kurtosis.power_law_monthly import (
    Book, Config, Experiment, daily_crash, quote_index, run, select_signal,
)


def chain(when=pd.Timestamp('2020-01-02'), expiry=pd.Timestamp('2020-07-02'), factor=.4):
    strikes = np.arange(5., 100.1, .5)
    bid = .3 * put_shape(np.minimum(strikes, 98.), 100., 3.)
    bid[strikes >= 90] = 2.
    sigma = np.log(100 / strikes) / (.2 * np.sqrt((expiry - when).days / 365))
    bid[(sigma >= 6) & (sigma <= 8)] *= factor
    return pd.DataFrame(dict(QUOTE_DATE=when, EXPIRE_DATE=expiry, STRIKE=strikes,
                             UNDERLYING_LAST=100., P_BID=bid, P_ASK=bid * 1.1,
                             P_IV=.2, P_DELTA=np.where(strikes == 100, -.5, -.01)))


def ticket():
    return dict(entry_id='test', signal_date=pd.Timestamp('2020-01-02'), entry_date=pd.Timestamp('2020-01-03'),
                expiry=pd.Timestamp('2020-07-02'), strike=40., qty=2, entry_mid=1., entry_cost=202.)


def book_quote(when, mid=1., strike=40.):
    return pd.DataFrame([dict(QUOTE_DATE=when, EXPIRE_DATE=pd.Timestamp('2020-07-02'), STRIKE=strike,
                              UNDERLYING_LAST=100., P_BID=mid * .9, P_ASK=mid * 1.1, P_IV=.2, P_DELTA=-.01)])


class MonthlyTests(unittest.TestCase):
    def test_selection_mid_anchor_global_ranking_and_band(self):
        cfg, pricing = Config(), PricingConfig(alpha=3, anchor_sigma=3, fee=1, min_edge=0)
        day = pd.Timestamp('2020-01-02')
        q = pd.concat([chain(day, day + pd.Timedelta(days=180), .4),
                       chain(day, day + pd.Timedelta(days=170), .2)], ignore_index=True)
        signal, audit = select_signal(q, day, cfg, pricing)
        self.assertTrue(signal)
        self.assertEqual(signal['signal_dte'], 170)
        self.assertTrue(6 <= signal['signal_sigma'] <= 8)
        self.assertGreater(signal['signal_discount'], 0)
        expected = signal['anchor_mid'] * put_shape(signal['strike'], 100., 3.) / put_shape(signal['anchor_strike'], 100., 3.)
        self.assertAlmostEqual(signal['signal_model_mid'], expected)
        self.assertAlmostEqual(signal['signal_discount'], 1 - (signal['signal_mid'] + .02) / expected)
        self.assertEqual(len(audit), 2)

    def test_overpriced_options_and_invalid_atm_produce_no_signal(self):
        day = pd.Timestamp('2020-01-02')
        pricing = PricingConfig(alpha=3, anchor_sigma=3)
        q = chain(factor=1.)
        self.assertIsNone(select_signal(q, day, Config(), pricing)[0])
        q = chain()
        q.loc[q.STRIKE.eq(100), 'P_IV'] = 0
        self.assertIsNone(select_signal(q, day, Config(), pricing)[0])

    def test_six_search_sessions_then_stop_and_reset_next_month(self):
        engine = Experiment()
        with patch('taleb_kurtosis.power_law_monthly.select_signal', return_value=(None, [])) as choose:
            for i, when in enumerate(pd.bdate_range('2020-01-02', periods=10), 1):
                engine.step(when, chain(when), i)
            self.assertEqual(choose.call_count, 6)
            self.assertEqual(sum(a['reason'] == 'month_skipped' for a in engine.entry_audit), 1)
            engine.step(pd.Timestamp('2020-02-03'), chain(pd.Timestamp('2020-02-03')), 1)
            self.assertEqual(choose.call_count, 7)

    def test_sixth_session_signal_fills_seventh_with_matched_integer_budget(self):
        engine = Experiment()
        days = pd.bdate_range('2020-01-02', periods=8)
        with patch('taleb_kurtosis.power_law_monthly.select_signal', return_value=(None, [])):
            for i, when in enumerate(days[:5], 1):
                engine.step(when, chain(when), i)
        engine.step(days[5], chain(days[5]), 6)
        self.assertIsNotNone(engine.pending)
        self.assertFalse(engine.entries)
        engine.step(days[6], chain(days[6]), 7)
        self.assertEqual(len(engine.entries), 1)
        entry = engine.entries[0]
        self.assertEqual(entry['signal_date'], days[5])
        self.assertEqual(entry['entry_date'], days[6])
        self.assertLessEqual(entry['entry_cost'] + entry['qty'], 1000.)
        self.assertIsInstance(entry['qty'], int)
        self.assertGreater(entry['qty'], 0)
        self.assertEqual(engine.books[0].active[0]['qty'], engine.books[1].active[0]['qty'])
        engine.step(days[7], chain(days[7]), 8)
        self.assertEqual(len(engine.entries), 1)

    def test_frozen_limit_rejects_higher_next_session_mid(self):
        engine = Experiment()
        when = pd.Timestamp('2020-01-02')
        engine.step(when, chain(when), 1)
        signal = engine.pending.copy()
        next_day = pd.Timestamp('2020-01-03')
        q = chain(next_day)
        mask = q.STRIKE.eq(signal['strike'])
        q.loc[mask, 'P_BID'] = signal['limit_mid'] * 1.1
        q.loc[mask, 'P_ASK'] = signal['limit_mid'] * 1.2
        engine.step(next_day, q, 2)
        self.assertFalse(engine.entries)
        self.assertTrue(any(a['reason'] == 'mid_above_frozen_model_limit' for a in engine.entry_audit))

    def test_daily_trigger_is_downside_strict_and_uses_prior_iv(self):
        sigma = .2 / np.sqrt(252)
        self.assertFalse(daily_crash(6 * sigma, .2)[0])
        self.assertFalse(daily_crash(-4.99 * sigma, .2)[0])
        self.assertTrue(daily_crash(-5.01 * sigma, .2)[0])
        self.assertFalse(daily_crash(-6 * sigma, 2.)[0])
        self.assertFalse(daily_crash(-6 * sigma, np.nan)[0])
        book = Book('crash5_or_30dte', Config())
        book.open(ticket(), pd.Timestamp('2020-01-03'))
        day = pd.Timestamp('2020-01-06')
        current = book_quote(day, 5.)
        current['P_IV'] = 2.
        previous = {ticket()['expiry']: dict(date=pd.Timestamp('2020-01-03'), iv=.2)}
        book.step(day, quote_index(current), -6 * sigma, previous, pd.Timestamp('2020-01-03'))
        self.assertFalse(book.closed)
        self.assertEqual(len(book.events), 1)
        book.step(pd.Timestamp('2020-01-07'), quote_index(book_quote(pd.Timestamp('2020-01-07'), 3.)), 0., {}, day)
        self.assertEqual(book.closed[0]['exit_mid'], 3.)
        self.assertEqual(book.closed[0]['exit_reason'], 'crash5_next_session')
        self.assertEqual(book.closed[0]['pnl'], 396.)

    def test_pending_crash_survives_missing_quote_and_bounce(self):
        book = Book('crash5_or_30dte', Config())
        book.open(ticket(), pd.Timestamp('2020-01-03'))
        first = pd.Timestamp('2020-01-06')
        prior = {ticket()['expiry']: dict(date=pd.Timestamp('2020-01-03'), iv=.2)}
        book.step(first, quote_index(book_quote(first, strike=41.)), -.1, prior, pd.Timestamp('2020-01-03'))
        book.step(pd.Timestamp('2020-01-07'), quote_index(book_quote(first, strike=41.)), .2, {}, first)
        self.assertFalse(book.closed)
        last = pd.Timestamp('2020-01-08')
        book.step(last, quote_index(book_quote(last, 2.)), 0., {}, pd.Timestamp('2020-01-07'))
        self.assertEqual(book.closed[0]['exit_signal_date'], first)
        self.assertEqual(book.closed[0]['exit_reason'], 'crash5_next_session')
        self.assertEqual(book.closed[0]['missing_mark_days'], 2)

    def test_control_ignores_crash_and_exits_at_30dte(self):
        book = Book('30dte', Config())
        book.open(ticket(), pd.Timestamp('2020-01-03'))
        day = pd.Timestamp('2020-01-06')
        previous = {ticket()['expiry']: dict(date=pd.Timestamp('2020-01-03'), iv=.2)}
        book.step(day, quote_index(book_quote(day)), -.2, previous, pd.Timestamp('2020-01-03'))
        self.assertFalse(book.closed)
        self.assertFalse(book.events)
        due = ticket()['expiry'] - pd.Timedelta(days=30)
        book.step(due, quote_index(book_quote(due, .5)), 0., {}, day)
        self.assertEqual(book.closed[0]['exit_reason'], '30dte')
        self.assertEqual(book.closed[0]['pnl'], -104.)

    def test_missing_source_session_prevents_multiday_return_trigger(self):
        engine = Experiment()
        first = pd.Timestamp('2020-01-02')
        engine.step(first, chain(first), 1)
        second = pd.Timestamp('2020-01-03')
        engine.step(second, chain(second), 2)
        missing = pd.Timestamp('2020-01-06')
        engine.step(missing, chain(missing).iloc[:0], 3)
        day = pd.Timestamp('2020-01-07')
        q = chain(day)
        q['UNDERLYING_LAST'] = 50.
        engine.step(day, q, 4)
        self.assertTrue(np.isnan(engine.market[-1]['daily_log_return']))
        self.assertFalse(engine.books[1].events)

    def test_no_fabricated_settlement_for_missing_expiry_quotes(self):
        book = Book('30dte', Config())
        book.open(ticket(), pd.Timestamp('2020-01-03'))
        day = ticket()['expiry']
        book.step(day, quote_index(book_quote(day)), 0., {}, day - pd.Timedelta(days=1), final=True)
        book.record_equity(day)
        summary, trades, equity = book.finish()
        self.assertEqual(summary['unresolved'], 1)
        self.assertEqual(summary['stale_sessions'], 1)
        self.assertTrue(np.isnan(trades.iloc[0].pnl))
        self.assertEqual(equity.equity_pnl.iloc[0], -4.)

    def test_missing_30dte_quote_delays_exit_without_resetting_instruction(self):
        book = Book('30dte', Config())
        book.open(ticket(), pd.Timestamp('2020-01-03'))
        due = ticket()['expiry'] - pd.Timedelta(days=30)
        book.step(due, quote_index(book_quote(due, strike=41.)), 0., {}, due - pd.Timedelta(days=1))
        self.assertFalse(book.closed)
        next_day = due + pd.Timedelta(days=1)
        book.step(next_day, quote_index(book_quote(next_day, .5)), 0., {}, due)
        self.assertEqual(book.closed[0]['exit_date'], next_day)
        self.assertEqual(book.closed[0]['exit_reason'], '30dte')

    def test_empty_entry_run_still_writes_readable_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data, output = root / 'data', root / 'out'
            data.mkdir()
            days = pd.bdate_range('2020-01-02', periods=3)
            pd.concat([chain(when, factor=1.) for when in days], ignore_index=True).to_parquet(data / 'spx_eod_202001.parquet', index=False)
            manifest = run(data, output)
            self.assertEqual(manifest['traded_months'], 0)
            self.assertTrue(pd.read_csv(output / 'entries.csv').empty)
            self.assertTrue(pd.read_csv(output / 'trades.csv').empty)
            self.assertTrue(pd.read_csv(output / 'summary.csv').total_pnl.eq(0).all())

    def test_complete_run_outputs_and_reconciled_accounting(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data, output = root / 'data', root / 'out'
            data.mkdir()
            days = pd.bdate_range('2020-01-02', periods=8)
            pd.concat([chain(when) for when in days], ignore_index=True).to_parquet(data / 'spx_eod_202001.parquet', index=False)
            manifest = run(data, output)
            self.assertEqual(manifest['traded_months'], 1)
            for name in ['report.html', 'performance.png', 'summary.csv', 'trades.csv', 'entries.csv',
                         'entry_audit.csv', 'scan_audit.csv', 'annual.csv', 'crash_events.csv', 'daily_equity.parquet']:
                self.assertTrue((output / name).exists(), name)
            trades = pd.read_csv(output / 'trades.csv')
            summary = pd.read_csv(output / 'summary.csv')
            self.assertTrue(trades.exit_reason.eq('end_of_data').all())
            for row in summary.itertuples():
                self.assertAlmostEqual(row.total_pnl, trades.loc[trades.strategy_id.eq(row.strategy_id), 'pnl'].sum())
            self.assertEqual(trades.entry_cost.nunique(), 1)
            with self.assertRaises(ValueError):
                run(data, output)


if __name__ == '__main__':
    unittest.main()
