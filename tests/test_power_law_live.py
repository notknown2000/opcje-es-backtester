import unittest
from unittest.mock import patch

import pandas as pd

from taleb_kurtosis.power_law_live import DataUnavailable, fetch_chain, rank_snapshot, validate_snapshot
from tests.test_power_law import chain


class LiveTests(unittest.TestCase):
    def setUp(self):
        self.now = pd.Timestamp('2026-10-07T12:00:00Z')
        q = chain().rename(columns=dict(STRIKE='strike', P_BID='bid', P_ASK='ask', P_IV='iv', P_DELTA='delta'))
        self.frame = q.assign(symbol='ES', option_series='EW', expiry='2027-02-04', underlying_contract='ESH7', underlying_price=100.,
                              underlying_quote_time=self.now, quote_time=self.now, multiplier=50,
                              exercise_style='European', market_data_type=1)

    def test_live_ranking_and_multiplier(self):
        ranked, audit = rank_snapshot(self.frame, 'ES', now=self.now)
        self.assertFalse(ranked.empty)
        row = ranked.iloc[0]
        self.assertAlmostEqual(row.cost_usd, row.P_ASK * 50 + 2)
        self.assertEqual(audit.reason.iloc[0], 'ok')
        micro = self.frame.assign(symbol='MES', multiplier=5)
        ranked, _ = rank_snapshot(micro, 'MES', now=self.now)
        self.assertAlmostEqual(ranked.cost_usd.iloc[0], ranked.P_ASK.iloc[0] * 5 + 2)

    def test_wrong_product_multiplier_and_duplicates(self):
        for frame in [self.frame.assign(symbol='SPY'), self.frame.assign(multiplier=100), pd.concat([self.frame, self.frame])]:
            with self.assertRaises(DataUnavailable):
                validate_snapshot(frame, 'ES', now=self.now)

    def test_stale_future_delayed_frozen_and_american_blocked(self):
        bad = [dict(quote_time=self.now-pd.Timedelta(minutes=3)), dict(underlying_quote_time=self.now-pd.Timedelta(minutes=3)),
               dict(quote_time=self.now+pd.Timedelta(minutes=3)), dict(market_data_type=2), dict(market_data_type=3),
               dict(market_data_type=4), dict(exercise_style='American'), dict(exercise_style='unknown'), dict(expiry='2025-01-01')]
        for update in bad:
            with self.subTest(update=update), self.assertRaises(DataUnavailable):
                validate_snapshot(self.frame.assign(**update), 'ES', now=self.now)

    def test_inconsistent_future_prices(self):
        frame = self.frame.copy()
        frame.loc[0, 'underlying_price'] = 110
        with self.assertRaises(DataUnavailable):
            rank_snapshot(frame, 'ES', now=self.now)

    def test_different_series_not_combined(self):
        frame = pd.concat([self.frame, self.frame.assign(option_series='E2', bid=self.frame.bid*.5, ask=self.frame.ask*.5)])
        ranked, audit = rank_snapshot(frame, 'ES', now=self.now)
        self.assertEqual(len(audit), 2)
        self.assertEqual(ranked.option_series.nunique(), 2)

    @patch('taleb_kurtosis.power_law_live.connect')
    def test_unknown_style_does_not_connect(self, connect):
        with self.assertRaises(DataUnavailable):
            fetch_chain('ES', 123, 'ES', '20270204')
        connect.assert_not_called()


if __name__ == '__main__':
    unittest.main()
