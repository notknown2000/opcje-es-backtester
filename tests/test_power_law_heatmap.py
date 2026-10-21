import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from taleb_kurtosis.power_law import PricingConfig, prepare_chain, price_vector, put_shape
from taleb_kurtosis.power_law_heatmap import (
    DTE_LABELS, SIGMA_LABELS, daily_bins, dte_label, heatmap, matrix, run, scan_chain, summarize,
)


def chain(iv=.2, spot=100., dte=120, when=pd.Timestamp("2020-01-02")):
    strikes = np.arange(40., 100.1, .5)
    bid = .01 * put_shape(np.minimum(strikes, 98.), 100, 3.)
    return pd.DataFrame(dict(QUOTE_DATE=when, EXPIRE_DATE=when+pd.Timedelta(days=dte),
                             STRIKE=strikes, UNDERLYING_LAST=spot, P_BID=bid, P_ASK=bid*1.1,
                             P_IV=iv, P_DELTA=np.where(strikes == 100, -.5, -.01)))


class HeatmapTests(unittest.TestCase):
    def scan(self, q, alphas=(3.,)):
        return scan_chain(q, q.QUOTE_DATE.iloc[0], q.EXPIRE_DATE.iloc[0], alphas,
                          PricingConfig(alpha=3, anchor_sigma=3, fee=0))

    def test_kernel_discount_and_midpoint_semantics(self):
        rows, audit = self.scan(chain())
        self.assertEqual(audit[0]['reason'], 'ok')
        self.assertTrue(rows)
        self.assertLessEqual(abs(audit[0]['anchor_sigma']-3), .25)
        for row in rows:
            self.assertAlmostEqual(row['discount'], -.1)
            self.assertAlmostEqual(row['mid_discount'], 0)
            self.assertEqual(row['frequency'], 0)

    def test_green_when_deeper_options_cost_less_than_model(self):
        q = chain()
        _, audit = self.scan(q)
        deeper = q.STRIKE < audit[0]['anchor_strike'] - 1
        q.loc[deeper, ['P_BID', 'P_ASK']] *= .5
        rows, _ = self.scan(q)
        far = [r for r in rows if r['sigma_bin'] == '6-7'][0]
        self.assertAlmostEqual(far['discount'], .45)
        self.assertEqual(far['frequency'], 1)

    def test_mid_reprices_both_legs_keeps_fees_and_identical_sample(self):
        q = chain()
        _, original_audit = self.scan(q)
        q.loc[q.STRIKE < original_audit[0]['anchor_strike'] - 1, ['P_BID', 'P_ASK']] *= .5
        cfg = PricingConfig(alpha=3, anchor_sigma=3, fee=1)
        args = (q, q.QUOTE_DATE.iloc[0], q.EXPIRE_DATE.iloc[0], [3.], cfg)
        conservative, audit = scan_chain(*args)
        mid_rows, mid_audit = scan_chain(*args, price_basis='mid')
        pd.testing.assert_frame_equal(pd.DataFrame(audit), pd.DataFrame(mid_audit))
        prepared, _ = prepare_chain(q, 100., 120, cfg)
        model, anchor_index, _ = price_vector(prepared, 100., cfg)
        anchor = prepared.iloc[anchor_index]
        mid_model = model * (anchor.P_BID + anchor.P_ASK) / (2 * anchor.P_BID)
        cost = (prepared.P_BID.to_numpy() + prepared.P_ASK.to_numpy()) / 2 + .02
        selected = np.isfinite(model) & prepared.sigma.ge(6) & prepared.sigma.lt(7)
        expected_discount = 1 - cost[selected] / mid_model[selected]
        expected_edge = mid_model[selected] / cost[selected] - 1
        far = next(r for r in mid_rows if r['sigma_bin'] == '6-7')
        self.assertAlmostEqual(far['discount'], np.median(expected_discount))
        self.assertAlmostEqual(far['median_edge'], np.median(expected_edge))
        self.assertEqual(far['frequency'], np.mean(expected_discount > 0))
        self.assertEqual(far['frequency_10pct'], np.mean(expected_edge >= cfg.min_edge))
        for bid_ask, mid in zip(conservative, mid_rows):
            self.assertEqual(bid_ask['quotes'], mid['quotes'])
            self.assertAlmostEqual(bid_ask['mid_discount'], mid['mid_discount'])
            self.assertGreater(mid['discount'], bid_ask['discount'])
            self.assertLess(mid['discount'], mid['mid_discount'])
            self.assertGreaterEqual(mid['frequency'], bid_ask['frequency'])

    def test_invalid_price_basis_rejected(self):
        q = chain()
        with self.assertRaises(ValueError):
            scan_chain(q, q.QUOTE_DATE.iloc[0], q.EXPIRE_DATE.iloc[0], [3.], PricingConfig(), 'last')

    def test_mid_frequencies_are_recomputed_not_copied_from_bid_ask(self):
        for factor in [.96, .9]:
            with self.subTest(factor=factor):
                q = chain()
                _, audit = self.scan(q)
                q.loc[q.STRIKE < audit[0]['anchor_strike'] - 1, ['P_BID', 'P_ASK']] *= factor
                cfg = PricingConfig(alpha=3, anchor_sigma=3, fee=0)
                args = (q, q.QUOTE_DATE.iloc[0], q.EXPIRE_DATE.iloc[0], [3.], cfg)
                bid_ask = next(r for r in scan_chain(*args)[0] if r['sigma_bin'] == '6-7')
                mid = next(r for r in scan_chain(*args, price_basis='mid')[0] if r['sigma_bin'] == '6-7')
                self.assertAlmostEqual(mid['discount'], 1 - factor)
                self.assertEqual(mid['frequency'], 1)
                self.assertEqual(bid_ask['frequency'], int(factor < 1 / 1.1))
                self.assertEqual(mid['frequency_10pct'], int(factor < 1 / 1.1))
                self.assertEqual(bid_ask['frequency_10pct'], 0)

    def test_atm_iv_is_same_chain_and_same_date(self):
        _, a = self.scan(chain(iv=.2))
        _, b = self.scan(chain(iv=.3))
        self.assertEqual(a[0]['atm_iv'], .2)
        self.assertEqual(b[0]['atm_iv'], .3)
        self.assertLess(b[0]['anchor_strike'], a[0]['anchor_strike'])
        q = chain()
        q.loc[q.STRIKE.eq(100), 'P_IV'] = 0
        rows, audit = self.scan(q)
        self.assertFalse(rows)
        self.assertEqual(audit[0]['reason'], 'invalid_atm')

    def test_anchor_is_not_interpolated_and_no_inward_prices(self):
        q = chain()
        _, audit = self.scan(q)
        k = audit[0]['anchor_strike']
        rows, audit = self.scan(q[q.STRIKE.gt(k+10)])
        self.assertFalse(rows)
        self.assertEqual(audit[0]['reason'], 'no_liquid_anchor')
        rows, _ = self.scan(chain())
        self.assertTrue(all(r['sigma_bin'] in SIGMA_LABELS for r in rows))

    def test_expiry_day_and_inconsistent_underlying_are_rejected(self):
        rows, audit = self.scan(chain(dte=0))
        self.assertFalse(rows)
        self.assertEqual(audit[0]['reason'], 'nonpositive_dte')
        q = chain()
        q.loc[0, 'UNDERLYING_LAST'] = 110
        self.assertEqual(self.scan(q)[1][0]['reason'], 'inconsistent_spot')

    def test_lower_alpha_makes_deep_puts_relatively_more_valuable(self):
        rows, _ = self.scan(chain(), (2.5, 3., 3.5))
        values = {r['alpha']: r['discount'] for r in rows if r['sigma_bin'] == '6-7'}
        self.assertGreater(values[2.5], values[3.])
        self.assertGreater(values[3.], values[3.5])

    def test_dte_boundaries_cover_all_positive_terms(self):
        for dte, expected in [(0, None), (1, '1-7'), (7, '1-7'), (8, '8-14'), (30, '15-30'),
                              (31, '31-60'), (120, '91-120'), (121, '121-180'), (2200, '1096+')]:
            self.assertEqual(dte_label(dte), expected)

    def test_equal_chain_then_equal_session_weights_and_missing_mask(self):
        rows = []
        for day, expiry, count, value in [('2020-01-02', '2020-05-01', 1, 1.),
                                          ('2020-01-02', '2020-05-02', 1000, 0.),
                                          ('2021-01-02', '2021-05-01', 1, 0.)]:
            rows.append(dict(quote_date=pd.Timestamp(day), expiry=expiry, alpha=3., sigma_bin='6-7',
                             dte_bin='91-120', quotes=count, discount=value, mid_discount=value,
                             frequency=value, frequency_10pct=value, median_edge=value))
        daily = daily_bins(pd.DataFrame(rows))
        summary = summarize(daily, min_days=2, min_years=2)
        self.assertAlmostEqual(summary.frequency.iloc[0], .25)
        self.assertAlmostEqual(summary.median_discount.iloc[0], .25)
        self.assertEqual(summary.days.iloc[0], 2)
        self.assertTrue(summary.sufficient_coverage.iloc[0])
        self.assertEqual(matrix(summary, 3., 'median_discount').shape, (len(SIGMA_LABELS), len(DTE_LABELS)))
        sparse = summarize(daily, min_days=3, min_years=2)
        self.assertTrue(matrix(sparse, 3., 'median_discount').isna().all().all())
        self.assertEqual(matrix(sparse, 3., 'days', mask=False).loc['6-7', '91-120'], 2)
        figure = heatmap(summary, 3.)
        self.assertEqual(figure.data[0].colorscale[-1][1], '#087f4f')
        self.assertEqual(figure.layout.xaxis.type, 'category')
        self.assertEqual(figure.layout.yaxis.type, 'category')
        self.assertEqual(list(figure.layout.yaxis.categoryarray), SIGMA_LABELS)

    def test_complete_runner_outputs_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, output = root/'data', root/'results'
            data.mkdir()
            chain().to_parquet(data/'spx_eod_202001.parquet', index=False)
            result = run(data, output, min_days=1, min_years=1, workers=1)
            self.assertFalse(result.empty)
            for name in ['heatmap.png', 'heatmap_frequency.png', 'heatmap_coverage.png', 'report.html',
                         'summary.csv', 'yearly.csv', 'daily_bins.parquet', 'manifest.json', 'audit_counts.csv']:
                self.assertTrue((output/name).exists(), name)
            cached = run(data, output, min_days=1, min_years=1, workers=1, resume=True)
            pd.testing.assert_frame_equal(result, cached)
            with self.assertRaises(ValueError):
                run(data, output, alpha=2.75, resume=True, workers=1)
            mid_output = root/'mid'
            mid = run(data, mid_output, min_days=1, min_years=1, workers=1, price_basis='mid')
            self.assertTrue((mid.median_discount > result.median_discount).all())
            pd.testing.assert_frame_equal(mid[['days', 'chains', 'quotes']], result[['days', 'chains', 'quotes']])
            report = (mid_output/'report.html').read_text()
            self.assertIn('model z kotwicy mid', report)
            self.assertIn('Mid nie gwarantuje ceny wykonania', report)
            with self.assertRaises(ValueError):
                run(data, mid_output, min_days=1, min_years=1, resume=True, workers=1)


if __name__ == '__main__':
    unittest.main()
