import copy
import unittest
from dataclasses import replace
from unittest.mock import patch
from urllib.error import HTTPError

import numpy as np
import pandas as pd

from taleb_kurtosis.power_law import PricingConfig, relative_put_price
from taleb_kurtosis.power_law_heatmap import scan_chain
from taleb_kurtosis.spx_monitor import (
    FeedError, evaluate_snapshot, fetch_snapshot, freshness, parse_snapshot,
)
from tests.test_power_law import chain


def payload(now=None):
    now = pd.Timestamp(now or "2026-10-21T16:00:00Z")
    expiry = (now + pd.Timedelta(days=180)).strftime("%y%m%d")
    options = []
    for _, row in chain().iterrows():
        options.append(dict(option=f"SPX{expiry}P{int(row.STRIKE * 1000):08d}", bid=row.P_BID,
                            ask=row.P_ASK, iv=row.P_IV, delta=row.P_DELTA))
    return dict(symbol="_SPX", timestamp=now.tz_localize(None).isoformat(), data=dict(
        symbol="^SPX", current_price=100.,
        last_trade_time=(now - pd.Timedelta(minutes=15)).tz_convert("America/New_York").tz_localize(None).isoformat(),
        options=options))


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.now = pd.Timestamp("2026-10-21T16:00:00Z")
        self.raw = payload()
        self.snapshot = parse_snapshot(self.raw, self.now)

    def test_identifiers_times_and_decimal_iv(self):
        self.assertEqual(self.snapshot.generated_at, self.now)
        self.assertEqual(self.snapshot.underlying_at, self.now - pd.Timedelta(minutes=15))
        self.assertEqual(self.snapshot.puts.STRIKE.iloc[0], 40)
        self.assertEqual(self.snapshot.puts.P_IV.iloc[0], .2)
        self.assertEqual(self.snapshot.puts.expiry.iloc[0], pd.Timestamp("2027-04-19"))

    def test_timezone_dst_winter_and_explicit_offsets(self):
        raw = payload("2026-12-21T16:00:00Z")
        self.assertEqual(raw["data"]["last_trade_time"], "2026-12-21T10:45:00")
        snapshot = parse_snapshot(raw)
        self.assertEqual(snapshot.underlying_at, pd.Timestamp("2026-12-21T15:45:00Z"))
        raw["data"]["last_trade_time"] = "2026-12-21T15:45:00Z"
        self.assertEqual(parse_snapshot(raw).underlying_at, snapshot.underlying_at)

    def test_missing_times_and_wrong_symbols_rejected(self):
        variants = [dict(symbol="SPY"), dict(timestamp=None), dict(timestamp="garbage")]
        for update in variants:
            with self.subTest(update=update), self.assertRaises(FeedError):
                parse_snapshot({**self.raw, **update})
        for field, value in [("current_price", np.nan), ("last_trade_time", None), ("options", [])]:
            raw = copy.deepcopy(self.raw)
            raw["data"][field] = value
            with self.subTest(field=field), self.assertRaises(FeedError):
                parse_snapshot(raw)

    def test_calls_and_other_roots_not_ranked(self):
        extra = dict(self.raw["data"]["options"][0])
        extra["option"] = "SPX270419C00040000"
        self.raw["data"]["options"].extend([extra, {**extra, "option": "SPY270419P00040000"}])
        snapshot = parse_snapshot(self.raw)
        self.assertEqual(len(snapshot.puts), len(self.snapshot.puts))
        self.assertEqual(snapshot.rejected_symbols, 1)

    def test_freshness_checks_generation_and_underlying(self):
        self.assertTrue(freshness(self.snapshot, self.now)[0])
        self.assertFalse(freshness(self.snapshot, self.now + pd.Timedelta(hours=2))[0])
        self.assertFalse(freshness(replace(self.snapshot, underlying_at=self.now-pd.Timedelta(days=1)), self.now)[0])
        self.assertFalse(freshness(replace(self.snapshot, generated_at=self.now+pd.Timedelta(minutes=2)), self.now)[0])
        self.assertFalse(freshness(replace(self.snapshot, underlying_at=self.now+pd.Timedelta(minutes=2)), self.now)[0])

    @patch("taleb_kurtosis.spx_monitor.urlopen")
    def test_api_failure_is_explicit(self, get):
        get.side_effect = HTTPError("https://example.com", 429, "Limit", {}, None)
        with self.assertRaisesRegex(FeedError, "429"):
            fetch_snapshot()

    def test_mid_anchor_and_costs(self):
        result, audit, _ = evaluate_snapshot(self.snapshot)
        self.assertFalse(result.empty)
        self.assertEqual(audit.reason.iloc[0], "ok")
        row = result.iloc[0]
        self.assertAlmostEqual(row.model_price, relative_put_price(row.STRIKE, 100, row.anchor_strike, row.anchor_mid, 3))
        self.assertAlmostEqual(row.discount, 1-(row.mid+.02)/row.model_price)
        self.assertAlmostEqual(row.cost_usd, row.mid*100+2)
        self.assertTrue(result.STRIKE.lt(result.anchor_strike).all())
        self.assertTrue(result.sigma.ge(3).all())
        self.assertTrue(result.data_type.eq("delayed_indicative").all())

    def test_bidask_same_sample_and_more_conservative(self):
        mid, _, _ = evaluate_snapshot(self.snapshot)
        conservative, _, _ = evaluate_snapshot(self.snapshot, basis="bid_ask")
        merged = mid.merge(conservative, on="option", suffixes=("_mid", "_ba"))
        self.assertEqual(len(merged), len(mid))
        self.assertTrue((merged.discount_mid >= merged.discount_ba).all())
        np.testing.assert_allclose(merged.bid_ask_discount_mid, merged.discount_ba)

    def test_exact_heatmap_reconciliation(self):
        result, _, _ = evaluate_snapshot(self.snapshot)
        when = pd.Timestamp("2026-10-21")
        q = self.snapshot.puts.assign(UNDERLYING_LAST=100.)
        records, _ = scan_chain(q, when, q.expiry.iloc[0], [3.],
                                PricingConfig(alpha=3, anchor_sigma=3, min_edge=0), "mid")
        bins = [3, 3.5, 4, 4.5, 5, 6, 7, 8, 10, 12, 15, np.inf]
        labels = ["3-3.5", "3.5-4", "4-4.5", "4.5-5", "5-6", "6-7", "7-8", "8-10", "10-12", "12-15", "15+"]
        result["bin"] = pd.cut(result.sigma, bins, labels=labels, right=False)
        for record in records:
            subset = result[result.bin.eq(record["sigma_bin"])]
            self.assertAlmostEqual(subset.discount.median(), record["discount"])

    def test_no_mixed_series_or_expiries(self):
        extra = []
        for original in self.raw["data"]["options"]:
            extra.append({**original, "option": original["option"].replace("SPX", "SPXW"),
                          "bid": original["bid"]*.5, "ask": original["ask"]*.5})
        self.raw["data"]["options"].extend(extra)
        result, audit, _ = evaluate_snapshot(parse_snapshot(self.raw))
        self.assertEqual(len(audit), 2)
        self.assertEqual(result.series.nunique(), 2)
        self.assertTrue(result.groupby("series").anchor_mid.first().loc["SPX"] > result.groupby("series").anchor_mid.first().loc["SPXW"])

    def test_bad_quotes_duplicates_and_atm(self):
        q = self.snapshot.puts.copy()
        q.loc[0, "P_BID"] = 0
        q.loc[1, "P_ASK"] = q.loc[1, "P_BID"] / 2
        q = pd.concat([q, q.iloc[[2]]])
        result, audit, _ = evaluate_snapshot(replace(self.snapshot, puts=q))
        self.assertFalse(result.STRIKE.isin([40., 42., 44.]).any())
        self.assertEqual(audit.duplicate_rows.iloc[0], 2)
        q.loc[q.STRIKE.eq(100), "P_IV"] = 0
        result, audit, _ = evaluate_snapshot(replace(self.snapshot, puts=q))
        self.assertTrue(result.empty)
        self.assertEqual(audit.reason.iloc[0], "invalid_atm")

    def test_missing_anchor_and_outside_tenor(self):
        result, audit, _ = evaluate_snapshot(replace(self.snapshot, puts=self.snapshot.puts.query("STRIKE >= 90")))
        self.assertTrue(result.empty)
        self.assertEqual(audit.reason.iloc[0], "no_liquid_anchor")
        result, audit, _ = evaluate_snapshot(self.snapshot, dte_min=30, dte_max=60)
        self.assertTrue(result.empty)
        self.assertTrue(audit.empty)

    def test_sensitivity_requires_all_models(self):
        result, _, _ = evaluate_snapshot(self.snapshot)
        cols = ["discount_alpha_2.5", "discount_alpha_3", "discount_alpha_3.5"]
        expected = result[cols].notna().all(axis=1) & result[cols].gt(0).all(axis=1)
        self.assertTrue(result.robust_underpriced.equals(expected))
        np.testing.assert_allclose(result.discount, result.discount_alpha_3)

    def test_invalid_parameters(self):
        for kwargs in [dict(alpha=1), dict(dte_min=200, dte_max=100), dict(basis="last"), dict(fee=-1)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                evaluate_snapshot(self.snapshot, **kwargs)


if __name__ == "__main__":
    unittest.main()
