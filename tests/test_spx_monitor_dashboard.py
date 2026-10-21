import unittest
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from taleb_kurtosis.spx_monitor import FeedError, parse_snapshot
from tests.test_spx_monitor import payload


class MonitorDashboardTests(unittest.TestCase):
    def setUp(self):
        st.cache_data.clear()
        self.snapshot = parse_snapshot(payload(pd.Timestamp.now(tz="UTC")))

    def app(self):
        return AppTest.from_file(str(Path(__file__).resolve().parents[1] / "taleb_kurtosis/spx_monitor_dashboard.py"), default_timeout=40)

    def assert_clean(self, app):
        self.assertFalse(app.exception, [e.message for e in app.exception])

    def test_ranking_charts_and_filters(self):
        with patch("taleb_kurtosis.spx_monitor.fetch_snapshot", return_value=self.snapshot):
            app = self.app().run()
            self.assert_clean(app)
            self.assertEqual(len(app.get("plotly_chart")), 3)
            spec = json.loads(app.get("plotly_chart")[0].proto.spec)
            self.assertEqual(spec["layout"]["xaxis"]["type"], "category")
            self.assertTrue(len(app.dataframe[0].value))
            self.assertIn("opoznienie", app.warning[0].value)
            app.sidebar.selectbox[0].set_value("SPXW").run()
            self.assert_clean(app)
            self.assertTrue(any("Brak wycenionych" in message.value for message in app.info))

    def test_price_basis_and_alpha(self):
        with patch("taleb_kurtosis.spx_monitor.fetch_snapshot", return_value=self.snapshot):
            app = self.app().run()
            before = app.dataframe[0].value.set_index("option").discount
            app.get("segmented_control")[0].set_value("Bid / ask").run()
            self.assert_clean(app)
            after = app.dataframe[0].value.set_index("option").discount
            self.assertTrue((before.sort_index() >= after.sort_index()).all())
            app.number_input[0].set_value(3.5).run()
            self.assert_clean(app)

    def test_stale_block_and_explicit_historical_view(self):
        old = parse_snapshot(payload("2020-03-16T16:00:00Z"))
        with patch("taleb_kurtosis.spx_monitor.fetch_snapshot", return_value=old):
            app = self.app().run()
            self.assert_clean(app)
            self.assertTrue(app.error)
            self.assertFalse(app.dataframe)
            app.checkbox[0].check().run()
            self.assert_clean(app)
            self.assertTrue(app.dataframe)
            self.assertTrue(any("ARCHIWALNY" in message.value for message in app.warning))

    def test_api_failure_and_last_good_snapshot_label(self):
        with patch("taleb_kurtosis.spx_monitor.fetch_snapshot", side_effect=FeedError("Cboe HTTP 503")):
            app = self.app().run()
            self.assert_clean(app)
            self.assertIn("503", app.error[0].value)
            self.assertFalse(app.dataframe)
        st.cache_data.clear()
        with patch("taleb_kurtosis.spx_monitor.fetch_snapshot", return_value=self.snapshot):
            app.run()
            self.assert_clean(app)
            self.assertTrue(app.dataframe)
        st.cache_data.clear()
        with patch("taleb_kurtosis.spx_monitor.fetch_snapshot", side_effect=FeedError("Cboe HTTP 503")):
            app.run()
            self.assert_clean(app)
            self.assertTrue(any("Ostatni poprawny" in message.value for message in app.warning))
            self.assertTrue(app.dataframe)


if __name__ == "__main__":
    unittest.main()
