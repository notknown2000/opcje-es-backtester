import json
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from taleb_kurtosis.power_law_scaleout import POLICIES


@unittest.skipUnless(Path("results/power_law_scaleout/manifest.json").exists(), "Scaleout artifacts required")
class ScaleoutDashboardTests(unittest.TestCase):
    def app(self):
        app = AppTest.from_file("taleb_kurtosis/dashboard.py", default_timeout=60)
        app.query_params["view"] = "power-law-scaleout"
        return app.run()

    def assert_comparison(self, app):
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertEqual(app.selectbox(key="preset_name").value, "Power Law / Scaleout")
        chart = json.loads(app.get("plotly_chart")[0].proto.spec)
        self.assertEqual([t["name"] for t in chart["data"]], list(POLICIES.values()))
        self.assertEqual(len(app.dataframe[0].value), 4)

    def test_direct_link_has_all_policies_and_fractional_warning(self):
        app = self.app()
        self.assert_comparison(app)
        self.assertIn("teoretyczne", app.warning[0].value)
        self.assertEqual(app.selectbox(key="pl_scaleout_selection").value, "power_law")
        self.assertIn("Pakiety", [t.label for t in app.tabs])

    def test_selection_and_ledger_policy_remain_consistent(self):
        app = self.app()
        app.selectbox(key="pl_scaleout_selection").set_value("delta_1pct").run()
        self.assert_comparison(app)
        self.assertTrue(app.selectbox(key="pl_scaleout_package_delta_1pct").value.startswith("delta_1pct-"))
        app.selectbox(key="pl_scaleout_policy").set_value("scaleout_fractional").run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertEqual(app.selectbox(key="pl_scaleout_selection").value, "delta_1pct")


if __name__ == "__main__":
    unittest.main()
