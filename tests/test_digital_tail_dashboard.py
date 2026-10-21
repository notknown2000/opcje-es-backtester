import json
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


@unittest.skipUnless(Path("results/digital_tail_barrier_comparison_250k/summary.csv").exists(), "Comparison artifacts required")
class DigitalTailDashboardTests(unittest.TestCase):
    def assert_three_sigmas(self, app):
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertEqual(app.radio(key="digital_cmp_dimension").value, "Odległość spreadu")
        self.assertEqual(app.selectbox(key="digital_cmp_policy").value, "Bariera")
        spec = json.loads(app.get("plotly_chart")[0].proto.spec)
        self.assertEqual([trace["name"] for trace in spec["data"]],
                         ["2.5 sigma | Bariera", "3 sigma | Bariera", "3.5 sigma | Bariera"])

    def test_direct_link_opens_all_sigmas_and_does_not_override_later_choices(self):
        app = AppTest.from_file("taleb_kurtosis/dashboard.py", default_timeout=60)
        app.query_params["view"] = "digital-tail-250k"
        app.run()
        self.assertEqual(app.selectbox(key="preset_name").value, "Digital Tail / Bariery 250k")
        self.assert_three_sigmas(app)
        if Path("results/digital_tail_smile_diagnostics/manifest.json").exists():
            self.assertIn("Uśmiech IV", [tab.label for tab in app.tabs])
            specs = [json.loads(chart.proto.spec) for chart in app.get("plotly_chart")]
            self.assertTrue(any(len(spec["data"]) == 11 for spec in specs))
        app.radio(key="digital_cmp_dimension").set_value("Reguły wyjścia").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.radio(key="digital_cmp_dimension").value, "Reguły wyjścia")

    @unittest.skipUnless(Path("results/digital_tail_atm_iv_baseline/summary.csv").exists(), "Baseline artifacts required")
    def test_baseline_comparison_button_switches_run_and_chart_defaults(self):
        app = AppTest.from_file("taleb_kurtosis/dashboard.py", default_timeout=60)
        app.session_state["selected_results_dir"] = "results/digital_tail_atm_iv_baseline"
        app.run()
        app.button(key="open_digital_comparison").click().run()
        self.assertEqual(app.session_state["selected_results_dir"], "results/digital_tail_barrier_comparison_250k")
        self.assert_three_sigmas(app)


if __name__ == "__main__":
    unittest.main()
