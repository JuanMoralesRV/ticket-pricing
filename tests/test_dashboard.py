"""Regression checks for the default presentation and the promoter's live demo."""
import json
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=45)
        self.run_dashboard()

    def run_dashboard(self):
        self.app.run()
        self.assertEqual([error.message for error in self.app.exception], [])

    def with_label(self, elements, label):
        matches = [element for element in elements if element.label == label]
        self.assertEqual(len(matches), 1, f"Expected one dashboard element labeled {label!r}")
        return matches[0]

    def test_opening_view_reproduces_report(self):
        report = json.loads((ROOT / "resultados" / "resumen.json").read_text(encoding="utf-8"))
        self.assertEqual(
            [tab.label for tab in self.app.tabs], ["Calibration", "Evaluation (CRN)", "Episode trace"]
        )
        self.assertEqual(self.with_label(self.app.slider, "Evaluation scenarios (CRN)").value, 600)
        self.assertEqual(self.with_label(self.app.slider, "Fixed price (USD)").value, report["base"]["fixed"])
        table = self.app.tabs[1].dataframe[0].value.set_index("policy")
        policies = ["baseline", "pfa", "dla"]
        self.assertCountEqual(table.index, policies)
        for policy, stats in zip(policies, report["base"]["stats"], strict=True):
            with self.subTest(policy=policy):
                self.assertEqual(table.loc[policy, "n"], 600)
                self.assertAlmostEqual(table.loc[policy, "mean"], stats["media"], delta=0.0500001)

    def test_day_30_prices_respond_to_available_tickets(self):
        self.with_label(self.app.number_input, "Day of sale").set_value(30)
        self.with_label(self.app.number_input, "Reference price (USD)").set_value(100.0)
        for inventory, price in [(1600, "USD 90"), (1100, "USD 110")]:
            with self.subTest(inventory=inventory):
                self.with_label(self.app.number_input, "Available tickets").set_value(inventory)
                self.run_dashboard()
                self.assertEqual(self.with_label(self.app.metric, "Daily lookahead (DLA)").value, price)
