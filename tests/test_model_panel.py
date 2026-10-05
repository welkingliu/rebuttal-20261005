import json
import unittest
from pathlib import Path

from sgg_core.models.panel import load_model_panel, panel_summary


class ModelPanelTest(unittest.TestCase):
    def test_panel_has_twenty_candidates_and_separate_execution_target(self):
        panel = load_model_panel()
        families = [item["family"] for item in panel["models"]]
        self.assertEqual(len(families), 20)
        self.assertEqual(len(set(families)), 20)
        self.assertEqual(panel["formal_target_families"], 5)
        self.assertEqual(panel["recommended_target_families"], 7)
        self.assertEqual(panel["survey_literature_target_families"], 10)

    def test_all_five_dataset_keys_are_present(self):
        coverage = panel_summary()["native_family_coverage"]
        self.assertEqual(set(coverage), {"vg", "oi", "gqa", "psg", "vrd"})
        self.assertEqual(coverage["gqa"], [])
        self.assertGreaterEqual(len(coverage["psg"]), 6)

    def test_every_family_has_an_install_environment(self):
        panel_families = {
            item["family"] for item in load_model_panel()["models"]
        }
        path = Path(__file__).resolve().parents[1] / "sgg_core" / "models" / "environment_catalog.json"
        catalog = json.loads(path.read_text(encoding="utf-8"))
        mapped = {
            family
            for environment in catalog["environments"]
            for family in environment["families"]
        }
        self.assertEqual(mapped, panel_families)


if __name__ == "__main__":
    unittest.main()
