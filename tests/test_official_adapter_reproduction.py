import unittest

from sgg_core.models.official_adapter import OfficialSGGAdapter


class OfficialAdapterReproductionTest(unittest.TestCase):
    @staticmethod
    def _adapter():
        adapter = object.__new__(OfficialSGGAdapter)
        adapter.manifest = {
            "reference_dataset": "vg",
            "reference_eval_images": 100,
            "metric_scale": "fraction",
            "reproduction_tolerance": 0.02,
            "reference_metrics": {"SGDet/R@50": 0.25},
        }
        return adapter

    @staticmethod
    def _result(num_images):
        return {
            "tasks": {
                "sgdet": {
                    "num_images": num_images,
                    "metrics": {"R@50": 0.25},
                }
            }
        }

    def test_subset_cannot_pass_full_reference_contract(self):
        result = self._adapter().validate_reproduction(self._result(10), "vg")
        self.assertEqual(result["status"], "subset_diagnostic")
        self.assertEqual(result["expected_eval_images"], 100)
        self.assertEqual(result["observed_eval_images"], [10])

    def test_exact_reference_image_count_can_pass(self):
        result = self._adapter().validate_reproduction(self._result(100), "vg")
        self.assertEqual(result["status"], "pass")


if __name__ == "__main__":
    unittest.main()
