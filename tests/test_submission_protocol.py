import unittest

from sgg_core.submission_protocol import (
    EXTERNAL_DIAGNOSTIC_DATASETS,
    CORE_VG_MODEL_FAMILIES,
    DIAGNOSTIC_MODEL_FAMILIES,
    EXPERIMENT_2_FULL_LEVELS,
    EXPERIMENT_2_LIGHT_LEVELS,
    EXPERIMENT_3_DATASETS,
    EXPERIMENT_4_STEPS,
    GLOBAL_MODEL_FAMILY_TARGET,
    STANDARD_BENCHMARK_DATASETS,
    STANDARD_DATASET_FAMILY_TARGETS,
    STANDARD_TASK_FAMILY_TARGETS,
    parse_dataset_targets,
)


class SubmissionProtocolTest(unittest.TestCase):
    def test_submission_scope_is_three_standard_and_two_external_datasets(self):
        self.assertEqual(STANDARD_BENCHMARK_DATASETS, ("vg", "oi", "psg"))
        self.assertEqual(EXTERNAL_DIAGNOSTIC_DATASETS, ("gqa", "vrd"))
        self.assertEqual(GLOBAL_MODEL_FAMILY_TARGET, 5)
        self.assertEqual(len(CORE_VG_MODEL_FAMILIES), 5)
        self.assertEqual(len(DIAGNOSTIC_MODEL_FAMILIES), 2)
        self.assertEqual(
            STANDARD_DATASET_FAMILY_TARGETS,
            {"vg": 5, "oi": 2, "psg": 2},
        )
        self.assertEqual(
            STANDARD_TASK_FAMILY_TARGETS["vg"],
            {"predcls": 2, "sgcls": 2, "sgdet": 5},
        )

    def test_compute_bounded_experiment_profiles(self):
        self.assertEqual(len(EXPERIMENT_2_FULL_LEVELS), 5)
        self.assertEqual(EXPERIMENT_2_LIGHT_LEVELS, (0.0, 0.5, 1.0))
        self.assertEqual(EXPERIMENT_3_DATASETS, ("vg",))
        self.assertNotIn("pair", EXPERIMENT_4_STEPS)
        self.assertNotIn("graph", EXPERIMENT_4_STEPS)

    def test_dataset_target_parser_rejects_duplicates(self):
        self.assertEqual(
            parse_dataset_targets(["vg=6", "oi=4"], {}),
            {"vg": 6, "oi": 4},
        )
        with self.assertRaises(ValueError):
            parse_dataset_targets(["vg=6", "vg=4"], {})


if __name__ == "__main__":
    unittest.main()
