import json
import tempfile
import unittest
from pathlib import Path

from sgg_core.tools.aggregate_model_shards import aggregate


class AggregateModelShardsTest(unittest.TestCase):
    def test_one_valid_shard_can_form_a_debug_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp:
            shard = Path(tmp) / "shards" / "motifs" / "vg"
            result_dir = shard / "vg"
            result_dir.mkdir(parents=True)
            checkpoint = {
                "sha256": "a" * 64,
                "parameter_count": 10,
                "metadata": {
                    "training_dataset": "VG-150",
                    "training_seed": 17,
                    "baseline_mR": 0.1,
                    "paradigm": "sequential_context",
                },
            }
            provenance = {
                "architecture_family": "Neural Motifs",
                "implementation_kind": "official_adapter",
                "checkpoint_status": checkpoint,
            }
            summary = {
                "datasets": ["vg"],
                "model_provenance": {"motifs_seed17": provenance},
            }
            results = {
                "model_provenance": {"motifs_seed17": provenance},
                "reproduction_validation": {"motifs_seed17": {"status": "pass"}},
                "standard_sgg": {"motifs_seed17": {"tasks": {"sgdet": {"metrics": {"R@50": 0.2, "mR@50": 0.1}}}}},
                "feature_audit": {"motifs_seed17": {"status": "ok", "effective_rank": 3.0}},
                "pair_audit": {"motifs_seed17": {"status": "ok", "BRR": 0.8}},
                "graph_audit": {"motifs_seed17": {"status": "ok", "MAR": 0.5}},
                "physical_consistency": {"motifs_seed17": {"PVR": 0.1, "coverage": 1.0}},
            }
            (shard / "summary.json").write_text(json.dumps(summary))
            (result_dir / "results.json").write_text(json.dumps(results))

            combined = aggregate(Path(tmp) / "shards", ["vg"], 1, 1)
            self.assertEqual(combined["status"], "formal_complete")
            self.assertEqual(combined["model_family_count"], 1)
            self.assertEqual(combined["dataset_family_counts"]["vg"], 1)

    def test_external_diagnostics_are_reused_without_rerunning_experiment_four(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "shards" / "motifs" / "vg"
            result_dir = shard / "vg"
            result_dir.mkdir(parents=True)
            checkpoint = {
                "sha256": "b" * 64,
                "parameter_count": 10,
                "metadata": {
                    "training_dataset": "VG-150",
                    "training_seed": 17,
                    "baseline_mR": 0.1,
                    "paradigm": "sequential_context",
                },
            }
            provenance = {
                "architecture_family": "Neural Motifs",
                "implementation_kind": "official_adapter",
                "checkpoint_status": checkpoint,
            }
            (shard / "summary.json").write_text(json.dumps({
                "datasets": ["vg"],
                "model_provenance": {"motifs": provenance},
            }))
            (result_dir / "results.json").write_text(json.dumps({
                "model_provenance": {"motifs": provenance},
                "reproduction_validation": {"motifs": {"status": "pass"}},
                "standard_sgg": {
                    "motifs": {"tasks": {"sgdet": {"metrics": {
                        "R@50": 0.2, "mR@50": 0.1,
                    }}}}
                },
            }))
            diagnostic = root / "experiment_2" / "vrd" / "motifs"
            diagnostic.mkdir(parents=True)
            (diagnostic / "experiment_2.json").write_text(json.dumps({
                "dataset": "vrd",
                "pair_audit": {"motifs": {"status": "ok", "BRR": 0.7}},
                "dose_response_and_controls": {"motifs": {"status": "ok"}},
                "physical_consistency": {"motifs": {"status": "ok", "PVR": 0.1}},
                "model_provenance": {"motifs": provenance},
            }))

            combined = aggregate(
                root / "shards", ["vg"], 1, {"vg": 1},
                diagnostic_roots=[root / "experiment_2"],
                diagnostic_targets={"vrd": 1},
            )
            self.assertEqual(combined["diagnostic_family_counts"]["vrd"], 1)
            rows = [row for row in combined["cross_dataset_rows"] if row["dataset"] == "vrd"]
            self.assertEqual(rows[0]["BRR@5"], 0.7)


if __name__ == "__main__":
    unittest.main()
