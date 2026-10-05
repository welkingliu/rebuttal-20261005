import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from sgg_core.models.prediction_cache import (
    OfficialPredictionCacheModel, sha256_file,
)
from sgg_core.models.prediction_cache_writer import OfficialPredictionCacheWriter


class PredictionCacheTest(unittest.TestCase):
    def test_round_trip_and_training_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "model.pth"
            checkpoint.write_bytes(b"official-checkpoint")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            cache = root / "cache"
            writer = OfficialPredictionCacheWriter(
                cache,
                model_name="official",
                architecture_family="Family",
                source_commit="a" * 40,
                parameter_count=123,
                checkpoint_sha256_by_task={"sgdet": digest},
                dataset="vg",
                ontology_id="vg:test",
                split="test",
                tasks=("sgdet",),
            )
            writer.add(
                "sgdet", 7,
                pred_rel_pairs=np.asarray([[0, 1]]),
                pred_rel_scores=np.asarray([[0.0, 2.0]]),
                pred_boxes=np.asarray([[0, 0, 1, 1], [0, 0, 1, 1]]),
                pred_entity_scores=np.asarray([[0, 2], [0, 2]]),
            )
            metadata_path = writer.finalize()
            manifest = {
                "name": "official",
                "architecture_family": "Family",
                "source_commit": "a" * 40,
                "supported_tasks": ["sgdet"],
                "diagnostic_task": "sgdet",
                "parameter_count": 123,
                "checkpoints": {
                    "sgdet": {"path": str(checkpoint), "sha256": digest}
                },
                "config": {
                    "prediction_cache_root": str(cache),
                    "prediction_cache_metadata_sha256": sha256_file(metadata_path),
                    "relation_score_mode": "independent_probabilities",
                },
            }
            model = OfficialPredictionCacheModel(manifest)
            output = model.predict_scene_graph({"image_id": 7}, "sgdet")
            self.assertEqual(tuple(output["pred_rel_scores"].shape), (1, 2))
            self.assertEqual(
                output["pred_rel_score_mode"], "independent_probabilities"
            )
            with self.assertRaises(RuntimeError):
                model.train(True)


if __name__ == "__main__":
    unittest.main()
