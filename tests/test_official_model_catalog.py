import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from official_model_catalog import CORE_MODELS, MANUAL_MODELS, MODELS
from prepare_official_models import _download_url, prepare_weight


class OfficialModelCatalogTest(unittest.TestCase):
    def test_default_download_panel_contains_only_live_sources(self):
        self.assertEqual(len(CORE_MODELS), 9)
        self.assertEqual(
            set(MANUAL_MODELS),
            {
                "bgnn_vg", "bgnn_oi", "sgtr_vg", "sgtr_oi",
                "openpsg_motifs_psg", "openpsg_vctree_psg",
                "openpsg_psgformer_psg", "motifs_vg", "grcnn_vg",
                "reldn_vg", "reldn_oi",
                "causal_motifs_sum_predcls_vg",
                "causal_motifs_sum_sgcls_vg",
                "causal_motifs_sum_sgdet_vg",
                "kern_sgcls_predcls_vg", "kern_sgdet_vg",
            },
        )
        for name in CORE_MODELS:
            self.assertEqual(
                MODELS[name].get("download_status", "available"), "available"
            )

    def test_submission_catalog_spans_ten_distinct_families(self):
        expected = {
            "Neural Motifs", "VCTree", "BGNN", "RelTR", "SGTR", "EGTR",
            "PSGTR", "PSGFormer", "PGSG", "OvSGTR",
        }
        architectures = {spec["architecture"] for spec in MODELS.values()}
        self.assertTrue(expected.issubset(architectures))

    def test_sgtr_catalog_points_to_extracted_checkpoint_and_config(self):
        for name in ("sgtr_vg", "sgtr_oi"):
            spec = MODELS[name]
            self.assertTrue(spec["relative_path"].endswith(".pth"))
            self.assertEqual(len(spec.get("required_paths", [])), 1)
            self.assertTrue(spec["required_paths"][0].endswith("config.json"))

    def test_huggingface_endpoint_rewrite_is_implemented(self):
        source = Path(SCRIPTS / "prepare_official_models.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('os.environ.get("HF_ENDPOINT"', source)
        self.assertTrue(callable(_download_url))

    def test_retired_url_requires_manual_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                FileNotFoundError, "manual verified source"
            ):
                prepare_weight(Path(tmp), "motifs_vg", verify_only=False)


if __name__ == "__main__":
    unittest.main()
