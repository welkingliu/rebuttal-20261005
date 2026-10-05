import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from extract_official_weight_archives import prepare_one
from official_model_catalog import MODELS


class OfficialArchiveExtractionTest(unittest.TestCase):
    def test_extracts_only_runtime_checkpoint_and_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = MODELS["egtr_vg"]
            archive_path = root / spec["relative_path"]
            archive_path.parent.mkdir(parents=True)
            with tarfile.open(archive_path, "w:gz") as archive:
                checkpoint = b"x" * (1024 * 1024 + 1)
                checkpoint_info = tarfile.TarInfo(
                    "official/run/checkpoints/" + spec["checkpoint_member"]
                )
                checkpoint_info.size = len(checkpoint)
                archive.addfile(checkpoint_info, io.BytesIO(checkpoint))
                config = json.dumps({"model": "egtr"}).encode("utf-8")
                config_info = tarfile.TarInfo("official/run/config.json")
                config_info.size = len(config)
                archive.addfile(config_info, io.BytesIO(config))
                unrelated = b"not extracted"
                unrelated_info = tarfile.TarInfo("../../outside.txt")
                unrelated_info.size = len(unrelated)
                archive.addfile(unrelated_info, io.BytesIO(unrelated))

            payload = prepare_one(root, "egtr_vg", verify_only=False)
            self.assertTrue((root / spec["runtime_checkpoint"]).is_file())
            self.assertTrue((root / spec["runtime_config"]).is_file())
            self.assertFalse((root.parent / "outside.txt").exists())
            self.assertEqual(payload["model"], "egtr_vg")
            prepare_one(root, "egtr_vg", verify_only=True)


if __name__ == "__main__":
    unittest.main()
