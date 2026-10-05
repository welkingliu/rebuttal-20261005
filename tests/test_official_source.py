import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from sgg_core.models.official_adapter import verify_official_source


class OfficialSourceTest(unittest.TestCase):
    def test_checksum_pinned_archive_is_accepted_without_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            archive = root / "model.tar.gz"
            archive.write_bytes(b"official commit archive")
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            commit = "a" * 40
            repository = "https://github.com/example/model.git"
            marker = {
                "source_type": "official_commit_archive",
                "repository_url": repository,
                "commit": commit,
                "archive_path": str(archive),
                "archive_sha256": digest,
            }
            (source / ".official_source.json").write_text(json.dumps(marker))

            result = verify_official_source(source, repository, commit)
            self.assertEqual(result["source_type"], "official_commit_archive")
            self.assertEqual(result["archive_sha256"], digest)

    def test_archive_checksum_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            archive = root / "model.tar.gz"
            archive.write_bytes(b"changed")
            marker = {
                "repository_url": "https://github.com/example/model.git",
                "commit": "a" * 40,
                "archive_path": str(archive),
                "archive_sha256": "0" * 64,
            }
            (source / ".official_source.json").write_text(json.dumps(marker))
            with self.assertRaises(RuntimeError):
                verify_official_source(
                    source, "https://github.com/example/model", "a" * 40
                )


if __name__ == "__main__":
    unittest.main()
