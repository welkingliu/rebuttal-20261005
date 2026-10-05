import json
import unittest

from sgg_core.vg_ontology import assert_vg150_alignment, load_vg150_ontology


def _payload():
    return {
        "label_to_idx": {f"object-{index}": index for index in range(1, 151)},
        "predicate_to_idx": {f"predicate-{index}": index for index in range(1, 51)},
    }


class VG150OntologyTest(unittest.TestCase):
    def test_alignment_accepts_identical_order(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        with TemporaryDirectory() as directory:
            left = Path(directory) / "left.json"
            right = Path(directory) / "right.json"
            left.write_text(json.dumps(_payload()))
            right.write_text(json.dumps(_payload()))
            result = assert_vg150_alignment(left, right)
            self.assertEqual(result["status"], "aligned")
            self.assertEqual(
                load_vg150_ontology(left).object_classes[0], "__background__"
            )

    def test_alignment_rejects_reordered_predicate(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        with TemporaryDirectory() as directory:
            left = Path(directory) / "left.json"
            right = Path(directory) / "right.json"
            left.write_text(json.dumps(_payload()))
            altered = _payload()
            altered["predicate_to_idx"]["predicate-1"] = 2
            altered["predicate_to_idx"]["predicate-2"] = 1
            right.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "ontology order mismatch"):
                assert_vg150_alignment(left, right)


if __name__ == "__main__":
    unittest.main()
