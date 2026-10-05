import unittest
from types import SimpleNamespace

from sgg_core.experiments.experiment_4 import _loader_args


class Experiment4LoaderArgsTest(unittest.TestCase):
    def test_psg_box_iou_run_does_not_require_panoptic_root(self):
        args = SimpleNamespace(
            psg_train_ann="/data/psg/train.json",
            psg_eval_ann="/data/psg/test.json",
            psg_image_root="/data/coco",
            psg_panoptic_root=None,
        )
        values = _loader_args(args, "psg")
        self.assertIsNone(values["panoptic_root"])

    def test_psg_still_requires_scene_graph_annotations(self):
        args = SimpleNamespace(
            psg_train_ann=None,
            psg_eval_ann="/data/psg/test.json",
            psg_image_root="/data/coco",
            psg_panoptic_root=None,
        )
        with self.assertRaisesRegex(ValueError, "train_ann"):
            _loader_args(args, "psg")


if __name__ == "__main__":
    unittest.main()
