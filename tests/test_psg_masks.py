import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from sgg_core.data.gqa_psg_data_utils import build_psg_loader


class PSGMaskLoaderTest(unittest.TestCase):
    def test_panoptic_segment_ids_become_node_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panoptic = root / "panoptic"
            panoptic.mkdir()
            ids = np.array([[10, 10], [11, 11]], dtype=np.int64)
            rgb = np.stack(
                (ids % 256, (ids // 256) % 256, (ids // 65536) % 256), axis=-1
            ).astype(np.uint8)
            Image.fromarray(rgb).save(panoptic / "one.png")
            annotation = root / "psg.json"
            annotation.write_text(json.dumps({
                "thing_classes": ["person", "car"],
                "predicate_classes": ["beside"],
                "data": [{
                    "image_id": 1,
                    "pan_seg_file_name": "one.png",
                    "width": 2,
                    "height": 2,
                    "segments_info": [
                        {"id": 10, "category_id": 0},
                        {"id": 11, "category_id": 1},
                    ],
                    "annotations": [
                        {"bbox": [0, 0, 2, 1], "bbox_mode": 0, "category_id": 0},
                        {"bbox": [0, 1, 2, 2], "bbox_mode": 0, "category_id": 1},
                    ],
                    "relations": [[0, 1, 0]],
                }],
            }))
            loader = build_psg_loader(
                str(annotation), num_samples=1, panoptic_root=str(panoptic)
            )
            batch = next(iter(loader))
            self.assertEqual(tuple(batch["masks"].shape), (2, 2, 2))
            self.assertEqual(int(batch["masks"][0].sum()), 2)
            self.assertEqual(int(batch["masks"][1].sum()), 2)


if __name__ == "__main__":
    unittest.main()
