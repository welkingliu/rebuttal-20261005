import numpy as np

from scripts.convert_legacy_vg_predictions import _kern_arrays, _validate_arrays


def test_kern_conversion_normalizes_boxes_and_preserves_51_scores():
    entry = {
        "pred_boxes": np.asarray([[0.0, 0.0, 1024.0, 512.0]], dtype=np.float32),
        "pred_classes": np.asarray([7]),
        "obj_scores": np.asarray([0.8], dtype=np.float32),
        "pred_rel_inds": np.zeros((0, 2), dtype=np.int64),
        "rel_scores": np.zeros((0, 51), dtype=np.float32),
    }
    arrays = _kern_arrays(entry, {"width": 1000, "height": 500})
    _validate_arrays(arrays)
    np.testing.assert_allclose(arrays["pred_boxes"], [[0.0, 0.0, 1.0, 1.0]])
    assert arrays["pred_entity_scores"].shape == (1, 151)
    assert arrays["pred_entity_scores"][0, 7] == np.float32(0.8)
