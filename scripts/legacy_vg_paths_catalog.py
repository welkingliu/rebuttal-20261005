"""Dataset catalog overlay for the pinned Kaihua SGG repository."""

import os


def _project_root():
    value = os.environ.get("SGG_PROJECT_ROOT")
    if not value:
        raise RuntimeError("SGG_PROJECT_ROOT is required")
    return os.path.abspath(os.path.expanduser(value))


class DatasetCatalog(object):
    @staticmethod
    def get(name, cfg):
        split = name.rsplit("_", 1)[-1]
        if split not in {"train", "val", "test"} or not name.startswith("VG_"):
            raise RuntimeError("Unsupported legacy dataset: {}".format(name))
        root = _project_root()
        native = os.path.join(root, "external", "official_repos", "PySGG", "datasets", "vg")
        args = {
            "img_dir": os.path.join(native, "stanford_spilt", "VG_100k_images"),
            "roidb_file": os.path.join(native, "VG-SGG-with-attri.h5"),
            "dict_file": os.path.join(native, "VG-SGG-dicts-with-attri.json"),
            "image_file": os.path.join(native, "image_data.json"),
            "split": split,
            "filter_non_overlap": (
                (not cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX)
                and cfg.MODEL.RELATION_ON
                and cfg.MODEL.ROI_RELATION_HEAD.REQUIRE_BOX_OVERLAP
            ),
            "filter_empty_rels": cfg.MODEL.RELATION_ON,
            "flip_aug": cfg.MODEL.FLIP_AUG,
            "custom_eval": cfg.TEST.CUSTUM_EVAL,
            "custom_path": cfg.TEST.CUSTUM_PATH,
        }
        eval_samples = int(os.environ.get("SGG_LEGACY_EVAL_SAMPLES", "-1"))
        if eval_samples > 0:
            args["num_im"] = eval_samples
        return {"factory": "VGDataset", "args": args}


class ModelCatalog(object):
    @staticmethod
    def get(name):
        raise RuntimeError("This evaluation must use a local MODEL.WEIGHT path: {}".format(name))
