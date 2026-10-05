#!/usr/bin/env python3
"""Apply the pinned PySGG PredCls evaluation compatibility fix."""

from __future__ import annotations

import argparse
from pathlib import Path


OLD = '''        if self.training:
            # relation subsamples and assign ground truth label during training
'''

NEW = '''        # PredCls proposals must expose their GT-derived object scores before
        # test-pair pruning. Upstream PySGG initialises these fields only after
        # prepare_test_pairs(), which raises KeyError when an image exceeds the
        # configured pair cap.
        if self.mode == "predcls":
            device = features[0].device
            for proposal in proposals:
                obj_labels = proposal.get_field("labels")
                proposal.add_field(
                    "predict_logits", to_onehot(obj_labels, self.num_obj_cls)
                )
                proposal.add_field(
                    "pred_scores", torch.ones(len(obj_labels), device=device)
                )
                proposal.add_field("pred_labels", obj_labels.to(device))

        if self.training:
            # relation subsamples and assign ground truth label during training
'''

LATE_BLOCK = '''
        if self.mode == "predcls":
            # overload the pred logits by the gt label
            device = features[0].device
            for proposal in proposals:
                obj_labels = proposal.get_field("labels")
                proposal.add_field("predict_logits", to_onehot(obj_labels, self.num_obj_cls))
                proposal.add_field("pred_scores", torch.ones(len(obj_labels)).to(device))
                proposal.add_field("pred_labels", obj_labels.to(device))
'''

PREDCLS_MARKER = "PredCls proposals must expose their GT-derived object scores before"

ROC_OLD = '''            fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
            auc = metrics.auc(fpr, tpr)
'''

ROC_NEW = '''            # Per-image proposal AUC is undefined when the selected range
            # contains only one label. sklearn returns NaN and emits one warning per
            # image; return the same undefined value explicitly so downstream code
            # keeps excluding it without flooding formal-run logs.
            if np.unique(y).size < 2:
                empty = np.asarray([], dtype=np.float32)
                return {
                    "fpr": empty,
                    "tpr": empty,
                    "thresholds": empty,
                    "auc": float("nan"),
                }

            fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
            auc = metrics.auc(fpr, tpr)
'''

ROC_MARKER = "Per-image proposal AUC is undefined"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    target = root / (
        "external/official_repos/PySGG/pysgg/modeling/roi_heads/"
        "relation_head/relation_head.py"
    )
    text = target.read_text(encoding="utf-8")
    updated = text
    changes = []

    if PREDCLS_MARKER not in updated:
        if updated.count(OLD) != 1 or updated.count(LATE_BLOCK) != 1:
            raise RuntimeError(
                "Pinned PySGG source does not match the expected PredCls code; "
                "refusing an unsafe patch"
            )
        updated = updated.replace(OLD, NEW, 1).replace(LATE_BLOCK, "", 1)
        changes.append("predcls")

    eval_target = root / (
        "external/official_repos/PySGG/pysgg/data/datasets/evaluation/"
        "vg/sgg_eval.py"
    )
    eval_text = eval_target.read_text(encoding="utf-8")
    updated_eval = eval_text
    if ROC_MARKER not in updated_eval:
        if updated_eval.count(ROC_OLD) != 1:
            raise RuntimeError(
                "Pinned PySGG source does not match the expected ROC code; "
                "refusing an unsafe patch"
            )
        updated_eval = updated_eval.replace(ROC_OLD, ROC_NEW, 1)
        changes.append("undefined_roc")

    if updated != text:
        target.write_text(updated, encoding="utf-8")
    if updated_eval != eval_text:
        eval_target.write_text(updated_eval, encoding="utf-8")
    if changes:
        print(f"[patched] {','.join(changes)}")
    else:
        print(f"[already-patched] {target.parent.parent.parent.parent.parent}")


if __name__ == "__main__":
    main()
