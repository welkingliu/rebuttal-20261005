"""Read-only historical identity audit; writes a separate report, never a new gate."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def keyed(rows):
    result = {str(row["image_id"]): row for row in rows}
    if not rows or len(result) != len(rows):
        raise ValueError("Empty or duplicate image IDs")
    return result


def paired_counts(base, changed, count, correct, draws=10000):
    b, c = keyed(base), keyed(changed)
    if b.keys() != c.keys():
        raise ValueError("Unpaired image IDs")
    ids = sorted(b)
    n = np.array([b[i][count] for i in ids], dtype=np.int64)
    nc = np.array([c[i][count] for i in ids], dtype=np.int64)
    old = np.array([b[i][correct] for i in ids], dtype=np.int64)
    new = np.array([c[i][correct] for i in ids], dtype=np.int64)
    if not np.array_equal(n, nc) or (n < 0).any() or n.sum() == 0:
        raise ValueError("Changed or empty denominator")
    if ((old < 0) | (new < 0) | (old > n) | (new > n)).any():
        raise ValueError("Invalid correct count")
    rng = np.random.default_rng(17029)
    samples = []
    for _ in range(draws):
        ix = rng.integers(0, len(ids), len(ids))
        denominator = n[ix].sum()
        if denominator:
            samples.append((new[ix].sum() - old[ix].sum()) / denominator)
    delta = float((new.sum() - old.sum()) / n.sum())
    return dict(images=len(ids), denominator=int(n.sum()), native_correct=int(old.sum()),
                changed_correct=int(new.sum()), net_correct=int(new.sum() - old.sum()),
                native_accuracy=float(old.sum() / n.sum()), changed_accuracy=float(new.sum() / n.sum()),
                delta_fraction=delta, delta_percentage_points=100 * delta,
                relative_improvement_percent=100 * (new.sum() - old.sum()) / old.sum() if old.sum() else None,
                paired_image_95ci_pp=(100 * np.quantile(samples, [.025, .975])).tolist(),
                valid_bootstrap_draws=len(samples), requested_bootstrap_draws=draws,
                interval_scope="Descriptive, conditional on fitted/selected models; no selection or multiplicity correction")


def ece(confidence, correct):
    confidence, correct = np.asarray(confidence), np.asarray(correct)
    if not len(confidence) or confidence.shape != correct.shape:
        raise ValueError("Empty or unaligned calibration input")
    if not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
        raise ValueError("Invalid confidence")
    bin_id = np.minimum((confidence * 15).astype(int), 14)
    residual = np.bincount(bin_id, weights=confidence - correct, minlength=15)
    return float(np.abs(residual).sum() / len(confidence))


def foreground_confidence(logits):
    logits = np.asarray(logits, dtype=np.float64)
    exp = np.exp(logits - logits.max(1, keepdims=True))
    probability = exp / exp.sum(1, keepdims=True)
    conditional = probability[:, 1:] / probability[:, 1:].sum(1, keepdims=True)
    return probability[:, 1:].max(1), conditional.max(1)


def assert_close(actual, expected, name, tolerance=1e-10):
    if not np.isclose(actual, expected, atol=tolerance, rtol=0):
        raise RuntimeError("{} mismatch: {} vs {}".format(name, actual, expected))


def audit_vh(root, draws):
    import torch
    folder = root / "results/VH_independent_identity"
    rows, summary = read(folder / "development_images.json"), read(folder / "summary.json")
    checkpoint = root / "checkpoints/VH_independent_identity/linear_seed17.pth"
    state = torch.load(str(checkpoint), map_location="cpu")
    head = torch.nn.Linear(768, 150).eval()
    head.load_state_dict(state["head"])
    if state["protocol_sha256"] != summary["protocol_sha256"]:
        raise RuntimeError("V-H checkpoint protocol mismatch")
    indexed = {name: keyed(value) for name, value in rows.items()}
    replayed = 0
    for iid in indexed["native"]:
        f = torch.load(str(root / "cache/VH_independent_identity/development" / (iid + ".pt")), map_location="cpu")
        b = torch.load(str(root / "cache/VF/sgcls/development" / (iid + ".pt")), map_location="cpu")
        if str(f["image_id"]) != iid or str(b["image_id"]) != iid or not torch.equal(f["labels"], b["targets"]):
            raise RuntimeError("V-H feature/target alignment mismatch")
        if f["protocol_sha256"] != state["protocol_sha256"]:
            raise RuntimeError("V-H feature protocol mismatch")
        y, logits = f["labels"], b["baseline"]
        with torch.no_grad():
            expert = head(f["features"]).softmax(-1)
        prob = logits.softmax(-1)
        mass = prob[:, 1:].sum(-1, keepdim=True)
        conf, old_label = (prob[:, 1:] / mass.clamp_min(1e-12)).max(-1)
        expert_conf, expert_label = expert.max(-1)
        eligible = (conf < .7) & (expert_conf >= .5) & (old_label != expert_label)
        labels = {"native": old_label + 1, "expert_only": expert_label + 1}
        for alpha in [.25, .5]:
            mixed = (1 - alpha) * prob[:, 1:] + alpha * mass * expert
            labels["selective_alpha{:.2f}".format(alpha)] = torch.where(eligible, mixed.argmax(-1), old_label) + 1
        native_correct = labels["native"] == y
        for name, prediction in labels.items():
            correct = prediction == y
            record = indexed[name][iid]
            calculated = dict(objects=len(y), correct=int(correct.sum()), baseline_correct=int(native_correct.sum()),
                              repaired=int((~native_correct & correct).sum()), damaged=int((native_correct & ~correct).sum()),
                              label_flips=int((prediction != labels["native"]).sum()),
                              eligible=(0 if name == "native" else len(y) if name == "expert_only" else int(eligible.sum())),
                              class_count=torch.bincount(y, minlength=151)[1:].tolist(),
                              class_correct=torch.bincount(y[correct], minlength=151)[1:].tolist())
            for key, value in calculated.items():
                if record[key] != value:
                    raise RuntimeError("V-H replay mismatch: {} {} {}".format(iid, name, key))
            if record["repaired"] - record["damaged"] != record["correct"] - record["baseline_correct"]:
                raise RuntimeError("V-H repair/damage arithmetic mismatch")
        replayed += 1
    result = {}
    for name, records in rows.items():
        item = paired_counts(rows["native"], records, "objects", "correct", draws)
        assert_close(item["delta_fraction"], summary["results"][name]["delta"], "V-H delta")
        item.update(repairs=sum(r["repaired"] for r in records), damages=sum(r["damaged"] for r in records))
        result[name] = item
    print("V-H: replayed {} images and all four conditions".format(replayed), flush=True)
    return dict(results=result, cpu_replay_images=replayed, checkpoint_epoch=state["epoch"],
                checkpoint_sha256=sha(checkpoint), source_summary_sha256=sha(folder / "summary.json"),
                scope="GT-box development classification; this same development split selected the expert checkpoint",
                independent_confirmation=False, sgdet_evaluated=False)


def load_image_rows(folder):
    rows = [read(p) for p in sorted(folder.glob("*.json"))]
    keyed(rows)
    return rows


def macro50(rows):
    values = np.asarray([r["class_recalls"][4] for r in rows], dtype=float)
    count = np.isfinite(values).sum(0)
    if values.shape[1] != 50:
        raise ValueError("Wrong predicate ontology")
    return float(np.divide(np.nansum(values, 0), count, out=np.zeros(50), where=count > 0).mean())


def audit_vk(root, task, draws):
    import torch
    folder = root / "results/VK_native_confirmation" / task
    decision = read(folder / "decision.json")
    arms = {name: load_image_rows(folder / "gate" / name / "images") for name in ["native", "selective"]}
    indexed = {name: keyed(rows) for name, rows in arms.items()}
    if any(len(rows) != 1000 for rows in arms.values()):
        raise RuntimeError("Incomplete V-K gate")
    result = {stage: paired_counts(arms["native"], arms["selective"], "positive_objects", key, draws)
              for stage, key in [("pre_nms", "positive_correct"), ("post_nms", "post_nms_correct")]}
    assert_close(result["post_nms"]["delta_fraction"], decision["delta"]["object"], "V-K identity")
    result["relation_delta_pp"] = dict(
        R50=100 * np.mean([indexed["selective"][i]["recalls"][4] - indexed["native"][i]["recalls"][4] for i in indexed["native"]]),
        mR50=100 * (macro50(arms["selective"]) - macro50(arms["native"])))
    for key, value in result["relation_delta_pp"].items():
        assert_close(value / 100, decision["delta"][key], "V-K " + key)
    calibration = {name: {key: [] for key in ["full", "conditional", "pre_correct", "post", "post_correct"]} for name in arms}
    for pos, iid in enumerate(indexed["native"], 1):
        raw = torch.load(str(root / "cache/R19_vk_mechanism" / task / (iid + ".pt")), map_location="cpu")
        y = raw["target"].numpy()
        positive = y > 0
        for name, logit_name in [("native", "native_logits"), ("selective", "repaired_logits")]:
            row = indexed[name][iid]
            if row["protocol_sha256"] != decision["protocol_sha256"]:
                raise RuntimeError("V-K row protocol mismatch")
            logits = raw[logit_name].numpy()
            pre = logits[:, 1:].argmax(1) + 1
            with np.load(folder / "gate" / name / "predictions" / (iid + ".npz")) as data:
                scores = data["pred_entity_scores"]
                post = scores[:, 1:].argmax(1) + 1
                emitted = scores[:, 1:].max(1)
            if len(post) != len(y) or int(positive.sum()) != row["positive_objects"]:
                raise RuntimeError("V-K proposal count mismatch")
            for key, labels in [("positive_correct", pre), ("post_nms_correct", post)]:
                if int((labels[positive] == y[positive]).sum()) != row[key]:
                    raise RuntimeError("V-K cached prediction/count mismatch: {} {} {}".format(task, iid, key))
            full, conditional = foreground_confidence(logits[positive])
            c = calibration[name]
            for key, value in [("full", full), ("conditional", conditional), ("pre_correct", pre[positive] == y[positive]),
                               ("post", emitted[positive]), ("post_correct", post[positive] == y[positive])]:
                c[key].extend(value.tolist())
        if pos % 250 == 0:
            print("V-K {}: cached logits and outputs {}/1000".format(task, pos), flush=True)
    result["calibration"] = {}
    for name, values in calibration.items():
        old = decision["native" if name == "native" else "repaired"]["ece"]
        full = ece(values["full"], values["pre_correct"])
        assert_close(full, old, "Original ECE replay", tolerance=2e-6)
        result["calibration"][name] = dict(full_posterior_foreground_score_ece=full,
            foreground_conditional_top1_ece=ece(values["conditional"], values["pre_correct"]),
            emitted_label_score_ece=ece(values["post"], values["post_correct"]))
    result.update(original_gate_accepted=decision["accepted"], original_gate_checks=decision["checks"],
                  source_decision_sha256=sha(folder / "decision.json"), prediction_replay_images=1000,
                  calibration_scope="Positive fixed input proposals; not detection ECE/AP; supplemental variants do not replace the original gate")
    return result


def audit_unique(root, task, draws):
    folder = root / "results/R19_proposal_sensitivity" / task
    rows, previous = load_image_rows(folder / "images"), read(folder / "summary.json")
    result = {}
    for stage in ["pre", "post"]:
        b = [dict(image_id=r["image_id"], n=r["matched_unique_objects"], correct=r["unique_correct"][stage + "_native"]) for r in rows]
        c = [dict(image_id=r["image_id"], n=r["matched_unique_objects"], correct=r["unique_correct"][stage + "_repaired"]) for r in rows]
        result[stage] = paired_counts(b, c, "n", "correct", draws)
        assert_close(result[stage]["delta_fraction"], previous["unique_support_results"][stage]["delta"], "Unique support delta")
    for name in ["native_top50_endpoint", "not_native_top50_endpoint"]:
        b = [dict(image_id=r["image_id"], n=r["rank_strata"][name]["objects"], correct=r["rank_strata"][name]["correct_native"]) for r in rows]
        c = [dict(image_id=r["image_id"], n=r["rank_strata"][name]["objects"], correct=r["rank_strata"][name]["correct_repaired"]) for r in rows]
        result[name] = paired_counts(b, c, "n", "correct", draws)
    result["scope"] = "Reaggregation of stored class-agnostic unique-input-support matching and fixed-native-rank strata; not new detection matching"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite prior audit")
    if args.draws < 1000:
        raise ValueError("Use at least 1000 bootstrap draws")
    report = dict(status="running", audit_version="identity_metric_history_v1", source_sha256=sha(Path(__file__)),
                  bootstrap_seed=17029, selection_adjusted=False, changes_to_old_metrics_or_gate=False)
    report["VH"] = audit_vh(args.root, args.draws)
    report["VK"] = {task: audit_vk(args.root, task, args.draws) for task in ["sgcls", "sgdet"]}
    report["unique_support"] = {task: audit_unique(args.root, task, args.draws) for task in ["sgcls", "sgdet"]}
    report["status"] = "complete"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("Audit complete: {}".format(args.output), flush=True)


if __name__ == "__main__":
    main()
