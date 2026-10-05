"""R5: paired linear/MLP capacity sensitivity on identical cached SAM-mask views."""
import argparse
import json
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import ROOT, OLD, ensure_storage, atomic_json, output_path, sha256, sole_match
from sgg_core.audits.object_grounding import (
    LinearObjectProbe, deterministic_image_split, frequency_groups, batched_logits,
    fit_temperature, evaluate_object_logits, relationship_endpoint_summary,
    paired_accuracy_delta,
)


def train_probe(features, labels, train, val, num_classes, architecture, seed, config, output):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    width = features.shape[1]
    model = (LinearObjectProbe(width, num_classes) if architecture == "linear" else
             nn.Sequential(nn.Linear(width, 256), nn.ReLU(), nn.Linear(256, num_classes)))
    model = model.to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    indices = train.nonzero().flatten()
    best_loss, best, wait = float("inf"), None, 0
    batch = config["probe_batch_size"]
    history = []
    for epoch in range(100):
        started = time.monotonic()
        model.train()
        order = indices[torch.randperm(len(indices), generator=generator)]
        total = 0.
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            loss = F.cross_entropy(model(features[idx].cuda()), labels[idx].cuda())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        validation_logits = batched_logits(model, features[val], "cuda", batch)
        value = float(F.cross_entropy(validation_logits, labels[val]))
        if not np.isfinite(value):
            raise RuntimeError("Non-finite validation loss")
        improved = value < best_loss
        if improved:
            best_loss, wait = value, 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        row = dict(epoch=epoch + 1, train_cross_entropy=total / len(order),
                   validation_cross_entropy=value, improved=improved,
                   seconds=time.monotonic() - started)
        history.append(row)
        print(json.dumps(dict(architecture=architecture, seed=seed, **row)), flush=True)
        atomic_json(output / "progress.json", dict(status="training", architecture=architecture,
                    seed=seed, epoch=epoch + 1, best_validation_loss=best_loss))
        if wait >= 10:
            break
    model.load_state_dict(best)
    return model, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True, choices=["dinov2_b", "cradio_v4_so400m"])
    args = parser.parse_args()
    ensure_storage()
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    out = ROOT / "results/R5" / args.backbone
    out.mkdir(parents=True, exist_ok=True)
    train_path = sole_match("data/derived/features/experiment_1a/%s/*train_5000*.pt" % args.backbone)
    eval_path = sole_match("data/derived/features/experiment_1a/%s/*eval_1000*.pt" % args.backbone)
    summary_path = OLD / "artifacts/experiment_1a/exp1a_converged_20260718_174828" / args.backbone / "summary.json"
    source = json.loads(summary_path.read_text())
    config = source["config"]
    if config.get("feature_normalization", "none") != "none":
        raise ValueError("This replication requires the original unnormalized features")
    train = torch.load(train_path, map_location="cpu", weights_only=False)
    test = torch.load(eval_path, map_location="cpu", weights_only=False)
    if set(train["image_ids"]) & set(test["image_ids"]):
        raise ValueError("Train/test image overlap")
    labels, target = train["labels"].long(), test["labels"].long()
    mask, val = deterministic_image_split(train["image_ids"], config["validation_fraction"], seed=997)
    assert int(mask.sum()) == source["image_disjoint_probe_split"]["train_objects"]
    assert int(val.sum()) == source["image_disjoint_probe_split"]["validation_objects"]
    classes = source["num_classes"]
    groups = frequency_groups(labels[mask], classes)
    x, y = train["views"]["pred_mask"].float(), test["views"]["pred_mask"].float()
    assert len(target) == source["eval_objects"]
    protocol = dict(backbone=args.backbone, view="pred_mask", seeds=[17, 23, 31],
                    max_epochs=100, patience=10, hidden_dim=256, activation="ReLU",
                    feature_normalization="none", hyperparameter_selection="fixed_before_test",
                    train_cache_sha256=sha256(train_path), eval_cache_sha256=sha256(eval_path),
                    source_summary_sha256=sha256(summary_path), source_summary=str(summary_path),
                    train_objects=int(mask.sum()), validation_objects=int(val.sum()),
                    eval_objects=len(target), ontology_id=source["ontology_id"],
                    mask_protocol=source["predicted_mask_protocol"],
                    caveat="Sensitivity to probe capacity, not autonomous recognition")
    atomic_json(out / "protocol.json", protocol)
    predictions, runs = {}, []
    for seed in protocol["seeds"]:
        for architecture in ("linear", "mlp"):
            result_path = out / ("%s_seed%d.json" % (architecture, seed))
            pred_path = out / ("%s_seed%d_predictions.npz" % (architecture, seed))
            if result_path.exists() and pred_path.exists():
                record = json.loads(result_path.read_text())
                if record["protocol_sha256"] != sha256(out / "protocol.json"):
                    raise RuntimeError("Resume protocol mismatch")
                runs.append(record)
                predictions[(architecture, seed)] = torch.from_numpy(np.load(pred_path)["predictions"])
                continue
            model, history = train_probe(x, labels, mask, val, classes, architecture, seed, config, out)
            temperature = fit_temperature(batched_logits(model, x[val], "cuda", 512), labels[val])
            logits = batched_logits(model, y, "cuda", 512)
            prediction = logits.argmax(1)
            record = dict(seed=seed, architecture=architecture, history=history,
                          best_epoch=min(history, key=lambda row: row["validation_cross_entropy"])["epoch"],
                          reached_epoch_cap=len(history) == 100,
                          protocol_sha256=sha256(out / "protocol.json"), temperature=temperature,
                          parameter_count=sum(p.numel() for p in model.parameters()),
                          metrics=evaluate_object_logits(logits, target, test["image_ids"], groups,
                              areas=test["areas"], mask_iou=test["mask_iou"], temperature=temperature),
                          relationship_endpoints=relationship_endpoint_summary(prediction, target,
                              test["graph_records"], mask_iou=test["mask_iou"], bootstrap_seed=seed + 3000))
            torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},
                            protocol=protocol, architecture=architecture, temperature=temperature),
                       output_path(ROOT / "checkpoints/R5" / args.backbone / ("%s_seed%d.pt" % (architecture, seed))))
            np.savez_compressed(output_path(pred_path), logits=logits.numpy(), predictions=prediction.numpy(),
                                labels=target.numpy(), image_ids=np.asarray(test["image_ids"]))
            atomic_json(result_path, record)
            predictions[(architecture, seed)] = prediction
            runs.append(record)
            del model
    paired = {str(seed): paired_accuracy_delta(predictions[("linear", seed)], predictions[("mlp", seed)],
                  target, test["image_ids"], seed=seed + 1000) for seed in protocol["seeds"]}
    atomic_json(out / "summary.json", dict(status="complete", protocol=protocol, runs=runs,
                                          paired_mlp_minus_linear=paired))
    print("[COMPLETE] R5", args.backbone, flush=True)


if __name__ == "__main__":
    main()
