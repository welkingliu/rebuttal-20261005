"""One SGDet test pass after the fixed, reference-defined all-pair validation audit."""
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from common import ROOT, OLD, ensure_storage, atomic_json, output_path, sha256
from native_runtime import assets, image_id, infer
from repro_experiment import build, dataset, recall_row, KS
from repro_protocol import RUN, immutable, verify
from reproduction_pair_audit import ChunkUnion

SOURCE = Path(__file__).resolve()
OUT = ROOT / "results/R14_reference_sgdet_test"
VALIDATION = ROOT / "results/R14_pair_cap/sgdet"


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


class NonZeroShot:
    def __init__(self, official):
        self.official = official

    def __contains__(self, item):
        subject, predicate, obj = item
        return (subject, obj, predicate) not in self.official


def main():
    ensure_storage()
    torch.set_num_threads(4)
    state = ROOT / "status/R14_reference_sgdet_test.json"
    record = dict(status="waiting_dependency", pid=os.getpid(), gpu=[1], command=[str(SOURCE)],
                  completion=str(OUT / "summary.json"), progress_file=str(OUT / "progress.json"),
                  log=str(ROOT / "logs/R14_reference_sgdet_test.log"))
    names = ["evaluate_reference_pairs.py", "reproduction_pair_audit.py", "repro_experiment.py", "repro_protocol.py", "native_runtime.py", "common.py"]
    sources = {n: sha256(ROOT / "code" / n) for n in names}
    spec = dict(task="sgdet", checkpoint=str(RUN / "sgdet/training/model_final.pth"),
                checkpoint_sha256=sha256(RUN / "sgdet/training/model_final.pth"),
                registration="before the full validation audit finishes", sources=sources,
                rule="all non-self pairs, as documented reference source; no search over pair counts",
                gate="5000 validation images; capped control equals R9; unchanged shared logits, object outputs and boxes",
                metric_gate="not used to select test settings; run after integrity checks regardless of validation gain",
                official_zero_shot=True, reference_absolute_tolerance=.02)
    immutable(ROOT / "manifests/R14_reference_test.json", spec)
    atomic_json(state, record)
    while read(VALIDATION / "summary.json").get("status") != "complete":
        previous = read(ROOT / "status/R14_pair_cap_sgdet.json")
        if previous.get("status") in ["failed", "blocked"]:
            atomic_json(state, dict(record, status="blocked", reason="validation_integrity_audit_failed"))
            return
        if previous.get("status") == "running":
            try:
                os.kill(previous["pid"], 0)
            except OSError:
                atomic_json(state, dict(record, status="blocked", reason="validation_process_disappeared"))
                return
        time.sleep(20)
    val = read(VALIDATION / "summary.json")
    if val["images"] != 5000 or val["max_shared_probability_error"] > 1e-4:
        raise RuntimeError("Validation integrity contract failed")
    lock = (ROOT / "status/gpu1.resource.lock").open("a")
    atomic_json(state, dict(record, status="waiting_gpu"))
    fcntl.flock(lock, fcntl.LOCK_EX)
    atomic_json(state, dict(record, status="running"))
    try:
        verify()
        if sources != {n: sha256(ROOT / "code" / n) for n in names}:
            raise RuntimeError("Code changed after registering the test protocol")
        model, cfg, transform, provenance = build("sgdet", OUT, weights=Path(spec["checkpoint"]))
        if provenance["checkpoint_sha256"] != spec["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint changed")
        model.eval()
        ds = dataset(cfg, "test")
        if len(ds) != 26446:
            raise RuntimeError("Wrong test cardinality")
        zero_path = OLD / "external/official_repos/Scene-Graph-Benchmark.pytorch/maskrcnn_benchmark/data/datasets/evaluation/vg/zeroshot_triplet.pytorch"
        zero = {tuple(map(int, x)) for x in torch.load(str(zero_path), map_location="cpu").tolist()}
        seen = NonZeroShot(zero)
        pf = OUT / "protocol.json"
        immutable(pf, dict(spec=spec, validation_sha256=sha256(VALIDATION / "summary.json"),
                           official_zero_sha256=sha256(zero_path), model=provenance,
                           sampler_runtime_override="max_proposal_pairs=1000000 (no pruning for <=80 objects)",
                           union_chunk=256, image_ids=[image_id(ds, i) for i in range(len(ds))]))
        digest = sha256(pf)
        model.roi_heads.relation.samp_processor.max_proposal_pairs = 1000000
        chunk = ChunkUnion(model.roi_heads.relation.union_feature_extractor, 256)
        from export_pysgg_vg_task import convert_prediction
        rows, computed = [], 0
        started = time.monotonic()
        for index in range(len(ds)):
            iid = image_id(ds, index)
            path = OUT / "images" / (iid + ".json")
            cache = OUT / "predictions" / (iid + ".npz")
            if path.exists() and cache.exists():
                row = read(path)
                if row.get("protocol_sha256") != digest:
                    raise RuntimeError("Stale full-test row")
                rows.append(row)
                continue
            pred, _ = infer(model, cfg, transform, ds, index)
            row = recall_row(pred, ds.get_groundtruth(index, evaluation=True), "sgdet", seen)
            row.update(image_id=iid, protocol_sha256=digest)
            temp = output_path(cache.with_suffix(".tmp"))
            with temp.open("wb") as stream:
                np.savez_compressed(stream, **convert_prediction(pred.to("cpu")))
            temp.replace(cache)
            atomic_json(path, row)
            rows.append(row)
            computed += 1
            if computed == 1 or (index + 1) % 50 == 0 or index + 1 == len(ds):
                elapsed = time.monotonic() - started
                p = dict(images=index + 1, total=len(ds), seconds=elapsed,
                         eta_seconds=elapsed / computed * (len(ds) - index - 1),
                         detail="official all-pair and zero-shot rules; unchanged fixed R9 checkpoint")
                atomic_json(OUT / "progress.json", p)
                print(json.dumps(p), flush=True)
        chunk.close()
        r = np.asarray([x["R"] for x in rows]).mean(0)
        c = np.asarray([x["class_recall"] for x in rows], dtype=float)
        count = np.isfinite(c).sum(0)
        mr = np.divide(np.nansum(c, 0), count, out=np.zeros_like(count, dtype=float), where=count > 0).mean(1)
        z = np.asarray([x["zR"] for x in rows], dtype=float)
        count = np.isfinite(z).sum(0)
        zr = [float(np.nansum(z[:, j]) / count[j]) if count[j] else None for j in range(len(KS))]
        checks = {}
        for name, ref in assets("transformer", "sgdet")[0]["reference_metrics"].items():
            if not name.lower().startswith("sgdet/"):
                continue
            metric = name.split("/")[1]
            k = KS.index(int(metric.split("@")[1]))
            value = float(mr[k] if metric.startswith("mR") else r[k])
            checks[name] = dict(actual=value, reference=ref, delta=value - ref, passes=abs(value - ref) <= .02)
        if not checks:
            raise RuntimeError("Missing reference checks")
        passed = all(v["passes"] for v in checks.values())
        atomic_json(OUT / "summary.json", dict(status="complete", images=len(rows), protocol_sha256=digest,
                    R=dict(zip(map(str, KS), r.tolist())), mR=dict(zip(map(str, KS), mr.tolist())),
                    official_zR=dict(zip(map(str, KS), zr)), zero_shot_relations=sum(x["zero_shot_relations"] for x in rows),
                    reference_checks=checks, reproduction_passed=passed,
                    remaining_training_caveats=["PySGG hard-negative sampler", "per-image gradient accumulation and BatchNorm"],
                    interpretation="Within-tolerance corrected reimplementation if passed, not an exact official-training reproduction"))
    except Exception as exc:
        atomic_json(state, dict(record, status="failed", reason=str(exc)))
        raise
    else:
        atomic_json(state, dict(record, status="complete"))
    finally:
        lock.close()


if __name__ == "__main__":
    main()
