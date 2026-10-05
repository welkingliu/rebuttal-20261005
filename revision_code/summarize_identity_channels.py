"""R11: reanalyze existing R2 records, without changing any model or old result."""
import json
import time

import numpy as np

from common import ROOT, ensure_storage, atomic_json, sha256
from identity_evidence import image_record, summarize_records


def main():
    ensure_storage()
    destination = ROOT/"results/R11_identity_channels"
    all_summaries = {}
    for family in ("tde_motifs", "transformer"):
        source = ROOT/"results/R2"/family/"test"
        protocol = json.loads((source/"protocol.json").read_text())
        digest = sha256(source/"protocol.json")
        aggregate = json.loads((source/"summary.json").read_text())
        if aggregate["status"] != "complete" or not aggregate["invariance_passed"]:
            raise RuntimeError("R2 incomplete or invariance failed")
        records, hashes = [], {}
        for index, iid in enumerate(protocol["image_ids"]):
            p, j = source/"images"/(iid+".npz"), source/"images"/(iid+".json")
            row = json.loads(j.read_text())
            if row["protocol_sha256"] != digest:
                raise RuntimeError("R2 image protocol differs")
            with np.load(p, allow_pickle=False) as payload:
                record = image_record(payload)
            record["image_id"] = iid
            records.append(record)
            hashes[iid] = dict(npz=sha256(p), json=sha256(j))
            if (index+1)%200 == 0:
                print(json.dumps(dict(family=family, images=index+1, total=len(protocol["image_ids"]))), flush=True)
        summary = summarize_records(records)
        for name, original in aggregate["results"].items():
            recomputed = summary["conditions"][name]["all"]["delta_hit1"]["value"]
            if not np.isclose(original["delta_hit1"], recomputed, atol=1e-12):
                raise RuntimeError("Reanalysis disagrees with source metric")
        atomic_json(destination/family/"image_counts.json", records)
        atomic_json(destination/family/"sources.json", dict(protocol_sha256=digest, image_sha256=hashes))
        summary.update(status="complete", family=family, images=len(records), source=str(source),
                       source_protocol_sha256=digest, note="Descriptive follow-up of already inspected R2 test results, not a new confirmatory test")
        atomic_json(destination/family/"summary.json", summary)
        all_summaries[family] = str(destination/family/"summary.json")
    atomic_json(destination/"summary.json", dict(status="complete", families=all_summaries, finished_at=time.time()))


if __name__ == "__main__":
    main()
