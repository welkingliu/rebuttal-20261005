"""Compare inspected release sources and aggregates without modifying originals."""
import json
import math

from common import ROOT, OLD, atomic_json, sha256


def normalize(value):
    if isinstance(value,float) and math.isnan(value):
        return "<nonfinite_nan>"
    if isinstance(value,str):
        for prefix in (str(OLD),"/home/USER/desktop/lch/kdd_sgg_core_experiments"):
            value=value.replace(prefix,"${PROJECT_ROOT}")
        return value
    if isinstance(value,list):
        return [normalize(x) for x in value]
    if isinstance(value,dict):
        return {k:normalize(v) for k,v in value.items()}
    return value


def main():
    folder=ROOT/"imported/anonymous_release_20260929"
    audit=json.loads((folder/"AUDIT.json").read_text())
    code=[]; records=[]
    old_results=list((OLD/"artifacts/experiment_2").rglob("experiment_2.json"))
    old_results+=list((OLD/"artifacts/experiment_3").rglob("experiment_3.json"))
    for entry in audit["downloaded"]:
        name=entry["path"]
        if name.endswith(".py"):
            p=OLD/name
            code.append(dict(path=name,server_exists=p.is_file(),
                             same_hash=p.is_file() and sha256(p)==entry["sha256"]))
        if name.endswith(("experiment_2.json","experiment_3.json")):
            payload=json.loads((folder/name).read_text())
            matches=[]
            keys=[k for k in ("pair_audit","graph_audit","dose_response_and_controls",
                              "object_error_propagation") if k in payload]
            for path in old_results:
                old=json.loads(path.read_text())
                if old.get("dataset")==payload.get("dataset") and all(normalize(old.get(k))==normalize(payload[k]) for k in keys):
                    matches.append(str(path))
            records.append(dict(path=name,server_aggregate_matches=matches,compared_sections=keys,
                                release_sha256=entry["sha256"]))
    result=dict(status="complete",source_audit=str(folder/"AUDIT.json"),code=code,records=records,
                historical_per_image_records_recovered=False,
                limitation="Existing aggregate confidence intervals are available, but image-level paired effects cannot be reconstructed from those intervals.",
                actions="Archived remote files under imported only; no published or old source files overwritten.")
    atomic_json(ROOT/"results/anonymous_release_audit/summary.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
