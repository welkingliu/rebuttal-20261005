"""Recompute upstream frequency statistics without changing running models."""
import json
import torch
from common import ROOT, atomic_json, sha256
from r16_motifs import configure, dataset, save_torch
from native_runtime import assets


def main():
    torch.set_num_threads(2)
    out=ROOT/"results/R16_plain_motifs/statistics_audit"
    cfg=configure("sgcls",out)
    ds=dataset(cfg,"train")
    expected=ds.get_statistics()
    corrected=ROOT/"data/R16_plain_motifs/upstream_statistics.pth"
    save_torch(corrected,expected)
    path=assets("transformer","sgdet")[1].parent/"VG_stanford_filtered_with_attribute_train_statistics.cache"
    actual=torch.load(str(path),map_location="cpu")
    values={}
    for key in ["fg_matrix","pred_dist"]:
        a,b=actual[key],expected[key]
        values[key]=dict(shape=list(a.shape),exact_equal=torch.equal(a,b),
                         max_absolute_error=float((a-b).abs().max()),close=bool(torch.allclose(a.float(),b.float(),atol=1e-6,rtol=1e-6)))
    atomic_json(out/"summary.json",dict(status="complete",source=str(path),sha256=sha256(path),comparisons=values,
        legacy_cache_compatible=all(v["close"] for v in values.values()),
        corrected_cache=str(corrected),corrected_sha256=sha256(corrected),
        corrected_pred_dist_semantics="log(fg_matrix / sum_over_predicates + 1e-3)"))
    print(json.dumps(values),flush=True)


if __name__=="__main__":main()
