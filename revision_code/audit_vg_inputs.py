"""Check migrated VG image links and native-loader equivalence to standard VG-150."""
import collections
import json
import os
import time

import h5py
import numpy as np

from common import ROOT, OLD, atomic_json, ensure_storage, sha256


def main():
    ensure_storage();start=time.monotonic()
    standard=OLD/"data/vg/v1.4/VG-SGG.h5"
    native=OLD/"data/vg/VG-SGG-with-attri.h5"
    names=["split","labels","predicates","relationships","boxes_1024","img_to_first_box",
           "img_to_last_box","img_to_first_rel","img_to_last_rel"]
    with h5py.File(standard,"r") as a,h5py.File(native,"r") as b:
        fields={n:dict(standard_shape=list(a[n].shape),native_shape=list(b[n].shape),
                       equal=bool(np.array_equal(a[n][:],b[n][:]))) for n in names}
    a=json.loads((OLD/"data/vg/v1.4/VG-SGG-dicts.json").read_text())
    b=json.loads((OLD/"data/vg/VG-SGG-dicts-with-attri.json").read_text())
    vocab={key:({k:v for k,v in a[key].items() if v>0}=={k:v for k,v in b[key].items() if v>0})
           for key in ["label_to_idx","predicate_to_idx"]}
    image_dir=OLD/"external/official_repos/PySGG/datasets/vg/stanford_spilt/VG_100k_images"
    devices=collections.Counter();missing=[];checked=0;symlinks=0
    with os.scandir(image_dir) as entries:
        for item in entries:
            if not item.name.lower().endswith(".jpg"):
                continue
            checked+=1;symlinks+=int(item.is_symlink())
            try:
                stat=item.stat(follow_symlinks=True)
                devices[str(stat.st_dev)]+=1
            except OSError as exc:
                missing.append(dict(path=item.path,error=str(exc)))
    ok=all(x["equal"] for x in fields.values()) and all(vocab.values()) and not missing and checked>=108073
    record=dict(status="complete" if ok else "failed", standard_h5=str(standard),native_h5=str(native),
                standard_sha256=sha256(standard),native_sha256=sha256(native),fields=fields,vocabulary_equal=vocab,
                image_directory=str(image_dir.resolve()),image_entries=checked,symlinks=symlinks,
                missing_count=len(missing),missing_examples=missing[:50],image_device_counts=dict(devices),
                experiment_disk_device=str(ROOT.stat().st_dev),seconds=time.monotonic()-start,
                scope="Existence/resolved image device and exact SGG arrays; does not re-decode every image. Attribute training is disabled.")
    atomic_json(ROOT/"results/R9_path_audit/vg_inputs.json",record)
    print(json.dumps({k:record[k] for k in ["status","vocabulary_equal","image_entries","symlinks","missing_count","image_device_counts","experiment_disk_device","seconds"]}),flush=True)
    if not ok:
        raise SystemExit(1)


if __name__=="__main__":
    main()
