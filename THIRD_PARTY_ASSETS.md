# Third-Party Assets and Provenance

This release does not redistribute raw datasets, original model checkpoints,
or external repositories. Install them from their publishers and retain their
licenses and access terms.

## Causal Motifs / TDE-Motifs

The paper cites:

> Kaihua Tang, Yulei Niu, Jianqiang Huang, Jiaxin Shi, and Hanwang Zhang.
> Unbiased Scene Graph Generation from Biased Training. CVPR 2020.

Upstream implementation:

- repository: `https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch`
- pinned commit: `ceb71fa88461c2a97a6258a80f47669d89207296`
- code license at the pinned source: MIT

The checkpoint links below are the official OneDrive model-zoo links recorded
by the upstream project, not an unrelated mirror:

| Task | Official download | Required local path | SHA-256 |
| --- | --- | --- | --- |
| PredCls | `https://1drv.ms/u/s!AmRLLNf6bzcir9xx725wYjN7lytynA?e=0B65Ws` | `checkpoints/sgg/weights/causal_motifs_sum/vg/predcls/model_0030000.pth` | `57663ccf4c57ed8740e830afbbde8e5c4334577cc9883aebef9b1b73c9113ec0` |
| SGCls | `https://1drv.ms/u/s!AmRLLNf6bzcir9xyuLO_I8TSZ6kfyQ?e=Y5686s` | `checkpoints/sgg/weights/causal_motifs_sum/vg/sgcls/model_final.pth` | `467da372633dbd77720cd3e7e5cc056552b86d957dd0ef4d571757e0786fc674` |
| SGDet | `https://1drv.ms/u/s!AmRLLNf6bzcir9x7OYb6sKBlzoXuYA?e=s3Y602` | `checkpoints/sgg/weights/causal_motifs_sum/vg/sgdet/model_0028000.pth` | `f57891f578a04320d0078a7bda1ca3dc192eb847744e6082e1e085b39f530f20` |

The manuscript should cite the CVPR paper and name the upstream repository in
the reproducibility statement. OneDrive itself is a delivery mechanism and
does not need a scholarly citation. The release README records the exact URL
and hash so the checkpoint provenance is auditable.

The original checkpoint download page does not provide a separate explicit
redistribution grant. For that reason, the update ZIP includes only the six
small calibration states produced by this work and never includes the original
`.pth` checkpoints. A reviewer supplies the official base checkpoint and the
loader verifies its hash before applying a calibration state.

## Pinned Repositories

The executable catalog is `scripts/official_model_catalog.py`. Important
commits include:

| Repository | Commit |
| --- | --- |
| PySGG | `a63942a076932b3756a477cf8919c3b74cd36207` |
| OpenPSG | `34b2a892f7441966265e3d60ad01ee8eeae89041` |
| KERN | `de654c09c92dd57f11bedf859d3f760f5de90e31` |
| RelTR | `fca7397e9aaeccd95541e83afa4b971f3fa89014` |
| SGTR | `03bdd6554f12d521807cf95fe6a7daa7d3bb01dc` |
| EGTR | `7f87450f32758ed8583948847a8186f2ee8b21e3` |

Each generated model manifest records the source URL, commit, checkpoint hash,
ontology ID, score semantics, task coverage, and reference tolerance.

## Datasets

VG, Open Images, PSG, GQA, and VRD must be obtained under their original terms.
The code package contains loaders and derived-manifest schemas only. Do not
upload raw images or third-party annotations to the project repository unless
their publishers explicitly permit redistribution.
