"""One finite exploratory V-F pilot, using a fresh adaptation gate."""
import copy
from pathlib import Path

from common import ROOT, sha256
from repro_protocol import immutable
from vc_protocol import SPEC as VC_SPEC
from ve_experiment import read
from ve_protocol import verify as verify_ve

HERE = Path(__file__).resolve().parent
OUT = ROOT / "results/VF"
CACHE = ROOT / "cache/VF"
MANIFEST = ROOT / "manifests/VF_protocol.json"
SPEC = dict(version="vf_decoupled_rich_readout_v1", family="transformer", tasks=["sgcls", "sgdet"], seed=17,
            train_images=5000, development_images=500, gate_images=1000,
            train="native model frozen/eval; cache float32 ROI4096 + object-context512 features",
            head="layer-normalize each block; 4608-to-256 GELU-to-151 residual; zero output init; no amplitude cap",
            modes=["supervised", "score_protected"],
            objective="supervised: object CE; protected: CE + object KL + pair-score KL; weights 1; no predicate KL",
            relationship="predicate logits and semantic context fixed; native object NMS, boxes and triplet ranking recomputed",
            max_epochs=20, min_epochs=3, patience=3, learning_rate=.0003, weight_decay=.0001,
            image_batch_size=1, accumulate_images=4, gradient_clip=1.,
            selection="maximum development post-NMS foreground accuracy; foreground NLL breaks exact ties",
            development_stop="candidate identity gain must be >=0.005 before its one fresh gate; no parameter search",
            gate_exclusions="all VB/VC development+gate IDs and VE development+gate IDs for both tasks",
            gate_scope="fresh for adapter fitting/selection; native reference audits already observed VG validation",
            plan="SGCls pilot on GPU1; SGDet only after SGCls gate passes and VE terminates; one seed, no test tonight",
            interpretation="post-hoc exploratory repair motivated by prior diagnostics; not confirmatory efficacy or new SOTA",
            stop="one candidate configuration; any gate fail stops expansion; wall budget12h; no threshold relaxation")
SPEC["gate"] = copy.deepcopy(VC_SPEC["gate"])
SPEC["gate"]["candidate"] = "score_protected_seed17"


def registration():
    return dict(spec=SPEC, ve_registration=verify_ve(),
                sources={n: sha256(HERE / n) for n in ["vf_protocol.py", "vf_native.py", "vf_experiment.py", "vf_queue.py"]})


def register():
    immutable(MANIFEST, registration())


def verify():
    if read(MANIFEST) != registration():
        raise RuntimeError("Registered V-F sources/protocol changed")
    return sha256(MANIFEST)
