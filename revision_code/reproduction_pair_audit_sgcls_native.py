"""SGCls pair audit retaining native unchunked feature extraction exactly."""
from pathlib import Path

import reproduction_pair_audit as audit
from common import ROOT, sha256

SOURCE = Path(__file__).resolve()
audit.OUT = ROOT / "results/R14_pair_cap_sgcls_native"
audit.SPEC = dict(audit.SPEC, version="r14_sgcls_native_union_v1", tasks=["sgcls"], union_chunk=None,
                  reason="Chunked GEMM changed near-tied scores; retain original feature extraction, do not relax no-op tests")
original_fingerprints = audit.fingerprints
audit.fingerprints = lambda: dict(original_fingerprints(), **{str(SOURCE): sha256(SOURCE)})


class NativeUnion:
    def __init__(self, module, chunk):
        if module.training:
            raise RuntimeError("Validation model must be in eval mode")

    def close(self):
        pass


audit.ChunkUnion = NativeUnion

if __name__ == "__main__":
    audit.main()
