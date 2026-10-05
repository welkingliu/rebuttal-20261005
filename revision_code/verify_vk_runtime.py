"""CPU compatibility check; does not instantiate an SGG or DINO model."""
import argparse
import json
from pathlib import Path

import torch

from vj_mac import digest, write
from vk_gate_math import validate_fixture


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture",type=Path,required=True); p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True); a=p.parse_args()
    torch.set_num_threads(2)
    fixture=torch.load(str(a.fixture),map_location="cpu")
    state=torch.load(str(a.checkpoint),map_location="cpu")
    if digest(a.checkpoint)!=fixture["checkpoint_sha256"]: raise RuntimeError("Frozen candidate changed")
    result=dict(status="complete",torch_version=str(torch.__version__),cuda_initialized=torch.cuda.is_initialized(),
        checkpoint_sha256=digest(a.checkpoint),fixture_sha256=digest(a.fixture),**validate_fixture(fixture,state))
    if torch.cuda.is_initialized(): raise RuntimeError("CPU preflight unexpectedly initialized CUDA")
    write(a.report,result); print(json.dumps(result),flush=True)


if __name__=="__main__":main()
