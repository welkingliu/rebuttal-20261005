#!/usr/bin/env bash
set -euo pipefail
source /mnt/EXPERIMENT_DISK/kdd_sgg_rebuttal_20260929/code/storage_runtime.sh
family="${1:?family}"
gpu="${2:?gpu}"
exec 9>"$SGG_REBUTTAL_ROOT/status/gpu${gpu}.resource.lock"
flock 9
while test -n "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"; do
    echo "[WAIT] GPU $gpu occupied"
    sleep 30
done
export CUDA_VISIBLE_DEVICES="$gpu"
"$SGG_NATIVE_PYTHON" -u code/frequency_matched_controls_v2.py "$family" --smoke
"$SGG_NATIVE_PYTHON" -u code/frequency_matched_controls_v2.py "$family"
echo "[COMPLETE] R23 v2 $family"
