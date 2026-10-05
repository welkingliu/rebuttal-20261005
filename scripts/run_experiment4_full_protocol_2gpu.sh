#!/usr/bin/env bash
set -euo pipefail

ROOT="${SGG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
echo "[scope] converged protocol: broad native SGDet is reported separately;"
echo "[scope] this launcher runs the matched two-family VG tri-task depth panel."
exec bash scripts/run_experiment4_converged_depth_2gpu.sh
