# Generated PySGG Configurations

Task-specific YAML files are generated on the deployment machine because they
contain local dataset, checkpoint, output, and runtime paths.

Prerequisites:
- Install the core dependencies, including PyYAML.
- Prepare the pinned PySGG repository at `external/official_repos/PySGG`.
- Prepare `checkpoints/sgg/weights/pysgg/vg/shared_detector.pth`.
- Prepare `data/derived/glove/glove.6B.200d.txt`.

From the repository root:

```bash
source scripts/project_env.sh
"$SGG_PYTHON" scripts/generate_pysgg_vg_tritask_configs.py \
  --project_root "$SGG_PROJECT_ROOT"
```

The generator writes task-specific files to `configs/pysgg_vg_tritask/`.
Its defaults are training batch 8 across two GPUs, test batch 2, and four
workers. Use `--help` to override asset locations or batch settings.

These are generated deployment configurations, not recovered per-run historical
configuration snapshots. Exact historical reproduction also requires the run's
launcher overrides, upstream source version, checkpoint, and recorded protocol.
Do not treat successful configuration generation as metric reproduction.

