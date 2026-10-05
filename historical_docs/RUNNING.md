# Running and Monitoring

The canonical launcher is:

```bash
source scripts/project_env.sh
PAPER_RUN_ID=reviewer_reproduction \
  bash scripts/reproduce_paper.sh formal
```

Run only selected stages:

```bash
PAPER_STAGES="3 5 5_posthoc" \
PAPER_RUN_ID=focused_reproduction \
  bash scripts/reproduce_paper.sh formal
```

## Progress

All new formal launchers print progress to the terminal and retain logs.

```bash
tail -n 100 -f artifacts/experiment_5/RUN_ID/logs/*.log
tail -n 100 -f artifacts/experiment_3/RUN_ID/logs/*.log
tail -n 100 -f artifacts/experiment_4/RUN_ID/logs/*.log
```

GPU and process status:

```bash
watch -n 2 nvidia-smi
pgrep -af 'sgg_core|run_paper_experiment'
```

Experiment V emits a structured event every 100 valid training images by
default. Completed stages also write markers under
`artifacts/paper_reproduction/PAPER_RUN_ID/`. Re-running with `RESUME=1`
retains completed shards and seeds.

## Smoke and Preflight

```bash
bash scripts/reproduce_paper.sh preflight
bash scripts/reproduce_paper.sh smoke
```

`preflight` checks datasets, pinned source markers, official Causal Motifs
weights, and Python dependencies. Add `--verify_large_hashes` to
`scripts/preflight_paper_reproduction.py` when a full byte-level checkpoint
audit is required.

## Experiment V Only

```bash
source scripts/project_env.sh

"$SGG_PYTHON" scripts/register_tde_motifs_live_manifest.py \
  --project_root "$SGG_PROJECT_ROOT"

"$SGG_PYTHON" scripts/run_paper_experiment5_tde_motifs.py \
  --project_root "$SGG_PROJECT_ROOT" \
  --manifest "$SGG_OFFICIAL_MANIFEST_DIR/pysgg_tde_motifs_vg_live.json" \
  --data_root "$SGG_VG_ROOT" \
  --output_dir "$SGG_ARTIFACT_DIR/experiment_5/reviewer_exp5" \
  --gpus 0 1 \
  --resume

EXP5_RUN_ROOT="$SGG_ARTIFACT_DIR/experiment_5/reviewer_exp5" \
  bash scripts/run_paper_experiment5_posthoc.sh
```

The matrix validator checks the actual result history for task focus, frozen
relation parameters, object-only updates, 1.1/0.5 weight/bias delta scaling,
three-to-five epochs, and validation-only selection.
