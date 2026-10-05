# Reproducibility Guide

## 1. Hardware and Runtime

The formal schedule assumes two CUDA GPUs with at least 20 GB memory each,
Linux, Python 3.10 or 3.11, and enough local storage for the official datasets
and prediction caches. Image batch size is one. The PySGG and historical model
repositories use isolated environments because their CUDA and framework
requirements differ.

The submitted runs used a Linux Mint 22.2 server with kernel
`6.14.0-36-generic`, an Intel Xeon Silver 4210R CPU, 125 GiB system memory, and
two NVIDIA GeForce RTX 3090 GPUs with 24,576 MiB each. The recorded host stack
was NVIDIA driver `580.95.05`, CUDA toolkit `12.0.140`, and Python `3.12.3`.
Model-specific environments remain isolated and are recorded per run; the host
Python and CUDA versions are orchestration metadata, not a claim that every
legacy model uses one shared framework stack.

Create the main environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
source scripts/project_env.sh
```

Pinned external repositories are installed under `external/official_repos/`.
Do not replace the recorded commits with current default branches.

## 2. Assets

Follow `THIRD_PARTY_ASSETS.md`. Raw datasets and original checkpoints are not
included in the release. This is intentional: their access terms are governed
by their original publishers. The preflight checks the expected locations and
can verify the three Causal Motifs SHA-256 values.

```bash
"$SGG_PYTHON" scripts/prepare_official_models.py \
  --project_root "$SGG_PROJECT_ROOT"

"$SGG_PYTHON" scripts/preflight_paper_reproduction.py \
  --project_root "$SGG_PROJECT_ROOT" \
  --verify_large_hashes \
  --strict
```

Generate live PySGG manifests only after the source, runtime, caches, and
checkpoints are installed. Manifests intentionally contain machine-local
absolute paths and are therefore generated, not committed.

```bash
"$SGG_PYTHON" scripts/register_pysgg_live_manifests.py \
  --project_root "$SGG_PROJECT_ROOT" \
  --models motifs transformer

"$SGG_PYTHON" scripts/register_tde_motifs_live_manifest.py \
  --project_root "$SGG_PROJECT_ROOT"
```

## 3. Smoke Test

```bash
bash scripts/reproduce_paper.sh smoke
```

The smoke test compiles all Python modules, runs the complete unit-test suite
(105 tests in the release build), validates the curated appendix, and checks
every paper-specific shell entry point. Smoke outputs are never used in formal
tables.

## 4. Formal One-Click Run

```bash
PAPER_RUN_ID=reviewer_reproduction \
RESUME=1 \
  bash scripts/reproduce_paper.sh formal
```

To reproduce selected experiments:

```bash
PAPER_STAGES="2b 3 5 5_posthoc" \
PAPER_RUN_ID=reviewer_subset \
RESUME=1 \
  bash scripts/reproduce_paper.sh formal
```

The launcher creates one completion marker per stage under
`artifacts/paper_reproduction/PAPER_RUN_ID/`. Removing a marker causes that
stage to be reconsidered; the stage's own shard/seed resume checks still apply.

## 5. Fixed Experiment V Contract

Experiment V runs exactly six training jobs:

- family: TDE-Motifs;
- modes: `supervised_control`, `grounding`;
- seeds: `17`, `23`, `31`;
- learning rate: `3e-5`;
- training/validation images: `5000/1000`, disjoint VG split-0 partitions;
- maximum/minimum epochs: `5/3`, patience `1`;
- object loss weight: `2.25`;
- task-focused object weight: `3.0`;
- relation parameters: frozen;
- selected object weight/bias delta scales: `1.1/0.5`;
- stop boundary: validation SGCls `mR@50` drop no larger than `0.005`;
- test split: never used for selection.

`scripts/run_paper_experiment5_tde_motifs.py` validates these properties from
the saved result history. The post-hoc launcher evaluates the selected
supervised seed-17 state on all 26,446 VG test images, then evaluates all six
frozen states on exact VG-vocabulary overlaps in GQA and VRD.

## 6. Output Map

| Stage | Required completion artifact |
| --- | --- |
| I-A | `artifacts/experiment_1a/RUN/*/summary.json` |
| I-B | `artifacts/experiment_1b/RUN/*/summary.json` |
| II | `artifacts/experiment_2/RUN/summary.json` |
| III | `artifacts/experiment_3/RUN/validation_report.json` |
| IV native | `artifacts/experiment_4/RUN/summary.json` |
| IV depth | `artifacts/experiment_4/RUN/summary.json` |
| V matrix | `artifacts/experiment_5/RUN/summary.json` |
| V post-hoc | `artifacts/experiment_5/RUN/posthoc/validation_report.json` |

Each JSON records ontology identifiers, supports, coverage, checkpoint/source
provenance, metric semantics, and status. Logs remain under the corresponding
run root or `artifacts/logs/`.

## 7. Regenerating Figures and Paper

```bash
bash scripts/reproduce_paper.sh paper
```

The overview used by the manuscript is
`tex/kdd2027_submission/figures/benchmark_overview.pdf`. The file in this
release is the exact final artwork supplied on 2026-07-27.

Compile from `tex/kdd2027_submission/main.tex` with the ACM toolchain. The
paper-result snapshots in `results/reported/` let a reviewer inspect the
reported aggregates without downloading datasets or rerunning GPU jobs.

## 8. Expected Non-Identity

Old CUDA kernels, nondeterministic GPU reductions, or different library builds
can produce small floating-point differences. Reproduction status is evaluated
against each manifest's declared tolerance. Structural requirements such as
image count, ontology hash, task support, seed, split, and gate logic are exact.
