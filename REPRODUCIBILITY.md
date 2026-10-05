# Revision reproduction and boundaries

From this directory, using an environment with the required packages:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m unittest discover -s tests -p 'test_*.py'
PYTHONPATH=. python revision_code/test_v_scope_20261005.py
python scripts/recompute_paired_revision.py
```

The last command verifies the paired contrasts directly from per-image
frequency-minus-semantic correct-count differences and shared denominators.
It does not need datasets, model weights, or a GPU. The independent resampling
unit is the image, and zero-support images are excluded from the conditional
estimand. Full paired predictions are staged separately for source inspection.

## Native experiment deployment

`revision_code/` contains research execution sources as well as audits. It is
not a portable one-command replacement for the native environment. The old
assets and revision output roots must be set using `SGG_OLD_ROOT` and
`SGG_REBUTTAL_ROOT`. Historical mount-UUID guards, native worker environments,
source commits, config paths, and cache checks must be reviewed for a new host;
do not blindly disable them. Assets and compatible CUDA environments are still
required for full model runs. No downloads are needed for the count reanalysis.

SAM's new protocol uses the old frozen predicted-mask features, linear probes,
seeds 17/23/31, maximum 2,000 epochs and patience 20. Selection is minimum
validation NLL only. It restarts from original initialization because optimizer
states were unavailable; it does not continue from a selected best checkpoint.
The previously inspected test set is not described as independent confirmation.

The original Experiment V runner is retained for historical reproduction,
not recommended as an effective relation-grounding method. Read README.md's
objective correction before interpreting its mode names. New repair exploration
must use a separately versioned protocol and retain the joint acceptance rule.

## Publication checklist

1. Verify `MANIFEST.sha256` and `VALIDATION.json` before uploading this folder.
2. Upload the separate NPZ archive only under applicable derived-data terms.
3. Verify the downloaded archive against its SHA256SUMS after cloud upload.
4. Add a real accessible cloud URL only after that verification.
5. Do not upload `manuscript_for_overleaf/` as anonymous code: its author block is retained.

The archived historical documentation may refer to data or manuscript assets
not bundled here. Its old one-click paper launcher is not a validated entry
point for the revised manuscript.
