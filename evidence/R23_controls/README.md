# Matched identity-input controls (2026-10-05)

This supplement adds completed post-hoc controls, not independent confirmation
or successful mitigation. Predicate Top-1 is evaluated on fixed eligible pairs;
it is not standard triplet recall. Seeds 17/23/31 vary interventions, not training.

| Model/task | Native | GT frequency input | Matched incorrect input, three seeds |
| --- | --- | --- | --- |
| Motifs SGCls | 63.78% | 69.22% | 52.40% / 53.37% / 52.91% |
| Transformer SGDet | 60.12% | 68.91% | 45.41% / 45.91% / 45.44% |

Each processed 2,000 images. Motifs has 14,104 evaluable relations; Transformer
has 10,767 across 1,896 positive-support images. Random replacements change the
same originally incorrect nodes as GT substitution, exclude native/true labels,
and match the true label's validation-frequency stratum. GT-minus-random gaps
combine a GT benefit and random-control harm; they are not repair gains.
Semantic confusable-minus-random contrasts remain model dependent.

`frequency/*/images/` contains 4,000 JSON and 4,000 NPZ records in total.
`semantic/` includes per-image sufficient statistics for all three strengths
and seeds. Summaries contain paired image-bootstrap intervals (10,000 draws).
No multiplicity-adjusted discovery claim is made.

The v1 run stopped because background-inclusive ranking created a singleton
frequency bin. v2 ranks only 150 foreground classes in five bins of 30; all
formal images were rerun. This was a validity correction, not outcome selection.

Inference needs the existing legacy GPU environment, parent R20/R15 records,
validation confusion counts and audited checkpoints. This is an additive
release, not a standalone inference bundle. Configure storage_runtime.sh and
the existing runtime paths before rerunning. Public code paths are redacted;
protocol source hashes refer to execution originals, not redacted copies.
The public provenance map preserves original/public source hashes.

Run the pure replacement tests with:
`cd revision_code && python -m unittest test_frequency_matched_controls_v2`
Run the audits with the configured legacy environment and code directory:
`python frequency_matched_controls_v2.py motifs --smoke`, then omit `--smoke`.
Repeat for `transformer`; `python semantic_control_pairs.py` reuses parent caches.

This package does not include raw images, model weights, smoke records or runtime
caches. Parent paired-channel records remain in the previously released archive.
JSON protocol records preserve their execution paths and hashes for audit;
paths are provenance, not portable installation instructions.
