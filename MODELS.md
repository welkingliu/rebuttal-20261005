# Model Scope and Counting Rules

The literature taxonomy covers at least 10 SGG families. The executable
benchmark separates breadth from depth: native SGDet reproduction requires at
least 5 verified families, while the matched VG PredCls/SGCls/SGDet panel uses
Neural Motifs and SGG Transformer. Expensive interventions use the same two
families so every causal contrast is paired with a standard-task result.

## Recommended experimental panel

- Classic/context: Neural Motifs or VCTree.
- Graph/debiasing: BGNN or TDE-Motifs.
- End-to-end transformer: RelTR, SGTR, or EGTR.
- Panoptic/open-vocabulary external validation: PSGTR/PSGFormer, PGSG, or
  OvSGTR where the native ontology matches.

Use at least 5 distinct families in the SGDet breadth panel. VG must contain at
least 5 successful SGDet families and the two matched-depth families must each
provide PredCls, SGCls, and SGDet.
OI and PSG each require 2 native SGDet families. A model is counted only after
its source commit, checkpoint SHA, ontology, parameter count, reference metric,
and task output pass validation.

## Current downloaded assets

The catalog tracks 16 runtime assets across 10 families. These files are useful
but all currently declare SGDet only. Therefore they do not yet satisfy the
PredCls/SGCls task contract. Either obtain official task-specific checkpoints
or train official PredCls/SGCls configurations; newly trained results need 3
training seeds.

Downloaded weights are not adapters. `checkpoints/sgg/manifests/` must remain
empty until a live factory or a complete official prediction cache passes its
reference reproduction.

## Diagnostic and mitigation subsets

Experiment II-B uses the two matched VG adapters with real visual, union, and
node interventions. Experiment III is optional and is not part of the formal
submission contract. Perturbation seeds measure intervention variability and
are not model-training seeds.

Experiment V uses two live trainable families, one classic and one transformer.
Their `forward_grounding` output must align `pred_rel_scores` to GT relation
rows and `pred_entity_scores` to GT entity rows. The acceptance endpoint is
validation object Top-1 improvement, preserved ECE, and no material SGCls/
SGDet mR@50 degradation. VG split 2 remains untouched until final evaluation.

Prediction-cache manifests count for Experiment-IV standard metrics only. They
do not count toward the two live diagnostic or mitigation models.

RelTR and EGTR are the first two completed SGDet families. RelTR is a live
raw-image adapter; EGTR uses a provenance-locked official prediction cache from
its isolated runtime. EGTR's multi-label predicate sigmoid is preserved through
`relation_score_mode=independent_probabilities`, and its published mR protocol
is reported separately as image-conditioned `imR`.

SGTR is the third validated SGDet integration. Its released VG checkpoint is
evaluated in an isolated Python 3.10 runtime and exported to the same strict
cache schema. Its object no-object probability is retained as box objectness,
and its focal predicate sigmoid is never converted to a categorical softmax.
The exact released-checkpoint references are `R@50=24.30` and `mR@50=12.31`.

The OpenPSG panel adds four independently released PSG families: Neural
Motifs, VCTree, PSGTR, and PSGFormer. All four pass one-image strict-cache
validation under the exact 133-object/56-predicate ontology. Their unified
Experiment-IV results use box-IoU SGDet; OpenPSG's published panoptic-mask-IoU
numbers are recorded as a separate native protocol and are not treated as
direct reproduction targets for the box table. These cache integrations count
for standard Experiment IV only, not live perturbation or mitigation coverage.

See `ASSET_SETUP.md` for acquisition and deployment commands.
