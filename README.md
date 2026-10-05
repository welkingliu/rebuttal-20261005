# GroundedSGG-Bench: audited revision (2026-10-05)

This repository contains the audited revision source and selected evidence.
Submitted results, revised results, and exploratory repairs must not be merged
into one leaderboard. This is a public repository, not an anonymous review URL.

## What changed

| Evidence | Result | Interpretation |
| --- | --- | --- |
| Paired channel contrast, Motifs SGCls | +5.50 pp, 95% CI [4.78, 6.26], 14,104 pairs | Frequency oracle minus semantic oracle on fixed visual evidence |
| Paired channel contrast, Transformer SGDet | +9.25 pp, 95% CI [8.07, 10.45], 10,767 pairs | Same within-model contrast; not a model ranking |
| SAM linear-probe bounded refit | Best epochs 588/689/643; all early-stopped | Resolves cap truncation in this raw-feature protocol, not an intrinsic representation limit |
| SAM high-IoU endpoint disagreement | 51.84% mean, three seeds | 1,511 relations; both prompted masks IoU >=0.85; oracle box prompts |
| Original Experiment V audit | Weighted CE versus weighted CE + Brier on an affine score readout | Relation penalties have zero gradient to the updated readout |
| V-X exploratory visual adaptation | No trained candidate met the inner acceptance rule | Native fallback selected; not a positive mitigation result |

The SAM raw-feature sensitivity protocol is distinct from the historical
normalized-probe tables. Do not replace their values without matching feature
normalization, support, and decoder settings. Oracle substitutions are diagnostic,
not deployable corrections. Confidence intervals are image-clustered; channel
comparisons are post-hoc and are not multiplicity-corrected.

## Layout

- `sgg_core/`, `scripts/`, `tests/`, `configs/`: historical execution source.
- `revision_code/`: audit and exploratory experiment source, with machine paths redacted.
- `evidence/`: current selected summaries, paired sufficient statistics, and provenance.
- `historical_docs/`: archived instructions, not current claims of successful reproduction.
- `REPRODUCIBILITY.md`: supported local checks and deployment boundaries.
- `MANIFEST.sha256`: checksums for this staged version, generated after verification.

Full paired NPZ records are in `paired_channel_records_20261005.zip`, under the
read-only [OneDrive folder kdd_sgg_upload_20261005](https://1drv.ms/f/c/bbaa76995e4a814f/IgDQLM6an3FGR4QcPttIInadAfUA6Bp_-s6IHhmloTWIS98?e=EeRqaO).
On 2026-10-05, an unsigned-in browser could list the archive (2.34 GB),
README.md, and SHA256SUMS. Full downloaded-copy checksum verification remains
pending; listing visibility alone does not establish archive integrity.
Raw images and third-party weights are not included.

## Corrected scope of Experiment V

Only the 151x151 affine object-score readout and 151-element bias were updated
(22,952 parameters). Native vision, contextual object reasoning, and predicate
parameters remained frozen. The effective data-gradient objectives were
`2.25 * weighted CE` and `2.25 * weighted CE + 0.05 * Brier`.
The historical configuration label `grounding` is retained for artifact lookup;
it is not evidence of an effective relation-protection objective.

V test and external-transfer results inherit this interpretation. Static I-IV
entry-point imports do not invoke the V optimizer, but shared dynamic adapters
still require run-specific checkpoint/cache validation. Tests here do not prove
all historical outputs are unaffected.

Historical Experiment III paired image records were not recovered. Its old
matched-control contrasts and confidence intervals are excluded from revised
inferential claims; new experiments do not reconstruct those missing records.

## Validation status

The original R21 check failed; later checks passed after repairing dependencies
and metadata. The revision checks pass 105 unit tests, four adapter-gradient
tests, and six CLI help commands in the existing environment. This is not a
clean-install or full-dataset reproduction claim. Final checks of this exact
staged copy are recorded in `VALIDATION.json`.

Source and evidence paths were redacted. `evidence/PROVENANCE.json` retains
original and redacted-file hashes; hashes of the two versions legitimately differ.
Upstream asset terms remain in `THIRD_PARTY_ASSETS.md` and source licenses apply.
