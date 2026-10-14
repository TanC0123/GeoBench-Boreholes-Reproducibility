# FORCE external-dataset analysis: fixed before predictions

Date: 18 September 2026. Exploratory author-review extension; not preregistered.
The current archive contains 98 training wells, not the 83 stated in an older
starter notebook or all 118 wells described by the full dataset deposit.

Input: official repository train.zip, SHA-256
`e996c8466f2d0dc7116daa71485cc90c31fcfac6345a5bb33165026478b642e9`.
Source DOI 10.5281/zenodo.4351156 identifies the broader deposit and licenses.
Use only this training archive; do not mix former competition test sets.

Retain rows with finite measured depth and X/Y/Z and a provided lithology code.
Within each well and floor(measured_depth/5 m) bin, retain the observed row
nearest the bin midpoint, resolving ties by measured depth and source row.
Do not interpolate labels, infer contact intervals or use label transitions to
place queries. Inclusion does not filter interpretation confidence. Preserve it
as metadata and rescore high-confidence observations as a sensitivity.
Preflight yields 35,419 sampled observations from 98 wells and 12 classes.

Use a supplied 3D location as the raw input to RF and ResidualMLP. Do not use
logs, lithostratigraphic groups, formation names, confidence or well identity as
predictors. Retain native deviated-well coordinates. Track 1 and interval-based
donor augmentation are inapplicable and are not fabricated for this source.

Shared five-fold group and K-means spatial splits use well IDs and each well's
shallowest complete-coordinate observation for split geometry, respectively;
that location is not called a verified collar. Seeds and inner splits match
the original protocol. References: measured-depth-bin mode (5 m), Euclidean
3D nearest observed training point, 200-tree RandomForest, ResidualMLP with the
original 128-unit architecture and inner epoch selection / outer refitting.
Fit RF and MLP with three seeds per outer fold: 42+f, 142+f and 242+f; keep
split assignments and inner boreholes fixed. Deterministic references run once.
Total: 80 new model-seed-fold records. Run two CPU workers; numerical libraries
and models each use one thread. Never select the best seed for reporting.

Primary score: borehole-weighted macro-F1 using the existing implementation.
Retain pooled macro-F1, accuracy, coverage, training-unseen classes, sampled
support and all individual predictions. Report every seed and split, plus
mean/range over seeds for RF/MLP. Seed ranges are descriptive, not confidence
intervals. This is transfer of a protocol to independent geography with models
refitted there, not transfer of Zhengzhou-trained models or a reproduction of
the original log-based FORCE competition leaderboard.

## Amendment after the first data-contract audit

The initial name-wise run exposed one shared numeric well-family identifier:
34/5-1 A and 34/5-1 S (initially assigned to different group folds). Use the
numeric prefix as the grouping AND scoring unit for all models, giving 97
families from 98 wellbores. Average the shallowest-coordinate locations within
that family for spatial split geometry. Retain actual wellbore IDs in every
record. Repeat all 80 external records under this common amended protocol;
do not compare or select alternative groupings by model score. Initial name-wise
outputs remain in .tmp/geobench_review3/force_external for traceability; amended
outputs are in force_family and alone support the manuscript. The original
GeoBench main experiments are untouched. This is a leakage-prevention amendment,
not a prospectively registered protocol.
