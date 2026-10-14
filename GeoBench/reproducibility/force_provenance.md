# FORCE external-data provenance and applicability

The original dataset is [Bormann et al. (2020), Zenodo 4351156](https://doi.org/10.5281/zenodo.4351156).
Its record has CC BY 4.0 terms and retains the Norwegian public-data license
notice for original well logs. Dataset attribution and those original terms
are retained; GeoBench does not replace them with its software license.

The input used here is the official repository's
`lithology_competition/data/train.zip`, pinned to commit
`c8d01ee92c1c8e1ecba36f96cca6ea7b689338a1`:
[source archive](https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition/blob/c8d01ee92c1c8e1ecba36f96cca6ea7b689338a1/lithology_competition/data/train.zip).
Archive SHA-256:
`e996c8466f2d0dc7116daa71485cc90c31fcfac6345a5bb33165026478b642e9`.

The inspected archive contains 98 wellbores and 1,170,511 point records. The
full Zenodo deposit's 118-well count and the older starter notebook's 83-well
count refer to other versions/scopes and are not used to describe this input.
`X_LOC`/`Y_LOC` are provided projected UTM coordinates and `Z_LOC` is the
provided vertical location. The associated LAS export specifies project depth
units in metres. A full horizontal EPSG code is not invented.

The adapter retains native 3D trajectory positions and provided lithology
interpretations. It omits 10,775 rows lacking full coordinate support. Within
each wellbore's 5 m measured-depth bins, it selects the observed point closest
to the bin centre, yielding 35,419 points. It never fills between points.
Twelve lithology categories remain, including one represented by four samples.

The numeric well-family prefix is the split and scoring unit: 97 families,
including the shared family of `34/5-1 A` and `34/5-1 S`. Actual wellbore IDs
remain in records. Split positions are family means of the shallowest complete
coordinate observations, not certified collars. Grouping also applies to inner
epoch selection, training-only transformations and well-balanced scoring.

The extension supports observed-point classification only. No interface
identities, continuous interval tops/bottoms, source order, or interval-donor
features are inferred. Its coordinate-only references are not comparable to
the original competition leaderboard, which used well logs and another score.
Models are refitted in this domain; this is not cross-domain transfer of a
trained Zhengzhou model. The fixed analysis and grouping amendment are
documented in `provenance/force/analysis_plan.md` in the local release candidate.
