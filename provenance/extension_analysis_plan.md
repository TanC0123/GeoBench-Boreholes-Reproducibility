# GeoBench revision 3: additional analyses planned before their outputs

Author-requested exploratory strengthening, 18 September 2026. This is a local
analysis record, not a prospective study registration. Protocol-2.0 and
revision-2 outputs remain unchanged. No repository publication or submission.

## A. Structure-aware reference

Apply the established least-squares isotonic projection to saved Track-1
predictions only where an independently supplied source order exists: Zhengzhou
and SyntheticControl, both split schemes, both feature modes, all four models.
For each query column sort by source rank and project predicted elevations onto
the cone of non-increasing sequences. The projection function must not receive
true elevations. Kimberlina is excluded because its ranking would otherwise be
derived from held-out depths. Report every configuration's RMSE, changes,
violation rate, tied adjacent-pair rate and projection displacement.

Projection onto this closed convex cone cannot increase squared error to a
truth inside the cone. Check that implication per column; do not advertise the
standard result or PAVA algorithm as a new theorem/algorithm. Projection is
weakly monotone: ties can collapse thickness. It does not prove positive layer
thickness, correct presence or globally valid 3D structure.

## B. Independent public-domain evidence

Screen sources for actual measured boreholes, location/depth metadata, label
meaning, geometry, licensing and independent geographic origin before examining
model performance. Do not force interval/formation identity on log samples.
Choose applicable tracks from available information. Same-region borehole splits
and independent-site transfer remain distinct. Freeze any chosen external
dataset inclusion rules and model choices before generating its predictions.

dh2loop Yalgoo–Singleton currently fails naive vertical-column reuse: only five
of its joined example holes have entirely vertical survey records. Its exported
labels also mix lithology, mineralisation and structural descriptions. Retain
this as applicability evidence; do not silently relabel it as a third vertical
formation dataset. Further candidate screening may lead to a separate native
point-classification task or to an applicability audit only.

## C. Contribution and literature

Compare the framework explicitly with existing log standardization and
geological modelling software. Identify observation support, admissible
information, task applicability and fixed scoring estimands as linked contracts,
with executable counterexamples and structure-aware references. Do not claim
uniqueness from an incomplete search, invent acceptance probabilities, or add
new data to the paper before provenance and results are verified.
