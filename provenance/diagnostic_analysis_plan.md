# GeoBench 2.3: diagnostic attribution plan fixed before new runs

Author-side exploratory extension, 18 September 2026. Not a preregistration.
Main and previous sensitivity results are already known. Preserve them.

## 1. Matched feature-block diagnostic

Use Zhengzhou and Kimberlina, Track 2, original group folds and inner boreholes.
References are RandomForest and ResidualMLP, whose augmentation responses differ
in the main results. No architecture search or additional hyperparameter tuning.

Factor G adds collar, depth below collar, nearest-other-donor distance and donor
count within the training-derived radius. Factor D adds the original nearest
donor one-hot label and six-neighbour category frequencies, including unknown.
Four configurations: raw R; R+G; R+D; R+G+D (original augmented).
All columns are selected from the same original fold-local feature construction.
Seeds per fold are 42+f, 142+f, 242+f. Inner/outer fitting rules stay unchanged.
For offset zero, raw and full augmentation reuse original predictions byte for
byte after input/support checks; do not rerun passed original fits. There are
240 evidence records: 40 reused and 200 new. All seeds and configurations remain.

Compute OOF borehole-balanced macro-F1 for each seed, mean/range, G and D simple
effects, full-minus-raw, and interaction I=F_GD-F_G-F_D+F_R. Report effects for
each dataset/model and every seed. Three same-direction seed effects are
descriptive repeatability, not statistical significance or population inference.
Changing feature blocks changes input dimension and learning optimization;
attribution is to the configured pipeline, not a physical causal mechanism.

## 2. Interface seed sensitivity

Both empirical datasets, both original split schemes, CoordinateMLP raw/full,
three seeds as above. Reuse 40 original offset-zero model-fold files; fit 80
additional records. Preserve common eligibility, horizon schema, scaling,
epoch selection and original source curation. Compare seed mean/range with
fixed kriging and IDW references; do not select the best seed or redefine the
main ranking. This tests whether headline advantages/degradations are specific
to one initialization. Do not pool these repeats with repaired-contact runs.

## 3. Training-support audit

For both empirical datasets and both original split schemes, derive the number
of the six nearest training columns with unambiguous labels at each test depth.
Bins fixed at 0, 1–2, 3–6 supported donors. Score all six original classification
references and both nominal input modes within each bin; retain sample and hole
counts and active class support. Compare raw/full predictions on identical bin
queries. These bins describe an observational diagnostic, not causal effects,
calibration guarantees, or a basis for choosing a post hoc headline subset.

## 4. Interpretation and artifact policy

Frame results as observation -> tested perturbation -> evidence -> bounded
implication. Do not relabel model failures as universal success, hide adverse
results, fabricate external adoption, or claim mature components as new.
Use original p-value family unchanged; no new p-value search. Data evidence
remains three empirical domains, while new paired diagnostics deepen the two
original sources. All new analysis logs stay in .tmp; deliver permanent code,
complete result CSVs, updated English/Chinese artifacts and a private report.
Repository remains private. No external writes, journal submission or messages.
