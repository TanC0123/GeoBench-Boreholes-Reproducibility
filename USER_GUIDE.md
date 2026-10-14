# Scientific user guide

Run commands from this package root with Python 3.12 and requirements.txt.
All paths below are local. Use new output folders; do not overwrite results/.
Document export tools and earlier manuscripts are intentionally not included.

## Inputs and task contracts

`GeoBench/reproducibility/data.py` loads the workbooks or contact table and
constructs interface and classification queries. Data schema and assumptions
are described in README.md and the source modules. Track 1 requires a supplied
horizon identity and at least three outer-training boreholes supporting it.
Track 2 requires an unambiguous interval label or a native labelled FORCE point.
Missing observations are not geological absence. No probability calibration
or drilling utility is evaluated.

`features.py` fits donors/imputation on training partitions; a training query
excludes its own borehole. Classification uses relative-depth matching.
`metrics.py` averages per-hole RMSE and normalizes confusion matrices per hole.
Native FORCE trajectories are grouped and weighted by well family.

## Reuse saved predictions without training

```bash
python examples/rescore_one_comparison.py
python -m GeoBench.reproducibility.summarize_repairs --main results/main --sensitivity results/repair_sensitivity --output .tmp/repair_sensitivity.csv
python -m GeoBench.reproducibility.summarize_diagnostics --input results/feature_diagnostics --original results/main --output .tmp/diagnostics
python -m GeoBench.reproducibility.summarize_force --input results/force_family --output .tmp/force
```

Expected repair result: on 497 common group-test boundaries, original kriging
and raw MLP score approximately 102.83 and 95.16 m; refits score 89.58 and
95.30 m. This separates population change from the refitting response.
Diagnostic output contains all seed effects, including Kimberlina neural
donor-only macro-F1 change approximately -0.2195 and Zhengzhou +0.0264.

## Full verification or optional training

```bash
python -m GeoBench.reproducibility.analyze --input results/main --output .tmp/rescored
python -m GeoBench.reproducibility.verify_kimberlina_units --output .tmp/units.json
python -m GeoBench.reproducibility.run_suite --output .tmp/new_runs --workers 1
```

The rescorer checks main prediction hashes, common targets, donor isolation
and fit metadata. Training uses CPU and may take much longer than rescoring;
saved numerical results define the manuscript evidence. Platform/library
differences can change regenerated fits. Fixed hyperparameters are reference
configurations, not equal-budget optimization. Bootstrap intervals condition
on fitted predictions; three seeds do not represent independent geology.

For a new method, preserve query IDs, fold membership, targets and scoring
weights. Export one prediction per eligible query and retain permitted-input
and donor provenance. Use metrics.py to rescore on the shared population;
do not silently discard failed queries. If curation changes the target set,
first rescore the original model on the intersection. New sources need their
own justified adapter; intervals, order and missing labels must not be invented.

The workflow and bounded interpretation of each diagnostic are specified in
GeoBench/reproducibility/DIAGNOSTIC_PROTOCOL.md. Public release must not be
confused with validation of geological truth or journal acceptance.
