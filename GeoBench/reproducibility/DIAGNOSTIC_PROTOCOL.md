# Reusable diagnostic evaluation protocol

This method investigates why an assessment changes. A low score alone does
not diagnose a cause, and a passed invariant does not prove useful geology.
The contribution is a linked, executable comparison protocol, not a new
classifier, interpolator, isotonic algorithm or clinical diagnostic test.

## Comparison contract

Before fitting, specify observation type, admissible target, query identity,
allowed information, split/group unit, scoring support and aggregation. A
matched computational contrast keeps its target IDs, splits and aggregation
fixed while varying a declared component. When source curation changes eligible
targets, rescore the original fit on the common intersection before attributing
changes to refitting. Record inapplicable tasks rather than inventing labels.

## Execution sequence

1. Check source intervals/points, units, terminal treatment and label semantics.
   Use `source_audit.py` for the documented counterexamples. Source mutations
   do not certify true absence or correct formation correspondence.
2. Fit with isolated boreholes/well families and training-only transformations.
   Retain every prediction, split, donor list, selected epoch and source hash.
3. Run the original rescorer on common OOF targets. Compare regression accuracy
   with order diagnostics and classification scores with their exact estimands.
4. Apply declared perturbations: source-repair exclusion on matched support,
   source-order projection where order is independently known, or the matched
   geometry/donor feature design below. Preserve all changed and unchanged runs.
5. Describe response magnitudes, seed directions and scope; use training-support
   strata to investigate a particular alternative explanation. Do not discard
   unfavourable seeds, alter a hypothesis family or select the best stratum.

## Feature-factor diagnostic

Classification always receives raw coordinates R. Optional G supplies collar,
depth, nearest-other-donor distance and radius count. Optional D supplies nearest
donor categories and six-neighbour frequencies, including unknown. Fit R, R+G,
R+D and R+G+D on the same queries and splits. Report every seed, simple effect
and I=F_GD-F_G-F_D+F_R. The estimand includes changes to input dimension and
fitting behavior. An interaction alone does not determine whether a model is
better than raw input or explain geological causality.

From a standalone release root, reuse saved evidence or generate additional
fits with:

```bash
python -m GeoBench.reproducibility.feature_diagnostics --input results/main --output .tmp/diagnostics --workers 2
python -m GeoBench.reproducibility.summarize_diagnostics --input .tmp/diagnostics --original results/main --output .tmp/diagnostic_analysis
python -m GeoBench.reproducibility.support_diagnostics --input results/main --output .tmp/support_analysis
```

The empirical extension contains 360 diagnostic records: 80 original files
reused exactly and 280 additional fits. Output contracts require unique query
IDs, shared scoring support, outer/inner group separation, training feature
names, seed, prediction SHA-256 and independent rescoring. The original 600
main files remain authoritative for the original main tables.

## Interpretation boundaries

| Observable finding | Diagnostic comparison | Bounded implication |
| --- | --- | --- |
| Unsupported assigned labels | Explicit interval membership | Correct source handling before evaluating models |
| Curation-dependent rank | Original/refitted predictions on common targets | Qualify model selection by curation |
| Augmentation loss | Separate G/D effects and repeats | Investigate representation and configured learner |
| Loss persists with observed donors | Matched high-support queries | Missing nearby observations alone is insufficient |
| Zero inversions with tied interfaces | Projection plus tie statistics | Check thickness and presence independently |
| Spatial or seed instability | Shared split and seed records | State the evaluation geometry and training sensitivity |

These implications prioritize research checks. They neither establish causal
geological mechanisms nor demonstrate calibrated uncertainty or drilling utility.
