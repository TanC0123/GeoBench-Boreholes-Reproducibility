# GeoBench-Boreholes research package, candidate 2.4

Research snapshot approved for public release by the corresponding author
on 14 October 2026. Scientific evidence is unchanged from the reviewed candidate.
Scientific results use the preserved protocol-2.0/v2.3 evidence; no models
were retrained for this candidate. CAGEO-2.4 is a separate manuscript revision.

GeoBench-Boreholes evaluates conditional interface prediction and borehole
classification with explicit observation support, isolated training donors,
common scoring populations and matched diagnostic comparisons. It is not a
new prediction architecture or a universal geological reconstruction system.

## Installation and quick start

Use Python 3.12 in an isolated environment. The recorded CPU dependency
versions are in requirements.txt. Inputs and saved results are included;
scientific execution needs no network access after installing dependencies.

```bash
python -m venv .venv
# Activate .venv using the command appropriate to your operating system.
python -m pip install -r requirements.txt
python -m pytest tests -q -o cache_dir=.tmp/pytest
python examples/rescore_one_comparison.py
```

The example reads saved predictions, independently computes the per-borehole
RMSE on matched targets, and reproduces the contact-curation ranking reversal.
It creates no new training run and prints both original and refitted scores.
See USER_GUIDE.md for inputs, outputs, limitations and commands.

## Data and observable information

- Zhengzhou: two original workbooks, including borehole identifiers, metric
  coordinates, collar elevations, depths and geological descriptions. They
  are not anonymized coordinates. Interval tops are inferred from previous
  bottoms; the projected CRS is unspecified. Seven terminal-tail rows are
  outside the frozen complete-schema corpus, and five nonterminal targets
  have unresolved conflicting elevation fields.
- Kimberlina: the attributed Subsurface processed contact table and provenance;
  the upstream artificial repairs and their sensitivity are retained.
- FORCE: the attributed source archive, coordinate-based point sample,
  grouping of related sidetracks, provenance and saved predictions.
- SyntheticControl: deterministic source generator and saved evidence.

The data, predictions and donor IDs can reveal the original site geometry.
Public release of this candidate would expose those exact fields, not just
aggregate tables. See LICENSE_SCOPE.md and the original third-party notices.

Classification donors use the same depth below each donor's own collar as
the target depth below its collar. This is not equal absolute elevation or
verified geological correspondence. Donor support counts follow that rule.
Terminal empirical boundaries are excluded from regression targets/donors;
classification retains their observed intervals. FORCE points do not support
invented intervals, interface targets or a geological order constraint.

## Evidence and scope

results/ contains 1640 prediction CSVs, including 80 exact reuse copies.
Projection files transform saved predictions; these are not 1640 independent
training experiments. All seeds and adverse outcomes are preserved.
analysis/ contains scores, support strata and source-curation comparisons.
Source hashes, split assignments, donor lists and fit records accompany
predictions. SHA256SUMS.json identifies this exact candidate's contents.

The package excludes submission manuscripts, cover letters, editorial
decisions, coauthor administrative emails, private reports, Chinese reading
translations, browser traces and other papers. It contains scientific
author/provider attribution and correspondence contact where part of the
existing provenance. No journal acceptance, DOI or completed publication
is claimed.

Original code: MIT. Authorized Zhengzhou data and original outputs: CC BY 4.0.
Third-party material retains its original license and attribution.
Authors: Chao Tan, Xiaojie Zhai, Liping Zheng.
Intended project location: https://github.com/TanC0123/GeoBench-Boreholes-Reproducibility.
The repository contains the approved research snapshot; submission manuscripts
and administrative records are maintained separately.
