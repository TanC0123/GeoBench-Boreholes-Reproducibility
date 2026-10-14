"""Reproduce the matched-target curation comparison without model fitting."""
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
def load(directory, model):
    return pd.concat([pd.read_csv(directory / "predictions" /
        f"track1_raw_{model}_fold{fold}.csv.gz") for fold in range(5)],
        ignore_index=True)

frames = {(stage, model): load(root / directory, model)
    for stage, directory in [("original", "results/main/Kimberlina_group"),
                             ("refitted", "results/repair_sensitivity/group")]
    for model in ["OK", "CoordinateMLP"]}
keys = set.intersection(*(set(p.query_id) for p in frames.values()))
assert len(keys) == 497
print(f"Matched targets: {len(keys)}")
reference = None
for (stage, model), data in frames.items():
    data = data[data.query_id.isin(keys)].sort_values("query_id")
    assert data.query_id.is_unique
    truth = data[["query_id", "borehole", "target"]].reset_index(drop=True)
    if reference is None:
        reference = truth
    else:
        pd.testing.assert_frame_equal(reference, truth)
    errors = (data.prediction - data.target)**2
    score = float(np.sqrt(errors.groupby(data.borehole).mean()).mean())
    print(f"{stage:8s} {model:13s} {score:.8f} m")
