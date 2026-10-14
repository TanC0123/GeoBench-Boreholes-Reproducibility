"""Scientific counterexamples, not snapshots of implementation details."""
import numpy as np
import pandas as pd
import pytest
from GeoBench.reproducibility.data import labels_at_depth, build_queries
from GeoBench.reproducibility.features import DonorFeatures
from GeoBench.reproducibility.metrics import classification_scores

def intervals():
    rows = []
    for bid, x in [("A", 0.), ("B", 10.), ("C", 20.)]:
        for label, top, bottom, horizon in [("sand", 0., 5.3, "h1"), ("clay", 5.3, 10., "h2")]:
            rows.append(dict(borehole=bid, x=x, y=0., collar=100., label=label, top=top, bottom=bottom, horizon=horizon, rank=int(horizon[1]), boundary_observed=horizon == "h1"))
    return pd.DataFrame(rows)

def test_unknown_and_conflicting_intervals_do_not_become_labels():
    data = intervals().query('borehole == "A"').copy()
    labels, support = labels_at_depth(data, [-1., 0., 5.3, 10., 100.])
    assert labels.tolist() == ["", "sand", "clay", "", ""]
    data.loc[data.label == "clay", "top"] = 4.
    labels, support = labels_at_depth(data, [4.5])
    assert labels[0] == "" and support[0] == 2

def test_holdout_labels_and_depth_targets_do_not_change_features():
    data = intervals()
    train = data.query('borehole != "C"').copy()
    heldout = data.query('borehole == "C"').copy()
    feature = DonorFeatures(train, 1, ["h1", "h2"])
    feature.transform(train, fit=True)
    before = feature.transform(heldout)
    heldout["target"] = -99999.
    heldout["bottom"] = 99999.
    heldout["label"] = "invented"
    assert np.array_equal(before, feature.transform(heldout))
    assert feature.donor_boreholes == {"A", "B"}
    with pytest.raises(ValueError, match="training boreholes"):
        feature.transform(heldout, fit=True)

def test_self_labels_are_excluded_and_terminal_is_unknown():
    data = intervals().query('borehole == "A"').copy()
    query = data.assign(depth=[1., 100.], z=[99., 0.])
    feature = DonorFeatures(data, 2, ["clay", "sand"])
    result = feature.transform(query, fit=True)
    names = feature.feature_names()
    assert np.all(result[:, names.index("nearest_label:unknown")] == 1.)
    assert np.all(result[:, names.index("neighbour_fraction:unknown")] == 1.)

def test_query_grid_does_not_reset_at_true_boundaries():
    data = intervals()
    metadata = dict(dataset="Zhengzhou")
    _, points, _ = build_queries(data, metadata, step=1.)
    data.loc[data.horizon == "h1", "bottom"] = 5.7
    data.loc[data.horizon == "h2", "top"] = 5.7
    _, changed, _ = build_queries(data, metadata, step=1.)
    assert np.array_equal(points.depth, changed.depth)
    assert np.allclose(np.mod(points.depth, 1), 0.5)

def test_missing_horizon_is_not_an_observed_zero_elevation():
    data = intervals().query('borehole != "C"').copy()
    feature = DonorFeatures(data, 1, ["h1", "h2"])
    result = feature.transform(data, fit=True)
    names = feature.feature_names()
    assert np.all(result[data.horizon == "h2", names.index("boundary_support_present")] == 0)

def test_classification_perfection_and_hole_replication_invariance():
    evidence = pd.DataFrame({"borehole": ["A", "A", "B", "B"], "target": ["sand", "sand", "sand", "clay"], "prediction": ["sand", "sand", "sand", "clay"]})
    perfect, _, _ = classification_scores(evidence, ["sand", "clay", "unused"])
    assert perfect["balanced_macro_f1"] == 1.
    evidence.loc[3, "prediction"] = "sand"
    score, _, _ = classification_scores(evidence, ["sand", "clay", "unused"])
    duplicated = pd.concat([evidence, evidence.query('borehole == "A"')], ignore_index=True)
    repeated, _, _ = classification_scores(duplicated, ["sand", "clay", "unused"])
    assert score["balanced_macro_f1"] == repeated["balanced_macro_f1"]
    assert score["balanced_accuracy"] == repeated["balanced_accuracy"]

def test_real_terminal_boundaries_are_removed_from_targets_and_donors():
    from pathlib import Path
    from GeoBench.reproducibility.data import load_intervals
    root = Path(__file__).resolve().parents[1]
    data, metadata = load_intervals(root, "Zhengzhou")
    boundary, _, metadata = build_queries(data, metadata)
    assert len(boundary) == 623
    assert data.boundary_observed.sum() == len(boundary)
    assert set(boundary.source_row) == set(data.loc[data.boundary_observed, "source_row"])
