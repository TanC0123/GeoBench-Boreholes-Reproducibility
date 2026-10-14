"""Reproduce source-level counterexamples without depending on legacy code."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DATASETS, STEPS, labels_at_depth, load_intervals
from .metrics import regression_scores

LEGACY_KIM_ORDER = ["etchegoin", "macoma", "chanac", "santa_margarita", "fruitvale", "mclure", "round_mountain", "olcese", "freeman_jewett", "vedder", "eocene", "cretaceous", "basement"]

def audit(root, output):
    report = {"purpose": "Reconstruct specific legacy adapter and scoring counterexamples; not new model predictions", "datasets": {}}
    for dataset in DATASETS:
        intervals, metadata = load_intervals(root, dataset)
        old_points = []
        inferred_top_mismatches = 0
        for bid, group in intervals.groupby("borehole", sort=True):
            previous = 0.
            for row in group.sort_values("bottom").itertuples():
                inferred_top_mismatches += int(not np.isclose(previous, row.top))
                old_points.extend({"borehole": bid, "depth": float(depth), "label": row.label} for depth in np.arange(previous, row.bottom, STEPS[dataset]))
                previous = row.bottom
        old_points = pd.DataFrame(old_points)
        class_count = len(metadata["labels"])
        old_perfect_f1 = float(np.mean(old_points.groupby("borehole").label.nunique() / class_count))
        if dataset == "Zhengzhou":
            order = intervals.groupby("horizon").bottom.median().sort_values(kind="stable").index.tolist()
        elif dataset == "Kimberlina":
            order = LEGACY_KIM_ORDER
        else:
            order = intervals.sort_values("rank").horizon.drop_duplicates().tolist()
        truth = intervals.assign(target=intervals.collar-intervals.bottom)
        truth["prediction"] = truth.target
        truth["rank"] = truth.horizon.map({h: i for i, h in enumerate(order)})
        old_order = regression_scores(truth, 1., True)[0]["order_disagreement"]
        supported = unobserved = ambiguous = wrong = 0
        for bid, points in old_points.groupby("borehole", sort=True):
            group = intervals[intervals.borehole == bid]
            labels, count = labels_at_depth(group, points.depth.to_numpy())
            unobserved += int((count == 0).sum())
            ambiguous += int((count > 1).sum())
            supported += int(((count == 1) & (labels == points.label.to_numpy())).sum())
            wrong += int(((count == 1) & (labels != points.label.to_numpy())).sum())
        report["datasets"][dataset] = {"boreholes": metadata["boreholes"], "legacy_points": len(old_points), "legacy_perfect_macro_f1": old_perfect_f1, "legacy_true_boundary_order_disagreement": old_order, "points_supported": supported, "points_unobserved": unobserved, "points_ambiguous": ambiguous, "points_wrong_label": wrong, "inferred_top_mismatches": inferred_top_mismatches, "source_sha256": metadata["sources"]}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.output)
