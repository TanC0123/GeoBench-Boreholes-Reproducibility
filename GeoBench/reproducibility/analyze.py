"""Independently rescore OOF files and quantify conditional sampling sensitivity."""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DATASETS, sha256
from .metrics import classification_scores, macro_f1, regression_scores
from .run import write_json

def f1_batch(matrices):
    diagonal = np.diagonal(matrices, axis1=-2, axis2=-1)
    denominator = matrices.sum(axis=-2) + matrices.sum(axis=-1)
    f1 = np.divide(2 * diagonal, denominator, out=np.zeros_like(diagonal), where=denominator > 0)
    return np.divide(f1.sum(axis=-1), (denominator > 0).sum(axis=-1))

def bootstrap(values, blocks, track, n=10000):
    """Resample fixed OOF records: conditional uncertainty, not retraining CI."""
    rng = np.random.default_rng(20260916)
    values = np.asarray(values)
    indices = rng.integers(0, len(values), size=(n, len(values)))
    weights = np.stack([np.bincount(row, minlength=len(values)) for row in indices])
    if track == 1:
        estimates = weights @ values / weights.sum(axis=1)
    else:
        estimates = f1_batch(np.einsum("bn,nij->bij", weights, values))
    unique = sorted(set(blocks))
    # Enumerate all ordered size-K resamples for the five spatial clusters.
    cluster_draws = np.array(list(product(range(len(unique)), repeat=len(unique))))
    cluster_counts = np.stack([np.bincount(row, minlength=len(unique)) for row in cluster_draws])
    weights = cluster_counts[:, np.array([unique.index(block) for block in blocks])]
    if track == 1:
        cluster_estimates = weights @ values / weights.sum(axis=1)
    else:
        cluster_estimates = f1_batch(np.einsum("bn,nij->bij", weights, values))
    return {"hole_bootstrap_low": float(np.quantile(estimates, 0.025)), "hole_bootstrap_high": float(np.quantile(estimates, 0.975)), "spatial_resample_low": float(np.quantile(cluster_estimates, 0.025)), "spatial_resample_high": float(np.quantile(cluster_estimates, 0.975))}

def compare(first, second, track, vocabulary, block_lookup):
    if set(first.query_id) != set(second.query_id):
        raise AssertionError("Paired comparison has different target support")
    holes = sorted(first.borehole.unique())
    if track == 1:
        def values(frame):
            return np.array([np.sqrt(np.mean((g.prediction-g.target)**2)) for _, g in frame.groupby("borehole", sort=True)])
    else:
        def values(frame):
            return np.asarray(classification_scores(frame, vocabulary)[2])
    a, b = values(first), values(second)
    def score(v):
        return float(np.mean(v)) if track == 1 else macro_f1(v.sum(axis=0))
    delta = score(a) - score(b)
    blocks = np.array([block_lookup[hole] for hole in holes])
    possible = sorted(set(blocks))
    null = []
    for flags in product((False, True), repeat=len(possible)):
        swap = np.array([flags[possible.index(block)] for block in blocks])
        if track == 2:
            swap = swap[:, None, None]
        null.append(score(np.where(swap, b, a)) - score(np.where(swap, a, b)))
    p = float(np.mean(np.abs(null) >= abs(delta) - 1e-12))
    intervals = bootstrap(a - b, blocks, 1) if track == 1 else {}
    return {"difference": delta, "block_swap_p": p, "spatial_blocks": len(possible), "assignments": len(null), **intervals}

def analyze(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summaries, comparisons, quality, spatial, validations = [], [], [], [], []
    for dataset in DATASETS:
        spatial_locations = pd.read_csv(root / f"{dataset}_spatial/borehole_folds.csv")
        block_lookup = dict(zip(spatial_locations.borehole, spatial_locations.fold))
        for cv in ("group", "spatial"):
            directory = root / f"{dataset}_{cv}"
            metadata = json.loads((directory / "metadata.json").read_text())
            done = json.loads((directory / "completion.json").read_text())
            assert done["models_scored"] == 100 and done["folds"] == list(range(5))
            split_map = {s["fold"]: s for s in json.loads((directory / "splits.json").read_text())}
            locations = pd.read_csv(directory / "borehole_folds.csv")
            spatial.append({"dataset": dataset, "cv": cv, "median_nearest_distance": float(locations.nearest_training_distance.median()), "min_nearest_distance": float(locations.nearest_training_distance.min()), "max_nearest_distance": float(locations.nearest_training_distance.max())})
            if cv == "group":
                quality.append({k: metadata[k] for k in ("dataset", "boreholes", "source_rows", "boundary_rows", "excluded_terminal_boundaries", "labelled_points", "grid_points", "unobserved_points", "ambiguous_points", "step", "interface_elevation_span", "global_order_valid")} | {"source_classes": len(metadata["labels"]), "sampled_classes": len(metadata["sampled_labels"]), "unsampled_classes": ";".join(sorted(set(metadata["labels"]) - set(metadata["sampled_labels"])))})
            by_config, support, coverage, training_records = {}, {}, {}, []
            max_score_roundtrip_difference = 0.0
            for detail_path in sorted((directory / "predictions").glob("*.json")):
                detail = json.loads(detail_path.read_text())
                csv_path = detail_path.with_suffix(".csv.gz")
                assert sha256(csv_path) == detail["prediction_sha256"]
                frame = pd.read_csv(csv_path, dtype={"horizon": str})
                track, mode, name, fold = detail["track"], detail["mode"], detail["model"], detail["fold"]
                split = split_map[fold]
                train_holes, test_holes = set(split["training_boreholes"]), set(split["test_boreholes"])
                trace = detail["trace"]
                assert set(frame.borehole) <= test_holes
                assert train_holes.isdisjoint(test_holes)
                assert set(trace["outer_donor_boreholes"]) <= train_holes
                assert set(trace["inner_donor_boreholes"]) <= set(split["inner_training_boreholes"])
                assert set(trace["inner_donor_boreholes"]).isdisjoint(split["inner_validation_boreholes"])
                assert frame.query_id.is_unique and len(frame) == detail["scored_rows"]
                key = (track, fold)
                ids = set(frame.query_id)
                assert support.setdefault(key, ids) == ids
                coverage[key] = (detail["scored_rows"], detail["validation_rows"])
                if "training" in detail:
                    assert detail["training"]["outer_refit_rows"] == detail["training_rows"]
                    assert 1 <= detail["training"]["selected_epoch"] <= len(detail["training"]["inner_loss_history"]) <= 120
                    training_records.append({"track": track, "mode": mode, "fold": fold, "epoch": detail["training"]["selected_epoch"]})
                if track == 1:
                    score, _ = regression_scores(frame, metadata["interface_elevation_span"], metadata["global_order_valid"])
                else:
                    score, _, _ = classification_scores(frame, metadata["labels"])
                for metric, value in score.items():
                    expected = detail["scores"][metric]
                    # Torch emits float32 predictions; CSV round trips its
                    # shortest decimal. Recomputed tables use the saved values.
                    assert value is None and expected is None or np.isclose(value, expected, rtol=1e-6, atol=1e-6), (csv_path, metric)
                    if value is not None:
                        max_score_roundtrip_difference = max(max_score_roundtrip_difference, abs(value - expected))
                by_config.setdefault((track, mode, name), []).append(frame)
            frames = {}
            for (track, mode, name), chunks in by_config.items():
                assert len(chunks) == 5
                frame = pd.concat(chunks, ignore_index=True)
                assert frame.query_id.is_unique and frame.groupby("borehole").fold.nunique().max() == 1
                frames[(track, mode, name)] = frame
                if track == 1:
                    score, holes = regression_scores(frame, metadata["interface_elevation_span"], metadata["global_order_valid"])
                    values = np.array([h["rmse"] for h in holes])
                else:
                    score, holes, values = classification_scores(frame, metadata["labels"])
                blocks = [block_lookup[h["borehole"]] for h in holes]
                ci = bootstrap(values, blocks, track)
                numerator, denominator = map(sum, zip(*(coverage[(track, fold)] for fold in range(5))))
                summaries.append({"dataset": dataset, "cv": cv, "track": track, "mode": mode, "model": name, "scored_rows": len(frame), "coverage": numerator / denominator, "unseen_test_rows": int((~frame.train_label_seen).sum()), **score, **ci})
                per_hole = pd.DataFrame(holes)
                per_hole["spatial_block"] = [block_lookup[bid] for bid in per_hole.borehole]
                per_hole.to_csv(output / f"holes_{dataset}_{cv}_track{track}_{mode}_{name}.csv", index=False)
            for track, pairs in [(1, [("CoordinateMLP", "OK"), ("OK", "IDW")]), (2, [("ResidualMLP", "RandomForest"), ("RandomForest", "NearestObservedColumn")])]:
                for a, b in pairs:
                    comparison = compare(frames[(track, "augmented", a)], frames[(track, "augmented", b)], track, metadata["labels"], block_lookup)
                    comparisons.append({"dataset": dataset, "cv": cv, "track": track, "first": a, "second": b, "mode": "augmented", **comparison})
            validations.append({"dataset": dataset, "cv": cv, "prediction_files_verified": len(list((directory / "predictions").glob("*.csv.gz"))), "identical_model_support": True, "fold_and_donor_separation": True, "stored_scores_independently_recomputed": True, "max_float32_csv_score_roundtrip_difference": max_score_roundtrip_difference, "neural_refits_verified": len(training_records), "selected_epochs": training_records})
    # All 24 exploratory comparisons form one disclosed family.
    ordered = sorted(range(len(comparisons)), key=lambda i: comparisons[i]["block_swap_p"])
    running = 0.
    for rank, i in enumerate(ordered):
        running = max(running, min(1., (len(ordered) - rank) * comparisons[i]["block_swap_p"]))
        comparisons[i]["holm_p"] = running
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output / "comparisons.csv", index=False)
    pd.DataFrame(quality).to_csv(output / "data_quality.csv", index=False)
    pd.DataFrame(spatial).to_csv(output / "spatial_distances.csv", index=False)
    write_json(output / "verification.json", {"checks": validations, "files_verified": sum(x["prediction_files_verified"] for x in validations), "bootstrap_draws": 10000, "spatial_bootstrap_enumeration": 3125, "interval_interpretation": "conditional on fitted OOF models; no population or algorithm-training confidence guarantee", "block_swap_interpretation": "exploratory exchangeability sensitivity, not established spatially calibrated significance"})
    print(pd.DataFrame(summaries)[["dataset", "cv", "track", "mode", "model", "balanced_rmse", "balanced_macro_f1", "order_disagreement"]].to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.input, args.output)
