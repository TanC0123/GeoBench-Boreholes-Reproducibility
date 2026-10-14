"""Run protocol 2.0 and retain a complete, independently scoreable OOF record.

Example (from the project root):
    python -m GeoBench.reproducibility.run --dataset Zhengzhou --cv group
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import warnings
import xgboost

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

from .data import DATASETS, build_queries, labels_at_depth, load_intervals, sha256
from .features import DonorFeatures
from .metrics import classification_scores, regression_scores
from .models import PARAMETERS, classifier, fit_predict_mlp, interpolate, seed_all

PROTOCOL = "2.0"

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def make_splits(intervals, strategy, seed=42):
    """Use one shared borehole assignment for both tasks and all model families."""
    locations = intervals[["borehole", "x", "y"]].drop_duplicates("borehole").sort_values("borehole").reset_index(drop=True)
    if len(locations) < 5:
        raise ValueError("The evaluation requires at least five boreholes")
    if strategy == "group":
        splitter = GroupKFold(n_splits=5, shuffle=True, random_state=seed)
        assignments = np.empty(len(locations), dtype=int)
        for fold, (_, validation) in enumerate(splitter.split(locations, groups=locations.borehole)):
            assignments[validation] = fold
    elif strategy == "spatial":
        assignments = KMeans(n_clusters=5, n_init=20, random_state=seed).fit_predict(locations[["x", "y"]])
    else:
        raise ValueError(strategy)
    locations["fold"] = assignments
    splits = []
    for fold in range(5):
        train = locations[locations.fold != fold]
        test = locations[locations.fold == fold]
        distances = np.linalg.norm(test[["x", "y"]].to_numpy()[:, None] - train[["x", "y"]].to_numpy()[None, :], axis=2).min(axis=1)
        locations.loc[test.index, "nearest_training_distance"] = distances
        selector = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=1042 + fold)
        inner_train, inner_validation = next(selector.split(train, groups=train.borehole))
        split = {"fold": fold, "seed": 42 + fold, "training_boreholes": train.borehole.tolist(), "test_boreholes": test.borehole.tolist(), "inner_training_boreholes": train.iloc[inner_train].borehole.tolist(), "inner_validation_boreholes": train.iloc[inner_validation].borehole.tolist()}
        assert set(split["training_boreholes"]).isdisjoint(split["test_boreholes"])
        assert set(split["inner_training_boreholes"]).isdisjoint(split["inner_validation_boreholes"])
        splits.append(split)
    return locations, splits

def feature_bundle(intervals, train, test, inner_train, inner_validation, vocabulary, track, mode):
    outer_donors = intervals[intervals.borehole.isin(train.borehole)].copy()
    inner_donors = intervals[intervals.borehole.isin(inner_train.borehole)].copy()
    outer = DonorFeatures(outer_donors, track, vocabulary)
    inner = DonorFeatures(inner_donors, track, vocabulary)
    if set(test.borehole) & outer.donor_boreholes:
        raise AssertionError("Outer test boreholes leaked into donor table")
    if set(inner_validation.borehole) & inner.donor_boreholes:
        raise AssertionError("Inner validation boreholes leaked into donor table")
    train_x = outer.transform(train, mode, fit=True)
    test_x = outer.transform(test, mode)
    inner_x = inner.transform(inner_train, mode, fit=True)
    validation_x = inner.transform(inner_validation, mode)
    if not all(np.isfinite(x).all() for x in (train_x, test_x, inner_x, validation_x)):
        raise ValueError("Non-finite model input")
    trace = {"feature_names": outer.feature_names(mode), "outer_donor_boreholes": sorted(outer.donor_boreholes), "inner_donor_boreholes": sorted(inner.donor_boreholes), "outer_radius": outer.radius, "inner_radius": inner.radius, "outer_imputation": outer.fill.tolist() if outer.fill is not None else None, "inner_imputation": inner.fill.tolist() if inner.fill is not None else None}
    return (train_x, test_x, inner_x, validation_x), trace

def depth_prior(train, test, step):
    train_bins = np.floor(train.depth / step).astype(int)
    global_mode = train.target.value_counts().sort_index().idxmax()
    lookup = {}
    for key in sorted(train_bins.unique()):
        lookup[int(key)] = train.loc[train_bins == key, "target"].value_counts().sort_index().idxmax()
    return np.array([lookup.get(int(key), global_mode) for key in np.floor(test.depth / step).astype(int)], dtype=object)

def nearest_observed(intervals, train, test, step):
    """Closest supported column at query depth; depth-prior fallback if none."""
    columns = {bid: group for bid, group in intervals[intervals.borehole.isin(train.borehole)].groupby("borehole", sort=True)}
    ids = list(columns)
    coords = np.array([[columns[bid].x.iloc[0], columns[bid].y.iloc[0]] for bid in ids])
    prediction = depth_prior(train, test, step)
    fallback = np.ones(len(test), dtype=bool)
    for _, positions in test.groupby("borehole", sort=False).indices.items():
        query = test.iloc[positions]
        order = np.argsort(np.linalg.norm(coords - query[["x", "y"]].iloc[0].to_numpy(), axis=1), kind="stable")
        for index in order:
            label, _ = labels_at_depth(columns[ids[index]], query.depth.to_numpy())
            use = fallback[positions] & (label != "")
            prediction[positions[use]] = label[use]
            fallback[positions[use]] = False
    return prediction, int(fallback.sum())

def _subset(frame, ids):
    return frame[frame.borehole.isin(ids)].reset_index(drop=True)

def run_dataset(root, output, dataset, strategy, tracks=(1, 2), modes=("raw", "augmented"), folds=None, step=None):
    seed_all(42)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    intervals, metadata = load_intervals(root, dataset)
    interfaces, points, metadata = build_queries(intervals, metadata, step)
    locations, splits = make_splits(intervals, strategy)
    vocabulary = {1: metadata["horizons"], 2: metadata["labels"]}
    span = float(interfaces.target.max() - interfaces.target.min())
    metadata.update(protocol=PROTOCOL, cv=strategy, parameters=PARAMETERS, interface_elevation_span=span, python=platform.python_version(), environment={package: importlib.metadata.version(package) for package in ["numpy", "pandas", "scipy", "scikit-learn", "torch", "pykrige", "lightgbm", "openpyxl"]})
    metadata["environment"]["xgboost"] = xgboost.__version__
    metadata["code_sha256"] = {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    write_json(output / "metadata.json", metadata)
    write_json(output / "splits.json", splits)
    locations.to_csv(output / "borehole_folds.csv", index=False)
    interfaces.to_csv(output / "interface_targets.csv", index=False)
    points.to_csv(output / "classification_targets.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    results = []
    prediction_dir = output / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    for split in splits:
        fold, seed = split["fold"], split["seed"]
        if folds is not None and fold not in folds:
            continue
        for track in tracks:
            frame = interfaces if track == 1 else points
            train = _subset(frame, split["training_boreholes"])
            full_test = _subset(frame, split["test_boreholes"])
            if track == 1:
                count = train.groupby("horizon").borehole.nunique()
                eligible = set(count[count >= 3].index)
                test = full_test[full_test.horizon.isin(eligible)].reset_index(drop=True)
            else:
                test = full_test
            if train.empty or test.empty:
                raise ValueError(f"Empty training or scored test partition: {dataset}/{track}/{fold}")
            coverage = len(test) / len(full_test)
            inner_train = _subset(frame, split["inner_training_boreholes"])
            inner_validation = _subset(frame, split["inner_validation_boreholes"])
            labels = LabelEncoder().fit(train.target) if track == 2 else None
            # Labels in the outer training partition form the classifier's
            # output vocabulary. Test-only classes are never fitted or remapped.
            label_values = labels.classes_.tolist() if labels is not None else []
            for mode in modes:
                arrays, trace = feature_bundle(intervals, train, test, inner_train, inner_validation, vocabulary[track], track, mode)
                train_x, test_x, inner_x, validation_x = arrays
                if track == 1:
                    names = ["IDW", "RBF", "OK", "CoordinateMLP"]
                else:
                    names = ["RandomForest", "XGBoost", "LightGBM", "DepthPrior", "NearestObservedColumn", "ResidualMLP"]
                for name in names:
                    path = prediction_dir / f"track{track}_{mode}_{name}_fold{fold}.csv.gz"
                    detail_path = prediction_dir / f"track{track}_{mode}_{name}_fold{fold}.json"
                    start = time.monotonic()
                    detail = {"dataset": dataset, "cv": strategy, "track": track, "mode": mode, "model": name, "fold": fold, "seed": seed, "validation_rows": len(full_test), "scored_rows": len(test), "coverage": coverage, "training_rows": len(train), "inner_training_rows": len(inner_train), "inner_validation_rows": len(inner_validation), "class_vocabulary": label_values, "trace": trace}
                    if track == 1 and name != "CoordinateMLP":
                        prediction = np.empty(len(test))
                        for horizon, positions in test.groupby("horizon", sort=True).indices.items():
                            local = train[train.horizon == horizon]
                            prediction[positions] = interpolate(name, local[["x", "y"]], local.target, test.iloc[positions][["x", "y"]])
                    elif name in ("CoordinateMLP", "ResidualMLP"):
                        iy = inner_train.target.to_numpy() if track == 1 else labels.transform(inner_train.target)
                        vy = inner_validation.target.to_numpy() if track == 1 else labels.transform(inner_validation.target)
                        oy = train.target.to_numpy() if track == 1 else labels.transform(train.target)
                        prediction, training = fit_predict_mlp(inner_x, iy, validation_x, vy, train_x, oy, test_x, track, seed, 1 if track == 1 else len(label_values))
                        detail["training"] = training
                        if track == 2:
                            prediction = labels.inverse_transform(prediction)
                    elif name == "DepthPrior":
                        prediction = depth_prior(train, test, metadata["step"])
                    elif name == "NearestObservedColumn":
                        prediction, fallback = nearest_observed(intervals, train, test, metadata["step"])
                        detail["fallback_rows"] = fallback
                    else:
                        model = classifier(name, seed)
                        with warnings.catch_warnings(record=True) as captured:
                            model.fit(train_x, labels.transform(train.target))
                            prediction = labels.inverse_transform(np.asarray(model.predict(test_x), dtype=int))
                        detail["warnings"] = sorted(set(str(w.message) for w in captured))
                    if len(prediction) != len(test) or (track == 1 and not np.isfinite(prediction).all()):
                        raise RuntimeError(f"Invalid prediction: {name}")
                    columns = ["query_id", "borehole", "x", "y", "collar", "target"] + (["horizon", "rank"] if track == 1 else ["depth", "z"])
                    evidence = test[columns].copy()
                    evidence["prediction"] = prediction
                    evidence["fold"] = fold
                    evidence["train_label_seen"] = True if track == 1 else evidence.target.isin(label_values)
                    evidence.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
                    if track == 1:
                        score, _ = regression_scores(evidence, span, metadata["global_order_valid"])
                    else:
                        score, _, _ = classification_scores(evidence, metadata["labels"])
                    detail["elapsed_seconds"] = time.monotonic() - start
                    detail["prediction_sha256"] = sha256(path)
                    detail["scores"] = score
                    write_json(detail_path, detail)
                    results.append({k: detail[k] for k in ("dataset", "cv", "track", "mode", "model", "fold", "coverage", "scored_rows", "elapsed_seconds")} | score)
                    print(json.dumps({"dataset": dataset, "cv": strategy, "track": track, "mode": mode, "model": name, "fold": fold, "score": score, "seconds": round(detail["elapsed_seconds"], 2)}, ensure_ascii=False), flush=True)
    pd.DataFrame(results).to_csv(output / "fold_metrics.csv", index=False)
    write_json(output / "completion.json", {"protocol": PROTOCOL, "models_scored": len(results), "tracks": list(tracks), "modes": list(modes), "folds": list(folds) if folds is not None else list(range(5)), "status": "completed", "metadata_sha256": sha256(output / "metadata.json")})
    return results

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--cv", choices=["group", "spatial"], default="group")
    parser.add_argument("--tracks", type=int, nargs="+", default=[1, 2], choices=[1, 2])
    parser.add_argument("--modes", nargs="+", default=["raw", "augmented"], choices=["raw", "augmented"])
    parser.add_argument("--folds", type=int, nargs="+", choices=range(5))
    parser.add_argument("--step", type=float)
    args = parser.parse_args()
    run_dataset(args.root, args.output, args.dataset, args.cv, args.tracks, args.modes, args.folds, args.step)

if __name__ == "__main__":
    main()
