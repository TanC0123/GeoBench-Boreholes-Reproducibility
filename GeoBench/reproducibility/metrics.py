"""Metrics distinguish borehole weighting, class averaging and structural support."""
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

def macro_f1(matrix):
    matrix = np.asarray(matrix, dtype=float)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    supported = denominator > 0
    return float(np.mean(2 * np.diag(matrix)[supported] / denominator[supported])) if supported.any() else float("nan")

def classification_scores(predictions, vocabulary):
    matrices, holes = [], []
    for bid, group in predictions.groupby("borehole", sort=True):
        matrix = confusion_matrix(group.target, group.prediction, labels=vocabulary).astype(float)
        matrix /= len(group)
        matrices.append(matrix)
        holes.append({"borehole": bid, "n": len(group), "accuracy": float(np.trace(matrix)), "support_macro_f1": float(f1_score(group.target, group.prediction, average="macro", zero_division=0))})
    total = np.sum(matrices, axis=0)
    return {"balanced_macro_f1": macro_f1(total), "balanced_accuracy": float(np.trace(total) / total.sum()), "mean_hole_support_macro_f1": float(np.mean([h["support_macro_f1"] for h in holes])), "point_macro_f1": float(f1_score(predictions.target, predictions.prediction, average="macro", zero_division=0)), "holes": len(holes)}, holes, matrices

def regression_scores(predictions, span, global_order_valid):
    holes = []
    for bid, group in predictions.groupby("borehole", sort=True):
        error = group.prediction.to_numpy(float) - group.target.to_numpy(float)
        # Source ranks are independent metadata. If no justified total order is
        # available, the observed local sequence is only a labelled diagnostic.
        ordered = group.sort_values("rank") if global_order_valid else group.sort_values("target", ascending=False)
        differences = np.diff(ordered.prediction.to_numpy(float))
        holes.append({"borehole": bid, "n": len(group), "rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(np.abs(error))), "order_evaluable": len(group) >= 2, "order_disagreement": bool(np.any(differences > 1e-8)), "violated_pairs": int((differences > 1e-8).sum()), "pairs": len(differences)})
    evaluable = [h for h in holes if h["order_evaluable"]]
    rmse = float(np.mean([h["rmse"] for h in holes]))
    return {"balanced_rmse": rmse, "nrmse_percent": 100 * rmse / span, "balanced_mae": float(np.mean([h["mae"] for h in holes])), "point_rmse": float(np.sqrt(np.mean((predictions.prediction - predictions.target)**2))), "order_disagreement": float(np.mean([h["order_disagreement"] for h in evaluable])) if evaluable else None, "order_evaluable_holes": len(evaluable), "holes": len(holes)}, holes
