"""Fixed baseline settings and inner-epoch selection followed by outer refitting."""
import random
import numpy as np
import torch
from torch import nn
from scipy.interpolate import Rbf
from pykrige.ok import OrdinaryKriging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

PARAMETERS = {
    "IDW": {"power": 2}, "RBF": {"function": "multiquadric", "smooth": 0.5, "epsilon": "training_median_nonzero_distance"},
    "OK": {"variogram_model": "linear", "parameters": "fitted_on_training_horizon"},
    "RandomForest": {"n_estimators": 200, "max_depth": 20, "max_features": "sqrt"},
    "XGBoost": {"n_estimators": 200, "max_depth": 8, "learning_rate": 0.1, "tree_method": "hist"},
    "LightGBM": {"n_estimators": 200, "max_depth": 15, "num_leaves": 31, "learning_rate": 0.1},
    "MLP": {"hidden": 128, "residual_blocks": 2, "dropout": 0.1, "max_epochs": 120, "patience": 15, "min_delta": 0.0001, "batch_size": 256, "learning_rate": 0.001, "weight_decay": 0.0001, "selection_loss": "unweighted_MSE_or_cross_entropy", "outer_refit": True, "schedule": "constant_learning_rate"},
}

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

def interpolate(name, xy, target, query):
    xy, target, query = np.asarray(xy, float), np.asarray(target, float), np.asarray(query, float)
    if name == "IDW":
        distances = np.linalg.norm(query[:, None, :] - xy[None, :, :], axis=2)
        weights = 1 / np.maximum(distances, 1e-12)**2
        result = (weights / weights.sum(axis=1, keepdims=True)) @ target
        exact = distances <= 1e-12
        for i in np.flatnonzero(exact.any(axis=1)):
            result[i] = target[np.flatnonzero(exact[i])[0]]
        return result
    if name == "RBF":
        pairwise = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
        epsilon = float(np.median(pairwise[pairwise > 0]))
        model = Rbf(xy[:, 0], xy[:, 1], target, function="multiquadric", smooth=0.5, epsilon=epsilon)
        return model(query[:, 0], query[:, 1])
    if name == "OK":
        model = OrdinaryKriging(xy[:, 0], xy[:, 1], target, variogram_model="linear", verbose=False, enable_plotting=False)
        return np.asarray(model.execute("points", query[:, 0], query[:, 1])[0])
    raise ValueError(name)

def classifier(name, seed):
    settings = dict(PARAMETERS[name], random_state=seed, n_jobs=1)
    if name == "RandomForest":
        return RandomForestClassifier(**settings)
    if name == "XGBoost":
        return XGBClassifier(**settings, verbosity=0, eval_metric="mlogloss")
    if name == "LightGBM":
        return LGBMClassifier(**settings, verbosity=-1, deterministic=True, force_col_wise=True)
    raise ValueError(name)

class ResidualBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden))
    def forward(self, x):
        return torch.relu(x + self.layers(x))

class ResidualMLP(nn.Module):
    def __init__(self, features, outputs):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(features, 128), nn.BatchNorm1d(128), nn.ReLU(), ResidualBlock(128), ResidualBlock(128), nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, outputs))
    def forward(self, x):
        return self.layers(x)

def _train(x, y, outputs, track, seed, epochs, validation=None):
    seed_all(seed)
    model = ResidualMLP(x.shape[1], outputs)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_function = nn.MSELoss() if track == 1 else nn.CrossEntropyLoss()
    xt = torch.as_tensor(x, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32 if track == 1 else torch.int64)
    rng = np.random.default_rng(seed)
    best, best_epoch, bad = float("inf"), 1, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        # Balanced chunk sizes avoid a one-row BatchNorm batch without silently
        # dropping observations.
        chunks = np.array_split(rng.permutation(len(x)), max(1, int(np.ceil(len(x) / 256))))
        for indices in chunks:
            optimizer.zero_grad()
            prediction = model(xt[indices])
            if track == 1:
                prediction = prediction[:, 0]
            loss = loss_function(prediction, yt[indices])
            loss.backward()
            optimizer.step()
        if validation is not None:
            vx, vy = validation
            model.eval()
            with torch.no_grad():
                prediction = model(torch.as_tensor(vx, dtype=torch.float32))
                if track == 1:
                    prediction = prediction[:, 0]
                value = float(loss_function(prediction, torch.as_tensor(vy, dtype=yt.dtype)))
            if not np.isfinite(value):
                raise RuntimeError("Non-finite inner validation loss")
            history.append(value)
            if value < best - 0.0001:
                best, best_epoch, bad = value, epoch, 0
            else:
                bad += 1
            if bad >= 15:
                break
    return model, best_epoch, history

def fit_predict_mlp(inner_x, inner_y, validation_x, validation_y, outer_x, outer_y, test_x, track, seed, outputs):
    inner_scaler = StandardScaler().fit(inner_x)
    outer_scaler = StandardScaler().fit(outer_x)
    if track == 1:
        inner_target = StandardScaler().fit(np.asarray(inner_y).reshape(-1, 1))
        outer_target = StandardScaler().fit(np.asarray(outer_y).reshape(-1, 1))
        iy = inner_target.transform(np.asarray(inner_y).reshape(-1, 1)).ravel()
        vy = inner_target.transform(np.asarray(validation_y).reshape(-1, 1)).ravel()
        oy = outer_target.transform(np.asarray(outer_y).reshape(-1, 1)).ravel()
    else:
        iy, vy, oy = inner_y, validation_y, outer_y
    _, chosen_epoch, history = _train(inner_scaler.transform(inner_x), iy, outputs, track, seed, 120, (inner_scaler.transform(validation_x), vy))
    model, _, _ = _train(outer_scaler.transform(outer_x), oy, outputs, track, seed, chosen_epoch)
    model.eval()
    predictions = []
    test = outer_scaler.transform(test_x)
    with torch.no_grad():
        for start in range(0, len(test), 4096):
            predictions.append(model(torch.as_tensor(test[start:start + 4096], dtype=torch.float32)).numpy())
    prediction = np.concatenate(predictions)
    if track == 1:
        prediction = outer_target.inverse_transform(prediction[:, :1]).ravel()
    else:
        prediction = prediction.argmax(axis=1)
    return prediction, {"selected_epoch": chosen_epoch, "inner_loss_history": history, "outer_refit_rows": len(outer_x)}
