"""Fold-local features with explicit support and categorical identity encoding."""
import numpy as np
from .data import labels_at_depth

class DonorFeatures:
    def __init__(self, intervals, track, vocabulary):
        if track not in (1, 2):
            raise ValueError("track must be 1 or 2")
        if intervals.empty:
            raise ValueError("At least one training donor is required")
        self.intervals = intervals.copy(deep=True)
        self.track = track
        self.vocabulary = list(vocabulary)
        if len(set(self.vocabulary)) != len(self.vocabulary) or "" in self.vocabulary:
            raise ValueError("Vocabulary must be unique and reserve the empty unknown label")
        self.columns = {bid: group for bid, group in self.intervals.groupby("borehole", sort=True)}
        self.ids = list(self.columns)
        self.coords = np.array([[g.x.iloc[0], g.y.iloc[0]] for g in self.columns.values()])
        distances = np.linalg.norm(self.coords[:, None] - self.coords[None, :], axis=2)
        distances[distances == 0] = np.inf
        nearest = distances.min(axis=1)
        finite = nearest[np.isfinite(nearest)]
        self.radius = max(500.0, 3 * float(np.median(finite))) if len(finite) else 500.0
        self.fill = None

    @property
    def donor_boreholes(self):
        return frozenset(self.ids)

    def feature_names(self, mode="augmented"):
        names = ["x", "y"] if self.track == 1 else ["x", "y", "z"]
        if self.track == 1:
            names += [f"horizon:{name}" for name in self.vocabulary]
        if mode == "raw":
            return names
        names += ["collar", "nearest_other_distance", "donors_within_radius"]
        if self.track == 1:
            return names + ["nearest_supported_boundary", "median_supported_boundary", "boundary_support_present", "boundary_support_fraction_of_six"]
        labels = self.vocabulary + ["unknown"]
        return names + ["depth"] + [f"nearest_label:{name}" for name in labels] + [f"neighbour_fraction:{name}" for name in labels]

    def transform(self, query, mode="augmented", fit=False):
        if mode not in ("raw", "augmented"):
            raise ValueError("mode must be raw or augmented")
        if fit and not set(query.borehole).issubset(self.donor_boreholes):
            raise ValueError("Imputation can only be fit on current training boreholes")
        base_cols = ["x", "y"] if self.track == 1 else ["x", "y", "z"]
        base = query[base_cols].to_numpy(float)
        if self.track == 1:
            onehot = (query.horizon.to_numpy()[:, None] == np.array(self.vocabulary)[None, :]).astype(float)
            base = np.column_stack([base, onehot])
        if mode == "raw":
            return base
        # Two boundary summaries plus missing flags; class summaries have no
        # artificial ordinal relation between material names.
        n_extra = 7 if self.track == 1 else 4 + 2 * (len(self.vocabulary) + 1)
        extra = np.zeros((len(query), n_extra))
        for bid, positions in query.groupby("borehole", sort=False).indices.items():
            q = query.iloc[positions]
            coord = q[["x", "y"]].iloc[0].to_numpy(float)
            dist = np.linalg.norm(self.coords - coord, axis=1)
            order = [i for i in np.argsort(dist, kind="stable") if self.ids[i] != bid]
            common = np.array([float(q.collar.iloc[0]), float(dist[order[0]]) if order else np.nan, sum(dist[i] <= self.radius for i in order)], dtype=float)
            extra[positions, :3] = common
            neighbours = [self.columns[self.ids[i]] for i in order[:6]]
            if self.track == 1:
                donor_values = []
                for neighbour in neighbours:
                    # Real terminal records are excluded from interface donors.
                    usable = neighbour.loc[neighbour.boundary_observed] if "boundary_observed" in neighbour else neighbour
                    lookup = dict(zip(usable.horizon, usable.collar - usable.bottom))
                    donor_values.append(q.horizon.map(lookup).to_numpy(float))
                values = np.array(donor_values).T if donor_values else np.full((len(q), 1), np.nan)
                for local, position in enumerate(positions):
                    finite = values[local, np.isfinite(values[local])]
                    extra[position, 3:] = [finite[0] if len(finite) else np.nan, np.median(finite) if len(finite) else np.nan, float(len(finite) > 0), len(finite) / 6]
            else:
                extra[positions, 3] = q.depth.to_numpy(float)
                values = [labels_at_depth(neighbour, q.depth.to_numpy(float))[0] for neighbour in neighbours]
                labels = self.vocabulary + [""]
                for j, label in enumerate(labels):
                    extra[positions, 4 + j] = (values[0] == label).astype(float) if values else float(label == "")
                    extra[positions, 4 + len(labels) + j] = np.mean([value == label for value in values], axis=0) if values else float(label == "")
        if fit:
            counts = np.isfinite(extra).sum(axis=0)
            self.fill = np.divide(np.nansum(extra, axis=0), counts, out=np.zeros(extra.shape[1]), where=counts > 0)
        if self.fill is None:
            raise RuntimeError("Fit imputation statistics on training queries first")
        extra = np.where(np.isfinite(extra), extra, self.fill[None, :])
        return np.column_stack([base, extra])
