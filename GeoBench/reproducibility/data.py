"""Source-preserving data adapters for the reconstructed GeoBench protocol."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

DATASETS = ("Zhengzhou", "Kimberlina", "SyntheticControl")
STEPS = {"Zhengzhou": 0.5, "Kimberlina": 5.0, "SyntheticControl": 0.5}

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _code(value):
    return str(int(value)) if float(value).is_integer() else str(value)

def load_intervals(root, name):
    """Return source intervals and metadata without consulting any result cache."""
    root = Path(root)
    sources = {}
    if name == "Zhengzhou":
        directory = root / "Drilling_and_distributio_data"
        paths = [directory / "基本情况表.xlsx", directory / "分层信息表.xlsx"]
        base, layers = [pd.read_excel(path) for path in paths]
        base = base.dropna(subset=["钻孔编号"]).drop_duplicates("钻孔编号")
        base = base[["钻孔编号", "X(m)", "Y(m)", "高程(m)", "孔深"]].rename(columns={"钻孔编号": "borehole", "X(m)": "x", "Y(m)": "y", "高程(m)": "collar", "孔深": "penetration"})
        layers = layers.dropna(subset=["钻孔编号", "标准层号", "标准层号.1", "层底深度", "岩性"])
        intervals = layers[["钻孔编号", "标准层号", "标准层号.1", "层底深度", "岩性"]].rename(columns={"钻孔编号": "borehole", "标准层号": "horizon", "标准层号.1": "rank", "层底深度": "bottom", "岩性": "label"}).merge(base, on="borehole", validate="many_to_one")
        intervals["horizon"] = intervals.horizon.map(_code)
        intervals["rank"] = intervals["rank"].astype(int)
        intervals = intervals.sort_values(["borehole", "bottom"]).reset_index(drop=True)
        intervals["top"] = intervals.groupby("borehole").bottom.shift(1).fillna(0.0)
        for _, group in intervals.groupby("borehole"):
            if np.any(np.diff(group["rank"]) <= 0):
                raise ValueError("Source horizon ranks contradict a Zhengzhou column")
        if intervals.groupby("horizon")["rank"].nunique().max() != 1:
            raise ValueError("Ambiguous source horizon rank")
        metadata = {"interval_top_basis": "previous_recorded_bottom; first_record_assumed_to_start_at_collar", "global_order_valid": True, "data_provider": "Xiaojie Zhai, Henan Institute of Geological Research (河南省地质研究院)", "coordinate_units": "m_as_labelled_in_source; projected_CRS_not_documented"}
    elif name == "Kimberlina":
        paths = [root / "data/kim_ready.csv"]
        raw = pd.read_csv(paths[0])
        raw = raw[raw.formation.notna() & raw.formation.astype(str).str.strip().ne("topo")].copy()
        intervals = raw[["name", "x", "y", "altitude", "top", "base", "formation"]].rename(columns={"name": "borehole", "altitude": "collar", "base": "bottom", "formation": "label"})
        intervals["horizon"] = intervals.label.astype(str)
        intervals["rank"] = np.nan
        intervals["penetration"] = np.nan
        metadata = {"interval_top_basis": "explicit_source_top", "global_order_valid": False, "data_provider": "Software Underground Subsurface borehole-derived formation-contact table", "coordinate_units": "m", "horizontal_crs": "NAD27 / UTM zone 11N", "unit_provenance": "kimberlina_units.md; raw contact header and numerical reconstruction of all 583 used intervals", "source_interpretation": "Imported formation labels and documented upstream repairs are retained; geological reinterpretation is outside this adapter."}
    elif name == "SyntheticControl":
        paths = []
        rng = np.random.RandomState(123)
        xvals = np.linspace(0, 3000, 9) + rng.randn(9) * 30
        yvals = np.linspace(0, 2000, 5) + rng.randn(5) * 20
        xx, yy = np.meshgrid(xvals, yvals)
        collars = 80 + yy.ravel() / 2000 * 30 + rng.randn(45) * 2
        rows = []
        for index, (x, y, collar) in enumerate(zip(xx.ravel(), yy.ravel(), collars)):
            depth = 0.0
            for rank, base_thickness in enumerate([4, 6, 8, 5, 10, 7, 6], 1):
                thickness = max(1.0, base_thickness + rng.randn() * 1.2 + (x * np.tan(np.deg2rad(8)) + y * np.tan(np.deg2rad(3))) * (rank - 4) * 0.15)
                rows.append({"borehole": f"SYN-{index + 1:03d}", "x": x, "y": y, "collar": collar, "top": depth, "bottom": depth + thickness, "horizon": f"h{rank}", "label": f"class{rank}", "rank": rank, "penetration": np.nan})
                depth += thickness
        intervals = pd.DataFrame(rows)
        metadata = {"interval_top_basis": "generated_known_intervals", "global_order_valid": True, "data_provider": "deterministic thickness-varying synthetic control, seed 123", "coordinate_units": "m"}
    else:
        raise ValueError(name)
    for path in paths:
        sources[str(path.relative_to(root))] = sha256(path)
    intervals = intervals.sort_values(["borehole", "top", "bottom"]).reset_index(drop=True)
    if intervals[["x", "y", "collar", "top", "bottom"]].isna().any().any():
        raise ValueError("Missing required interval coordinates")
    if (intervals.bottom <= intervals.top).any() or (intervals.top < 0).any():
        raise ValueError("Invalid source interval")
    if intervals.duplicated(["borehole", "horizon"]).any():
        raise ValueError("Repeated horizon identity in a source borehole")
    intervals["source_row"] = np.arange(len(intervals))
    intervals["boundary_observed"] = True
    if name != "SyntheticControl":
        terminal = intervals.bottom == intervals.groupby("borehole").bottom.transform("max")
        intervals.loc[terminal, "boundary_observed"] = False
    metadata.update({"sources": sources, "dataset": name, "source_rows": len(intervals), "boreholes": int(intervals.borehole.nunique()), "labels": sorted(intervals.label.unique().tolist()), "horizons": sorted(intervals.horizon.unique().tolist())})
    return intervals, metadata

def labels_at_depth(intervals, depths):
    """Return observed labels; never extend a donor beyond its logged interval."""
    depths = np.asarray(depths)
    matches = (depths[:, None] >= intervals.top.to_numpy()[None, :]) & (depths[:, None] < intervals.bottom.to_numpy()[None, :])
    answer = np.full(len(depths), "", dtype=object)
    distinct = np.zeros(len(depths), dtype=int)
    for label in sorted(intervals.label.unique()):
        found = matches[:, intervals.label.to_numpy() == label].any(axis=1)
        answer[found] = label
        distinct += found
    answer[distinct != 1] = ""
    return answer, distinct

def build_queries(intervals, metadata, step=None):
    """Build boundary targets and collar-anchored fixed-depth classification queries."""
    name = metadata["dataset"]
    step = float(STEPS[name] if step is None else step)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Sampling step must be finite and positive")
    interfaces = intervals.copy()
    terminal = interfaces.groupby("borehole").bottom.transform("max") == interfaces.bottom
    excluded_terminal = int(terminal.sum()) if name != "SyntheticControl" else 0
    if name != "SyntheticControl":
        interfaces = interfaces.loc[~terminal].copy()
    interfaces["target"] = interfaces.collar - interfaces.bottom
    interfaces["query_id"] = interfaces.borehole.astype(str) + ":" + interfaces.horizon
    rows = []
    unknown = ambiguous = total = 0
    for borehole, group in intervals.groupby("borehole", sort=True):
        # The observed depth range defines evaluation support, not a predictor.
        grid = (np.arange(int(np.ceil(group.bottom.max() / step))) + 0.5) * step
        grid = grid[grid < group.bottom.max()]
        labels, support = labels_at_depth(group, grid)
        unknown += int((support == 0).sum())
        ambiguous += int((support > 1).sum())
        total += len(grid)
        mask = support == 1
        first = group.iloc[0]
        rows.append(pd.DataFrame({"borehole": borehole, "x": first.x, "y": first.y, "collar": first.collar, "depth": grid[mask], "z": first.collar - grid[mask], "target": labels[mask], "query_id": [f"{borehole}:{i}" for i in np.flatnonzero(mask)]}))
    points = pd.concat(rows, ignore_index=True)
    # Metadata vocabulary is fixed by source labels; classes with no sampled
    # truth are reported explicitly and excluded from the metric denominator.
    metadata = dict(metadata, step=step, boundary_rows=len(interfaces), excluded_terminal_boundaries=excluded_terminal, labelled_points=len(points), grid_points=total, unobserved_points=unknown, ambiguous_points=ambiguous, sampled_labels=sorted(points.target.unique().tolist()))
    return interfaces.reset_index(drop=True), points, metadata
