"""Compare original and repair-exclusion fits on exactly matched targets."""
import argparse
import json
from pathlib import Path

import pandas as pd
from .data import sha256
from .metrics import regression_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--sensitivity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for cv in ("group", "spatial"):
        base = args.sensitivity / cv
        meta = json.loads((base / "metadata.json").read_text())
        for mode in ("raw", "augmented"):
            for model in ("IDW", "RBF", "OK", "CoordinateMLP"):
                chunks = []
                originals = []
                for fold in range(5):
                    name = f"track1_{mode}_{model}_fold{fold}.csv.gz"
                    path = base / "predictions" / name
                    detail = json.loads(path.with_suffix("").with_suffix(".json").read_text())
                    assert sha256(path) == detail["prediction_sha256"]
                    chunks.append(pd.read_csv(path))
                    originals.append(pd.read_csv(args.main / f"Kimberlina_{cv}" / "predictions" / name))
                now = pd.concat(chunks, ignore_index=True)
                old = pd.concat(originals, ignore_index=True)
                keys = set(now.query_id) & set(old.query_id)
                span = meta["interface_elevation_span"]
                score, _ = regression_scores(now, span, False)
                old_common, _ = regression_scores(old[old.query_id.isin(keys)], span, False)
                new_common, _ = regression_scores(now[now.query_id.isin(keys)], span, False)
                records.append(dict(cv=cv, mode=mode, model=model,
                    excluded_artificial_nonterminal=meta["excluded_upstream_repaired_nonterminal_boundaries"],
                    scored_boundaries=len(now), **score, common_boundaries=len(keys),
                    original_on_common=old_common["balanced_rmse"],
                    sensitivity_on_common=new_common["balanced_rmse"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output, index=False)
    print(f"Verified and summarized {len(records)} sensitivity configurations.")


if __name__ == "__main__":
    main()
