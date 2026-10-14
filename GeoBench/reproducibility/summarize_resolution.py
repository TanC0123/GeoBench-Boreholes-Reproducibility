"""Validate saved resolution runs and recompute their classification summaries."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import sha256
from .metrics import classification_scores

def summarize(root, output):
    rows, files, refits = [], 0, 0
    for folder in sorted(Path(root).glob("*_step*")):
        if not folder.is_dir():
            continue
        done = json.loads((folder / "completion.json").read_text())
        assert done["models_scored"] == 60 and done["folds"] == list(range(5))
        metadata = json.loads((folder / "metadata.json").read_text())
        splits = {s["fold"]: s for s in json.loads((folder / "splits.json").read_text())}
        configs, support = {}, {}
        for record in sorted((folder / "predictions").glob("*.json")):
            detail = json.loads(record.read_text())
            csv = record.with_suffix(".csv.gz")
            assert sha256(csv) == detail["prediction_sha256"]
            frame = pd.read_csv(csv)
            fold = detail["fold"]
            assert support.setdefault(fold, set(frame.query_id)) == set(frame.query_id)
            split = splits[fold]
            assert set(frame.borehole) <= set(split["test_boreholes"])
            assert set(detail["trace"]["outer_donor_boreholes"]).isdisjoint(frame.borehole)
            assert set(detail["trace"]["inner_donor_boreholes"]).isdisjoint(split["inner_validation_boreholes"])
            scores, _, _ = classification_scores(frame, metadata["labels"])
            assert all(np.isclose(value, detail["scores"][key], atol=1e-10) for key, value in scores.items())
            if "training" in detail:
                assert detail["training"]["outer_refit_rows"] == detail["training_rows"]
                refits += 1
            configs.setdefault((detail["mode"], detail["model"]), []).append(frame)
            files += 1
        for (mode, model), chunks in configs.items():
            assert len(chunks) == 5
            frame = pd.concat(chunks, ignore_index=True)
            assert frame.query_id.is_unique
            scores, _, _ = classification_scores(frame, metadata["labels"])
            rows.append({"dataset": metadata["dataset"], "step": metadata["step"], "mode": mode, "model": model, "labelled_points": metadata["labelled_points"], "sampled_classes": len(metadata["sampled_labels"]), "unobserved_points": metadata["unobserved_points"], "ambiguous_points": metadata["ambiguous_points"], **scores})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    report = {"prediction_files_verified": files, "neural_outer_refits_verified": refits, "configuration_summaries": len(rows), "common_support_and_fold_separation": True, "scores_recomputed": True}
    output.with_suffix(".verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summarize(args.input, args.output)
