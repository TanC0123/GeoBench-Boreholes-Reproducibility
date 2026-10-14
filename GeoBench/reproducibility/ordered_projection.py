"""A source-order-only isotonic reference for saved boundary predictions."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.optimize import isotonic_regression
from .data import sha256
from .metrics import regression_scores


def project_column(predictions, ranks):
    """Project in source-rank order; true elevations are intentionally absent."""
    predictions=np.asarray(predictions,dtype=float)
    ranks=np.asarray(ranks,dtype=float)
    if predictions.shape!=ranks.shape or predictions.ndim!=1:
        raise ValueError("Predictions and source ranks must be matching vectors")
    if not np.isfinite(predictions).all() or not np.isfinite(ranks).all():
        raise ValueError("Finite predictions and independently supplied ranks required")
    if len(np.unique(ranks))!=len(ranks):
        raise ValueError("Repeated ranks need a separate structural definition")
    order=np.argsort(ranks,kind="stable")
    projected=np.empty_like(predictions)
    projected[order]=isotonic_regression(predictions[order],increasing=False).x
    return projected


def analyze(source,output):
    output.mkdir(parents=True,exist_ok=True)
    summaries=[];column_records=[];files=[]
    for ds in ("Zhengzhou","SyntheticControl"):
        for cv in ("group","spatial"):
            directory=source/f"{ds}_{cv}"
            metadata=json.loads((directory/"metadata.json").read_text())
            assert metadata["global_order_valid"]
            for mode in ("raw","augmented"):
                for model in ("IDW","RBF","OK","CoordinateMLP"):
                    frames=[]
                    for fold in range(5):
                        original=directory/"predictions"/f"track1_{mode}_{model}_fold{fold}.csv.gz"
                        record=json.loads(original.with_suffix("").with_suffix(".json").read_text())
                        assert sha256(original)==record["prediction_sha256"]
                        frame=pd.read_csv(original)
                        frame["original_prediction"]=frame.prediction
                        for hole,g in frame.groupby("borehole",sort=True):
                            projected=project_column(g.prediction,g["rank"])
                            frame.loc[g.index,"prediction"]=projected
                            order=np.argsort(g["rank"].to_numpy(),kind="stable")
                            truth=g.target.to_numpy()[order]
                            assert (np.diff(truth)<=1e-8).all()
                            old_rmse=float(np.sqrt(np.mean((g.prediction-g.target)**2)))
                            new_rmse=float(np.sqrt(np.mean((projected-g.target)**2)))
                            assert new_rmse<=old_rmse+1e-8
                            diffs=np.diff(projected[order])
                            assert (diffs<=1e-8).all()
                            column_records.append(dict(dataset=ds,cv=cv,mode=mode,model=model,fold=fold,borehole=hole,n=len(g),original_rmse=old_rmse,projected_rmse=new_rmse,tied_pairs=int((np.abs(diffs)<=1e-8).sum()),pairs=len(diffs),changed=bool(np.any(np.abs(projected-g.prediction)>1e-8))))
                        destination=output/f"{ds}_{cv}"/"predictions"/original.name
                        destination.parent.mkdir(parents=True,exist_ok=True)
                        frame.to_csv(destination,index=False,compression={"method":"gzip","mtime":0})
                        files.append({"file":str(destination.relative_to(output)),"sha256":sha256(destination),"original_prediction_sha256":sha256(original)})
                        frames.append(frame)
                    projected=pd.concat(frames,ignore_index=True)
                    original=projected.copy();original["prediction"]=original.original_prediction
                    before,_=regression_scores(original,metadata["interface_elevation_span"],True)
                    after,_=regression_scores(projected,metadata["interface_elevation_span"],True)
                    holes=[h for h in column_records if (h["dataset"],h["cv"],h["mode"],h["model"])==(ds,cv,mode,model)]
                    summaries.append(dict(dataset=ds,cv=cv,mode=mode,model=model,original_rmse=before["balanced_rmse"],projected_rmse=after["balanced_rmse"],change_rmse=after["balanced_rmse"]-before["balanced_rmse"],original_order_disagreement=before["order_disagreement"],projected_order_disagreement=after["order_disagreement"],tied_pair_fraction=sum(h["tied_pairs"] for h in holes)/sum(h["pairs"] for h in holes),changed_columns=sum(h["changed"] for h in holes),columns=len(holes)))
    pd.DataFrame(summaries).to_csv(output/"summary.csv",index=False)
    pd.DataFrame(column_records).to_csv(output/"columns.csv",index=False)
    report={"analysis":"exploratory source-order isotonic reference; no model retraining","algorithm":"scipy.optimize.isotonic_regression, non-increasing, equal weights","not_applicable":"Kimberlina: no independently verified total source order","prediction_files":len(files),"configurations":len(summaries),"per_column_nonincrease_checks":len(column_records),"projection_never_receives_targets":True,"known_boundary_query_population_only":True,"weak_order_can_collapse_layers":True,"files":files}
    (output/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();analyze(args.input,args.output)
