"""Training-only neighbour support with complete, matched OOF subgroup scoring."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from .data import load_intervals,labels_at_depth,sha256
from .metrics import classification_scores


def classify_support(intervals,queries,training_holes):
    donors={bid:g for bid,g in intervals[intervals.borehole.isin(training_holes)].groupby("borehole",sort=True)}
    if set(queries.borehole)&set(donors):raise ValueError("Query holes must be excluded from donors")
    ids=list(donors);xy=np.array([[g.x.iloc[0],g.y.iloc[0]] for g in donors.values()])
    rows=[]
    for bid,q in queries.groupby("borehole",sort=True):
        dist=np.linalg.norm(xy-q[["x","y"]].iloc[0].to_numpy(),axis=1)
        order=np.argsort(dist,kind="stable")[:6]
        known=np.array([labels_at_depth(donors[ids[i]],q.depth.to_numpy())[1]==1 for i in order])
        counts=known.sum(axis=0)
        for j,query in enumerate(q.itertuples()):
            count=int(counts[j])
            rows.append({"query_id":query.query_id,"borehole":bid,"known_donors":count,"support_bin":"none" if count==0 else "one_two" if count<=2 else "three_six","nearest_donor_supported":bool(known[0,j]),"nearest_training_distance":float(dist[order[0]])})
    return pd.DataFrame(rows)


def analyze(root,source,output):
    output.mkdir(parents=True,exist_ok=True);records=[];coverage=[];hashes=[];checks=[]
    for ds in ("Zhengzhou","Kimberlina"):
        intervals,meta=load_intervals(root,ds)
        for cv in ("group","spatial"):
            directory=source/f"{ds}_{cv}"
            splits=json.loads((directory/"splits.json").read_text())
            points=pd.read_csv(directory/"classification_targets.csv.gz")
            support=[]
            for split in splits:
                q=points[points.borehole.isin(split["test_boreholes"])].copy()
                support.append(classify_support(intervals,q,split["training_boreholes"]))
                # Counterfactual held-out labels are not accessed by the support
                # constructor. This checks the information boundary on real data.
                changed=q.copy();changed["target"]="held_out_perturbation"
                pd.testing.assert_frame_equal(support[-1],classify_support(intervals,changed,split["training_boreholes"]))
                checks.append({"dataset":ds,"cv":cv,"fold":split["fold"],"target_noninterference":True})
            support=pd.concat(support,ignore_index=True)
            assert support.query_id.is_unique and set(support.query_id)==set(points.query_id)
            support.to_csv(output/f"support_{ds}_{cv}.csv.gz",index=False,compression={"method":"gzip","mtime":0})
            for label in ("none","one_two","three_six"):
                g=support[support.support_bin==label]
                coverage.append(dict(dataset=ds,cv=cv,support_bin=label,points=len(g),fraction=len(g)/len(support),holes=g.borehole.nunique(),nearest_supported_fraction=float(g.nearest_donor_supported.mean()) if len(g) else None))
            for mode in ("raw","augmented"):
                for model in ("RandomForest","XGBoost","LightGBM","ResidualMLP","DepthPrior","NearestObservedColumn"):
                    frames=[]
                    for fold in range(5):
                        path=directory/"predictions"/f"track2_{mode}_{model}_fold{fold}.csv.gz"
                        detail=json.loads(path.with_suffix("").with_suffix(".json").read_text())
                        assert sha256(path)==detail["prediction_sha256"]
                        hashes.append({"path":str(path.relative_to(source)),"sha256":sha256(path)})
                        frames.append(pd.read_csv(path))
                    pred=pd.concat(frames,ignore_index=True).merge(support,on=["query_id","borehole"],validate="one_to_one")
                    assert len(pred)==len(points)
                    for label in ("none","one_two","three_six"):
                        g=pred[pred.support_bin==label]
                        if not len(g):continue
                        score,_,_=classification_scores(g,meta["labels"])
                        records.append(dict(dataset=ds,cv=cv,mode=mode,model=model,support_bin=label,points=len(g),true_classes=g.target.nunique(),active_classes=len(set(g.target)|set(g.prediction)),**score))
    pd.DataFrame(records).to_csv(output/"support_scores.csv",index=False)
    pd.DataFrame(coverage).to_csv(output/"support_coverage.csv",index=False)
    report={"bins":["0","1–2","3–6"],"bin_definition":"unambiguous labels among six nearest outer-training donor columns at query depth","checks":checks,"new_model_fits":0,"subgroup_estimand":"borehole-balanced within stratum, with recomputed active-class support; not causal attribution","input_hashes":hashes}
    (output/"support_verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(pd.DataFrame(coverage).to_string(index=False))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();analyze(a.root,a.input,a.output)
