"""Matched feature-block and seed diagnostics on immutable empirical splits."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import json
import platform
import shutil
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .data import load_intervals,build_queries,sha256
from .metrics import classification_scores,regression_scores
from .models import PARAMETERS,classifier,fit_predict_mlp,seed_all
from .run import feature_bundle,write_json


def select_block(names,variant,track):
    if track==1:
        return [i for i,n in enumerate(names) if variant=="augmented" or n in ("x","y") or n.startswith("horizon:")]
    raw={"x","y","z"}
    geometry={"collar","depth","nearest_other_distance","donors_within_radius"}
    if variant not in ("raw","geometry","donor","augmented"):raise ValueError(variant)
    return [i for i,n in enumerate(names) if n in raw or (variant in ("geometry","augmented") and n in geometry) or (variant in ("donor","augmented") and n.startswith(("nearest_label:","neighbour_fraction:")))]


def fit_job(job):
    root,source,output,ds,cv,track,fold=job
    root,source,output=map(Path,(root,source,output))
    seed_all(42+fold)
    directory=source/f"{ds}_{cv}"
    split=json.loads((directory/"splits.json").read_text())[fold]
    intervals,metadata=load_intervals(root,ds)
    boundaries,points,metadata=build_queries(intervals,metadata)
    frame=boundaries if track==1 else points
    subset=lambda key:frame[frame.borehole.isin(split[key])].reset_index(drop=True)
    train,test=subset("training_boreholes"),subset("test_boreholes")
    if track==1:
        counts=train.groupby("horizon").borehole.nunique();test=test[test.horizon.isin(counts[counts>=3].index)].reset_index(drop=True)
    itrain,ival=subset("inner_training_boreholes"),subset("inner_validation_boreholes")
    vocabulary=metadata["horizons"] if track==1 else metadata["labels"]
    arrays,trace=feature_bundle(intervals,train,test,itrain,ival,vocabulary,track,"augmented")
    names=trace["feature_names"]
    span=float(boundaries.target.max()-boundaries.target.min())
    encoder=LabelEncoder().fit(train.target) if track==2 else None
    oy,iy,vy=[encoder.transform(f.target) if encoder is not None else f.target.to_numpy() for f in (train,itrain,ival)]
    variants=("raw","geometry","donor","augmented") if track==2 else ("raw","augmented")
    models=("RandomForest","ResidualMLP") if track==2 else ("CoordinateMLP",)
    created=reused=0
    for variant in variants:
        indices=select_block(names,variant,track)
        tx,qx,ix,vx=[a[:,indices] for a in arrays]
        for model in models:
            for offset in (0,100,200):
                stem=f"track{track}_{variant}_{model}_offset{offset}_fold{fold}"
                dest=output/f"{ds}_{cv}"/"predictions";dest.mkdir(parents=True,exist_ok=True)
                path=dest/f"{stem}.csv.gz";recordpath=dest/f"{stem}.json"
                start=time.monotonic();seed=42+fold+offset
                original=directory/"predictions"/f"track{track}_{variant}_{model}_fold{fold}.csv.gz"
                if offset==0 and variant in ("raw","augmented"):
                    old=json.loads(original.with_suffix("").with_suffix(".json").read_text())
                    assert sha256(original)==old["prediction_sha256"]
                    existing=pd.read_csv(original)
                    assert existing.query_id.tolist()==test.query_id.tolist()
                    if track==1:np.testing.assert_allclose(existing.target,test.target,rtol=0,atol=1e-8)
                    else:assert existing.target.tolist()==test.target.tolist()
                    assert old["trace"]["feature_names"]==[names[i] for i in indices]
                    assert set(old["trace"]["outer_donor_boreholes"])==set(trace["outer_donor_boreholes"])
                    shutil.copy2(original,path)
                    detail=dict(old,variant=variant,seed_offset=offset,reused_original=True,original_relative_path=str(original.relative_to(source)),original_prediction_sha256=sha256(original),selected_feature_names=[names[i] for i in indices])
                    write_json(recordpath,detail);reused+=1;continue
                training=None
                if model=="RandomForest":
                    fitted=classifier(model,seed);fitted.fit(tx,oy)
                    prediction=encoder.inverse_transform(fitted.predict(qx))
                else:
                    prediction,training=fit_predict_mlp(ix,iy,vx,vy,tx,oy,qx,track,seed,1 if track==1 else len(encoder.classes_))
                    if encoder is not None:prediction=encoder.inverse_transform(prediction)
                columns=["query_id","borehole","x","y","collar","target"]+(["horizon","rank"] if track==1 else ["depth","z"])
                evidence=test[columns].copy();evidence["prediction"]=prediction;evidence["fold"]=fold
                evidence["train_label_seen"]=True if encoder is None else evidence.target.isin(encoder.classes_)
                evidence.to_csv(path,index=False,compression={"method":"gzip","mtime":0})
                if track==1:scores,_=regression_scores(evidence,span,metadata["global_order_valid"])
                else:scores,_,_=classification_scores(evidence,metadata["labels"])
                detail={"dataset":ds,"cv":cv,"track":track,"variant":variant,"model":model,"fold":fold,"seed":seed,"seed_offset":offset,"reused_original":False,"training_rows":len(train),"scored_rows":len(test),"class_vocabulary":encoder.classes_.tolist() if encoder is not None else [],"trace":trace,"selected_feature_names":[names[i] for i in indices],"training":training,"scores":scores,"prediction_sha256":sha256(path),"elapsed_seconds":time.monotonic()-start}
                write_json(recordpath,detail);created+=1
    print(json.dumps({"dataset":ds,"cv":cv,"track":track,"fold":fold,"new":created,"reused":reused}),flush=True)
    return created,reused


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--workers",type=int,choices=(1,2,3),default=2)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    metadata={"analysis":"exploratory matched feature blocks and seed sensitivity","parameters":PARAMETERS,"python":platform.python_version(),"seed_offsets":[0,100,200],"source_results":a.input.name,"new_fits_planned":280,"original_records_reused":80,"classification":"both empirical datasets, original group folds, RF and ResidualMLP, four blocks","regression":"both empirical datasets and split schemes, CoordinateMLP, raw/full","code_sha256":{n:sha256(Path(__file__).with_name(n)) for n in ("feature_diagnostics.py","features.py","models.py","data.py","run.py","metrics.py")}}
    write_json(a.output/"metadata.json",metadata)
    jobs=[(a.root,a.input,a.output,ds,cv,track,fold) for ds in ("Zhengzhou","Kimberlina") for track,schemes in ((2,("group",)),(1,("group","spatial"))) for cv in schemes for fold in range(5)]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        results=[f.result() for f in as_completed([pool.submit(fit_job,j) for j in jobs])]
    new=sum(r[0] for r in results);reused=sum(r[1] for r in results)
    assert (new,reused)==(280,80),(new,reused)
    write_json(a.output/"completion.json",{"new_predictions":new,"reused_predictions":reused,"records":new+reused,"status":"complete"})


if __name__=="__main__":main()
