"""Native observed-point extension; no fictitious intervals or horizon targets."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import argparse
import hashlib
import json
import time
import zipfile
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.preprocessing import LabelEncoder
from .data import sha256
from .metrics import classification_scores
from .models import PARAMETERS,classifier,fit_predict_mlp,seed_all
from .run import depth_prior,make_splits,write_json

EXPECTED_SHA="e996c8466f2d0dc7116daa71485cc90c31fcfac6345a5bb33165026478b642e9"


def prepare(archive,output,step=5.):
    if sha256(archive)!=EXPECTED_SHA:
        raise ValueError("Input differs from the documented official training archive")
    fields=["WELL","DEPTH_MD","X_LOC","Y_LOC","Z_LOC","FORCE_2020_LITHOFACIES_LITHOLOGY","FORCE_2020_LITHOFACIES_CONFIDENCE"]
    with zipfile.ZipFile(archive) as z:
        with z.open("train.csv") as f:raw=pd.read_csv(f,sep=";",usecols=fields)
    raw["source_row"]=np.arange(len(raw))
    valid=np.isfinite(raw[["DEPTH_MD","X_LOC","Y_LOC","Z_LOC"]].to_numpy()).all(axis=1)&raw[fields[-2]].notna()&raw.WELL.notna()
    clean=raw.loc[valid].copy()
    clean["sampling_bin"]=np.floor(clean.DEPTH_MD/step).astype(int)
    clean["bin_offset"]=(clean.DEPTH_MD-(clean.sampling_bin+.5)*step).abs()
    selected=clean.sort_values(["WELL","sampling_bin","bin_offset","DEPTH_MD","source_row"],kind="stable").drop_duplicates(["WELL","sampling_bin"])
    points=selected.rename(columns={"WELL":"wellbore","DEPTH_MD":"depth","X_LOC":"x","Y_LOC":"y","Z_LOC":"z",fields[-2]:"target",fields[-1]:"interpretation_confidence"})
    # Related sidetracks must stay in one split and one scoring unit.
    points["borehole"]=points.wellbore.str.extract(r"^(\d+/\d+-\d+)",expand=False)
    if points.borehole.isna().any():raise ValueError("Unrecognized source well-family identifier")
    points["target"]=points.target.astype(int).astype(str)
    points["query_id"]="FORCE:"+points.source_row.astype(str)
    points=points[["query_id","source_row","borehole","wellbore","depth","x","y","z","target","interpretation_confidence"]].sort_values(["borehole","wellbore","depth"]).reset_index(drop=True)
    output.mkdir(parents=True,exist_ok=True)
    points.to_csv(output/"points.csv.gz",index=False,compression={"method":"gzip","mtime":0})
    # Native trajectory points are retained for prediction. Only split geometry
    # uses the shallowest available observation, which is not claimed a collar.
    split_locations=clean.sort_values(["WELL","DEPTH_MD"]).drop_duplicates("WELL").rename(columns={"WELL":"borehole","X_LOC":"x","Y_LOC":"y"})[["borehole","x","y"]]
    split_locations["borehole"]=split_locations.borehole.str.extract(r"^(\d+/\d+-\d+)",expand=False)
    split_locations=split_locations.groupby("borehole",as_index=False)[["x","y"]].mean()
    split_records={}
    for cv in ("group","spatial"):
        locations,splits=make_splits(split_locations,cv)
        locations.to_csv(output/f"{cv}_folds.csv",index=False)
        write_json(output/f"{cv}_splits.json",splits)
        split_records[cv]=splits
    metadata={"dataset":"FORCE2020_observed_points","archive_sha256":EXPECTED_SHA,"source_doi":"10.5281/zenodo.4351156","raw_rows":len(raw),"raw_wells":int(raw.WELL.nunique()),"rows_with_coordinate_and_label_support":len(clean),"excluded_rows":int((~valid).sum()),"sampled_rows":len(points),"wells":int(points.borehole.nunique()),"labels":sorted(points.target.unique()),"sampling_step_m":step,"sampling":"nearest observed row to each measured-depth bin centre; no label interpolation","features":["x","y","z"],"excluded_predictors":["well identity","lithology confidence","logs","formation","group"],"track1_applicable":False,"interval_augmentation_applicable":False,"geometry":"native 3D sample coordinates; split reference at shallowest complete-coordinate sample","seed_offsets":[0,100,200],"parameters":PARAMETERS,"points_sha256":sha256(output/"points.csv.gz"),"code_sha256":{n:sha256(Path(__file__).with_name(n)) for n in ("force_external.py","models.py","metrics.py","run.py")}}
    metadata.update(wells=int(points.wellbore.nunique()),scoring_groups=int(points.borehole.nunique()),grouping="source numeric well-family prefix; sidetracks grouped in inner/outer splits and scoring",geometry="native 3D point coordinates; family mean of shallowest complete-coordinate sample positions used only for splitting",shared_family="34/5-1 A and 34/5-1 S")
    write_json(output/"metadata.json",metadata)
    return split_records


def fit_fold(job):
    output,cv,fold=job
    output=Path(output)
    seed_all(42+fold)
    points=pd.read_csv(output/"points.csv.gz",dtype={"target":str})
    metadata=json.loads((output/"metadata.json").read_text())
    split=json.loads((output/f"{cv}_splits.json").read_text())[fold]
    subset=lambda key:points[points.borehole.isin(split[key])].reset_index(drop=True)
    train,test=subset("training_boreholes"),subset("test_boreholes")
    itrain,ival=subset("inner_training_boreholes"),subset("inner_validation_boreholes")
    assert set(train.borehole).isdisjoint(test.borehole)
    assert set(itrain.borehole).isdisjoint(ival.borehole)
    features=["x","y","z"]
    tx,qx,ix,vx=[f[features].to_numpy(float) for f in (train,test,itrain,ival)]
    encoder=LabelEncoder().fit(train.target)
    oy,iy,vy=[encoder.transform(f.target) for f in (train,itrain,ival)]
    jobs=[("DepthPrior",0),("NearestPoint3D",0)]+[(model,offset) for model in ("RandomForest","ResidualMLP") for offset in (0,100,200)]
    records=[]
    for model,offset in jobs:
        seed=42+fold+offset;start=time.monotonic();training=None
        if model=="DepthPrior":prediction=depth_prior(train,test,metadata["sampling_step_m"])
        elif model=="NearestPoint3D":
            _,indices=cKDTree(tx).query(qx,k=1,workers=1)
            prediction=train.target.to_numpy()[indices]
        elif model=="RandomForest":
            fitted=classifier(model,seed);fitted.fit(tx,oy)
            prediction=encoder.inverse_transform(fitted.predict(qx))
        else:
            predicted,training=fit_predict_mlp(ix,iy,vx,vy,tx,oy,qx,2,seed,len(encoder.classes_))
            prediction=encoder.inverse_transform(predicted)
        evidence=test.copy();evidence["prediction"]=prediction;evidence["fold"]=fold;evidence["seed"]=seed;evidence["train_label_seen"]=evidence.target.isin(encoder.classes_)
        scores,_,_=classification_scores(evidence,metadata["labels"])
        directory=output/cv/"predictions";directory.mkdir(parents=True,exist_ok=True)
        stem=f"{model}_seedoffset{offset}_fold{fold}"
        path=directory/f"{stem}.csv.gz"
        evidence.to_csv(path,index=False,compression={"method":"gzip","mtime":0})
        detail={"dataset":metadata["dataset"],"cv":cv,"model":model,"fold":fold,"seed":seed,"seed_offset":offset,"training_rows":len(train),"scored_rows":len(test),"training_boreholes":split["training_boreholes"],"test_boreholes":split["test_boreholes"],"inner_training_boreholes":split["inner_training_boreholes"],"inner_validation_boreholes":split["inner_validation_boreholes"],"features":features if model!="DepthPrior" else ["measured_depth_bin"],"class_vocabulary":encoder.classes_.tolist(),"training":training,"scores":scores,"prediction_sha256":sha256(path),"elapsed_seconds":time.monotonic()-start}
        write_json(directory/f"{stem}.json",detail)
        records.append(detail)
    print(json.dumps({"cv":cv,"fold":fold,"records":len(records),"complete":True}),flush=True)
    return len(records)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,choices=(1,2),default=2)
    args=parser.parse_args()
    prepare(args.archive,args.output)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        counts=[f.result() for f in as_completed([pool.submit(fit_fold,(args.output,cv,fold)) for cv in ("group","spatial") for fold in range(5)])]
    assert sum(counts)==80
    write_json(args.output/"completion.json",{"records":sum(counts),"status":"complete","main_experiments_retrained":False})


if __name__=="__main__":main()
