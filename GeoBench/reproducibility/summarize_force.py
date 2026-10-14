"""Verify all external-domain predictions and report every model seed."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from .data import sha256
from .metrics import classification_scores


def summarize(source,output):
    metadata=json.loads((source/"metadata.json").read_text())
    vocabulary=metadata["labels"]
    records=[];files=refits=0;support={}
    for cv in ("group","spatial"):
        for model in ("DepthPrior","NearestPoint3D","RandomForest","ResidualMLP"):
            for offset in ((0,100,200) if model in ("RandomForest","ResidualMLP") else (0,)):
                frames=[]
                for fold in range(5):
                    stem=f"{model}_seedoffset{offset}_fold{fold}"
                    p=source/cv/"predictions"/f"{stem}.csv.gz"
                    detail=json.loads(p.with_suffix("").with_suffix(".json").read_text())
                    assert sha256(p)==detail["prediction_sha256"]
                    df=pd.read_csv(p,dtype={"target":str,"prediction":str})
                    assert df.query_id.is_unique
                    assert set(df.borehole)==set(detail["test_boreholes"])
                    assert set(df.borehole).isdisjoint(detail["training_boreholes"])
                    assert set(detail["inner_training_boreholes"]).isdisjoint(detail["inner_validation_boreholes"])
                    assert support.setdefault((cv,fold),set(df.query_id))==set(df.query_id)
                    scores,_,_=classification_scores(df,vocabulary)
                    for k,v in scores.items():assert np.isclose(v,detail["scores"][k],rtol=1e-10,atol=1e-10),(p,k)
                    if detail["training"]:
                        assert detail["training"]["outer_refit_rows"]==detail["training_rows"]
                        refits+=1
                    files+=1;frames.append(df)
                oof=pd.concat(frames,ignore_index=True)
                assert len(oof)==metadata["sampled_rows"] and oof.query_id.is_unique
                scores,_,_=classification_scores(oof,vocabulary)
                high=oof[oof.interpretation_confidence==1]
                highscore,_,_=classification_scores(high,vocabulary)
                records.append(dict(cv=cv,model=model,seed_offset=offset,points=len(oof),training_unseen_points=int((~oof.train_label_seen).sum()),**scores,high_confidence_points=len(high),high_confidence_holes=high.borehole.nunique(),high_confidence_macro_f1=highscore["balanced_macro_f1"]))
    output.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(records);frame.to_csv(output/"force_summary.csv",index=False)
    display=[]
    for (cv,model),g in frame.groupby(["cv","model"],sort=False):
        display.append(dict(cv=cv,model=model,seeds=len(g),mean_macro_f1=g.balanced_macro_f1.mean(),min_macro_f1=g.balanced_macro_f1.min(),max_macro_f1=g.balanced_macro_f1.max(),mean_pooled_f1=g.point_macro_f1.mean(),mean_high_confidence_f1=g.high_confidence_macro_f1.mean()))
    pd.DataFrame(display).to_csv(output/"force_display.csv",index=False)
    assert files==80 and refits==30
    verified={"prediction_files":files,"neural_outer_refits":refits,"model_seed_configurations":len(records),"shared_support_verified":True,"fold_separation_verified":True,"scores_recomputed":True,"no_best_seed_selection":True,"metadata":metadata}
    (output/"force_verification.json").write_text(json.dumps(verified,indent=2)+"\n")
    print(pd.DataFrame(display).to_string(index=False))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();summarize(a.input,a.output)
