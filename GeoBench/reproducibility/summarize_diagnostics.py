"""Rescore all factorial/seed records; retain every seed and interaction."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from .data import sha256
from .metrics import classification_scores,regression_scores


def summarize(source,original,output):
    output.mkdir(parents=True,exist_ok=True)
    records=[];files=reused=neural=0
    for ds in ("Zhengzhou","Kimberlina"):
        for track,cvs,models,variants in [(2,("group",),("RandomForest","ResidualMLP"),("raw","geometry","donor","augmented")),(1,("group","spatial"),("CoordinateMLP",),("raw","augmented"))]:
            for cv in cvs:
                meta=json.loads((original/f"{ds}_{cv}"/"metadata.json").read_text())
                splits=json.loads((original/f"{ds}_{cv}"/"splits.json").read_text());supports={}
                for model in models:
                    for variant in variants:
                        for offset in (0,100,200):
                            frames=[]
                            for fold in range(5):
                                stem=f"track{track}_{variant}_{model}_offset{offset}_fold{fold}"
                                p=source/f"{ds}_{cv}"/"predictions"/f"{stem}.csv.gz"
                                detail=json.loads(p.with_suffix("").with_suffix(".json").read_text())
                                assert sha256(p)==detail["prediction_sha256"]
                                f=pd.read_csv(p);split=splits[fold]
                                assert f.query_id.is_unique and supports.setdefault(fold,set(f.query_id))==set(f.query_id)
                                assert set(f.borehole)<=set(split["test_boreholes"])
                                assert set(f.borehole).isdisjoint(detail["trace"]["outer_donor_boreholes"])
                                assert set(detail["trace"]["inner_donor_boreholes"]).isdisjoint(split["inner_validation_boreholes"])
                                if detail["reused_original"]:
                                    assert sha256(original/detail["original_relative_path"])==sha256(p);reused+=1
                                if track==1:score,_=regression_scores(f,meta["interface_elevation_span"],meta["global_order_valid"])
                                else:score,_,_=classification_scores(f,meta["labels"])
                                for k,v in score.items():assert np.isclose(v,detail["scores"][k],rtol=1e-6,atol=1e-6),(p,k)
                                if detail.get("training"):
                                    assert detail["training"]["outer_refit_rows"]==detail["training_rows"];neural+=1
                                frames.append(f);files+=1
                            oof=pd.concat(frames,ignore_index=True);assert oof.query_id.is_unique
                            if track==1:score,_=regression_scores(oof,meta["interface_elevation_span"],meta["global_order_valid"])
                            else:score,_,_=classification_scores(oof,meta["labels"])
                            value=score["balanced_rmse"] if track==1 else score["balanced_macro_f1"]
                            records.append(dict(dataset=ds,cv=cv,track=track,model=model,variant=variant,seed_offset=offset,score=value,points=len(oof),**score))
    assert (files,reused,neural)==(360,80,240),(files,reused,neural)
    frame=pd.DataFrame(records);frame.to_csv(output/"diagnostic_seed_scores.csv",index=False)
    display=[];effects=[]
    for (ds,cv,track,model,variant),g in frame.groupby(["dataset","cv","track","model","variant"],sort=False):
        display.append(dict(dataset=ds,cv=cv,track=track,model=model,variant=variant,mean=g.score.mean(),low=g.score.min(),high=g.score.max(),seeds=len(g)))
    for (ds,cv,track,model,offset),g in frame.groupby(["dataset","cv","track","model","seed_offset"],sort=False):
        s=g.set_index("variant").score
        values={"full_minus_raw":s.augmented-s.raw}
        if track==2:values.update(geometry_without_donor=s.geometry-s.raw,donor_without_geometry=s.donor-s.raw,geometry_with_donor=s.augmented-s.donor,donor_with_geometry=s.augmented-s.geometry,interaction=s.augmented-s.geometry-s.donor+s.raw)
        effects.append(dict(dataset=ds,cv=cv,track=track,model=model,seed_offset=offset,**values))
    display=pd.DataFrame(display);effects=pd.DataFrame(effects)
    display.to_csv(output/"diagnostic_display.csv",index=False);effects.to_csv(output/"diagnostic_effects.csv",index=False)
    effect_summary=[]
    for keys,g in effects.groupby(["dataset","cv","track","model"],sort=False):
        for name in ("full_minus_raw","geometry_without_donor","donor_without_geometry","geometry_with_donor","donor_with_geometry","interaction"):
            x=g[name].dropna()
            if len(x):effect_summary.append(dict(zip(["dataset","cv","track","model"],keys))|dict(effect=name,mean=x.mean(),low=x.min(),high=x.max(),negative_seeds=int((x<0).sum()),positive_seeds=int((x>0).sum())))
    pd.DataFrame(effect_summary).to_csv(output/"diagnostic_effect_summary.csv",index=False)
    report={"records_verified":files,"original_records_reused_exactly":reused,"new_fits_verified":files-reused,"neural_outer_refits_including_reused":neural,"seed_level_summaries":len(frame),"configuration_summaries":len(display),"scores_recomputed":True,"support_and_fold_separation":True,"all_seeds_retained":True,"new_hypothesis_tests":0}
    (output/"diagnostic_verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report));print(display.to_string(index=False));print(pd.DataFrame(effect_summary).to_string(index=False))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input",type=Path,required=True);p.add_argument("--original",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();summarize(a.input,a.original,a.output)
