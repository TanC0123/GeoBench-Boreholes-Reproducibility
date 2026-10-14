"""Paired effect sizes and spatial-direction consistency from saved predictions."""
from pathlib import Path
from itertools import product
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from GeoBench.reproducibility.metrics import classification_scores
from GeoBench.reproducibility.analyze import f1_batch
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input",type=Path,default=ROOT/"results/main")
parser.add_argument("--output",type=Path,default=ROOT/".tmp/review_analysis")
args=parser.parse_args()
BASE=args.input
OUT=args.output
OUT.mkdir(parents=True,exist_ok=True)

def load(ds,cv,track,mode,model):
    return pd.concat([pd.read_csv(BASE/f"{ds}_{cv}/predictions/track{track}_{mode}_{model}_fold{f}.csv.gz") for f in range(5)],ignore_index=True)

def values(frame,track,labels):
    if track==1:return np.array([np.sqrt(np.mean((group.prediction-group.target)**2)) for _,group in frame.groupby("borehole",sort=True)])
    return np.array(classification_scores(frame,labels)[2])

def score(v,track):
    return np.asarray(v).mean(axis=-1) if track==1 else f1_batch(np.asarray(v).sum(axis=-3))

rows=[]
for ds in ("Zhengzhou","Kimberlina","SyntheticControl"):
    spatial=pd.read_csv(BASE/f"{ds}_spatial/borehole_folds.csv").set_index("borehole").fold.to_dict()
    metadata=json.loads((BASE/f"{ds}_group/metadata.json").read_text())
    for cv in ("group","spatial"):
        contrasts=[(1,"augmented","CoordinateMLP","augmented","OK","model"),(1,"raw","CoordinateMLP","raw","OK","model"),(2,"augmented","ResidualMLP","augmented","RandomForest","model"),(2,"augmented","RandomForest","raw","NearestObservedColumn","model")]
        contrasts += [(track,"augmented",model,"raw",model,"augmentation") for track,models in [(1,["CoordinateMLP"]),(2,["RandomForest","XGBoost","LightGBM","ResidualMLP"])] for model in models]
        for track,ma,a,mb,b,kind in contrasts:
            first=load(ds,cv,track,ma,a).sort_values(["borehole","query_id"])
            second=load(ds,cv,track,mb,b).sort_values(["borehole","query_id"])
            assert first.query_id.tolist()==second.query_id.tolist()
            holes=sorted(first.borehole.unique());blocks=np.array([spatial[h] for h in holes])
            va,vb=values(first,track,metadata["labels"]),values(second,track,metadata["labels"])
            estimates=[float(score(v,track)) for v in (va,vb)]
            delta=estimates[0]-estimates[1]
            rng=np.random.default_rng(20260918)
            counts=rng.multinomial(len(holes),[1/len(holes)]*len(holes),size=10000)
            def weighted(v,w):
                if track==1:return w@v/w.sum(axis=1)
                return f1_batch(np.einsum("bn,nij->bij",w,v))
            boot=weighted(va,counts)-weighted(vb,counts)
            block_draws=np.array(list(product(range(5),repeat=5)))
            block_counts=np.array([np.bincount(draw,minlength=5) for draw in block_draws])[:,blocks]
            block_boot=weighted(va,block_counts)-weighted(vb,block_counts)
            local=[float(score(va[blocks==f],track)-score(vb[blocks==f],track)) for f in range(5)]
            desired=-1 if track==1 else 1
            rows.append(dict(dataset=ds,cv=cv,track=track,kind=kind,first=f"{a}:{ma}",second=f"{b}:{mb}",first_score=estimates[0],second_score=estimates[1],difference=delta,relative_change_percent=100*delta/estimates[1],hole_low=float(np.quantile(boot,.025)),hole_high=float(np.quantile(boot,.975)),block_low=float(np.quantile(block_boot,.025)),block_high=float(np.quantile(block_boot,.975)),blocks_favouring_first=sum(desired*d>0 for d in local),block_differences=json.dumps(local)))
pd.DataFrame(rows).to_csv(OUT/"paired_effects.csv",index=False)
print(pd.DataFrame(rows).query('cv == "group" and (kind == "model" or first.str.startswith("ResidualMLP") or track == 1)',engine="python")[["dataset","track","kind","first","second","difference","relative_change_percent","hole_low","hole_high","blocks_favouring_first"]].to_string(index=False))
