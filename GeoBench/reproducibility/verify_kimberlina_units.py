"""Trace cached interval depths to metre-valued upstream contact elevations."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

import argparse
ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--sources",type=Path,default=ROOT/"GeoBench/provenance/kimberlina")
parser.add_argument("--output",type=Path,default=ROOT/".tmp/kimberlina_units_verified.json")
args=parser.parse_args()
S=args.sources
original=pd.read_csv(S/"doggr_source.dat",sep="\t",skiprows=41,header=None,dtype=str)
original["name"]=original[3]+original[4]
names=["topo","etchegoin","macoma","chanac","mclure","santa_margarita","fruitvale","round_mountain","olcese","freeman_jewett","vedder","eocene","cretaceous","basement","null"]
reconstructed=[]
last_elevation=None
for _,hole in original.iterrows():
    previous=0.
    collar=float(hole[10])
    for position,label in enumerate(names):
        elevation=float(hole[11+position])
        if elevation==last_elevation:
            continue
        last_elevation=elevation
        depth=collar-elevation
        top=previous
        previous=depth
        corrected=top>depth
        bottom=top+.01 if corrected else depth
        reconstructed.append(dict(wellid=str(hole[0]),name=hole["name"],formation=label,x=float(hole[1]),y=float(hole[2]),altitude=collar,top=top,base=bottom,upstream_reversed_interval_repair=corrected,source_contact_elevation=elevation))
expected=pd.DataFrame(reconstructed)
cached=pd.read_csv(ROOT/"data/kim_ready.csv",keep_default_na=False)
cached["formation"]=cached.formation.replace("","null")
cached["wellid"]=cached["0"].astype(str)
merged=expected.merge(cached,on=["wellid","formation"],how="outer",suffixes=("_reconstructed","_cached"),indicator=True,validate="one_to_one")
counts=merged._merge.value_counts().to_dict()
maxerrors={c:float(np.max(np.abs(merged[c+"_reconstructed"].to_numpy(float)-merged[c+"_cached"].to_numpy(float)))) for c in ["x","y","altitude","top","base"]}
assert counts.get("both")==len(cached)==len(expected)
retained=merged[~merged.formation.isin(["topo","null"])]
mask=np.zeros(len(merged),dtype=bool)
for c in ["x","y","altitude","top","base"]:mask|=~np.isclose(merged[c+"_reconstructed"],merged[c+"_cached"],atol=1e-8,rtol=0)
excluded_differences=merged.loc[mask,["wellid","name_cached","formation","top_reconstructed","top_cached","base_reconstructed","base_cached"]].to_dict("records")
retained_errors={c:float(np.max(np.abs(retained[c+"_reconstructed"].to_numpy(float)-retained[c+"_cached"].to_numpy(float)))) for c in ["x","y","altitude","top","base"]}
assert all(v<1e-8 for v in retained_errors.values()),retained_errors
report={"date":"2026-09-18","upstream_commit":"816db75bad6f1eda71209a4635490d2969eadeb0","upstream_source":"tests/data/borehole/doggr_jlw_vedder_final.utm.dat","units":"metres","horizontal_crs":"NAD27 / UTM zone 11N (header: Universal Transverse Mercator, Zone 11, Clarke 1866/NAD27)","header_lines":"11: topo meters; 12–26: named horizons meters; Projection/Zone/Units/Ellipsoid near end of header","conversion":"base_depth = collar_elevation - source_contact_elevation; top_depth = previous retained base_depth; reversed intervals replaced upstream by top + 0.01 m","original_boreholes":len(original),"reconstructed_rows":len(expected),"cached_rows":len(cached),"matching_rows":int(counts["both"]),"max_absolute_difference_m":maxerrors,"retained_labelled_intervals":len(retained),"upstream_repairs_retained":int(retained.upstream_reversed_interval_repair.sum()),"repair_labels":retained.loc[retained.upstream_reversed_interval_repair,"formation"].value_counts().to_dict(),"source_file_sha256":hashlib.sha256((S/"doggr_source.dat").read_bytes()).hexdigest(),"cached_file_sha256":hashlib.sha256((ROOT/"data/kim_ready.csv").read_bytes()).hexdigest(),"interpretation":"Units are established by source header and complete numerical reconstruction, not inferred from magnitude. Geological contact fidelity and the upstream positional label assignment remain separate questions."}
report.update(max_absolute_difference_retained_m=retained_errors,excluded_topo_differences=excluded_differences,all_583_used_intervals_reconstructed=True,repaired_intervals=retained.loc[retained.upstream_reversed_interval_repair,["wellid","name_cached","formation","top_cached","base_cached"]].to_dict("records"))
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
