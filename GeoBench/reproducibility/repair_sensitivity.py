"""Remove documented upstream repair contacts from regression targets/donors."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output",type=Path,default=ROOT/".tmp/repair_sensitivity")
args=parser.parse_args()
OUT=args.output
OUT.mkdir(exist_ok=True)

def run(cv):
    from GeoBench.reproducibility import run as benchmark
    repairs=json.loads((ROOT/"GeoBench/provenance/kimberlina_units_verified.json").read_text())["repaired_intervals"]
    keys={(r["name_cached"],r["formation"]) for r in repairs}
    original_load=benchmark.load_intervals
    original_queries=benchmark.build_queries
    def load(root,name):
        intervals,metadata=original_load(root,name)
        mask=[(r.borehole,r.horizon) in keys for r in intervals.itertuples()]
        intervals.loc[mask,"boundary_observed"]=False
        metadata["regression_sensitivity"]="Documented upstream repaired contacts excluded from boundary donors and regression; interval labels unchanged."
        return intervals,metadata
    def queries(intervals,metadata,step):
        interface,points,metadata=original_queries(intervals,metadata,step)
        interface=interface[interface.boundary_observed].reset_index(drop=True)
        metadata["excluded_upstream_repaired_nonterminal_boundaries"]=metadata["boundary_rows"]-len(interface)
        metadata["boundary_rows"]=len(interface)
        return interface,points,metadata
    benchmark.load_intervals=load
    benchmark.build_queries=queries
    with (OUT/f"{cv}.log").open("w") as log,redirect_stdout(log),redirect_stderr(log):
        result=benchmark.run_dataset(ROOT,OUT/cv,"Kimberlina",cv,tracks=(1,))
    return {"cv":cv,"models_scored":len(result),"status":"complete"}

if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=2) as pool:
        for result in pool.map(run,("group","spatial")):print(json.dumps(result),flush=True)
