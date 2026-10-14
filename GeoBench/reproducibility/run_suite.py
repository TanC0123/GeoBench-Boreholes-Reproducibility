"""Portable complete run entry point; preserves one log per evaluation job."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--resolution", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.resolution:
        jobs = [(name, "group", step) for name, steps in [("Zhengzhou", (0.25, 1.)), ("Kimberlina", (2.5, 10.)), ("SyntheticControl", (0.25, 1.))] for step in steps]
    else:
        jobs = [(name, cv, None) for cv in ("group", "spatial") for name in ("Zhengzhou", "Kimberlina", "SyntheticControl")]
    def execute(job):
        name, cv, step = job
        label = f"{name}_{cv}" if step is None else f"{name}_step{step:g}"
        command = [sys.executable, "-m", "GeoBench.reproducibility.run", "--root", str(args.root), "--dataset", name, "--cv", cv, "--output", str(args.output / label)]
        if step is not None:
            command += ["--tracks", "2", "--step", str(step)]
        environment = os.environ.copy()
        environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with (args.output / f"{label}.log").open("w") as log:
            status = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"{label}: exit={status}", flush=True)
        return status
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        statuses = [f.result() for f in as_completed([pool.submit(execute, job) for job in jobs])]
    raise SystemExit(int(any(statuses)))

if __name__ == "__main__":
    main()
