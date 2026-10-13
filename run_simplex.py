"""Train every model and make every figure in the README.

    python run_simplex.py                        # seeds 0, 1 and 2 (default)
    python run_simplex.py --seeds 0              # one seed
    python run_simplex.py --steps 300            # quick check that everything runs (blurry images)

For each seed it trains 17 small models into runs/seed<s>/:
  * one Mess3 process at residual widths 64 (the paper's size), 8, 4, 3 and 2;
  * two Mess3 processes, sparse and dense, at widths 8, 6, 5, 4, 3 and 2.
Then it writes:
  results/seed<s>/simplex_single.png/.json   the Shai et al. image, shuffle control, smaller widths
  results/seed<s>/simplex_pair.png/.json     two processes squeezed together
  results/seed<s>/caps/*.json                belief planes of the two-process models (capacity.py)
  results/simplex_single.png, simplex_pair.png   copies of the first seed's figures
  results/packing.png/.csv                   learned planes vs optimal packing, all seeds
  results/metrics.csv                        every number in the README: mean, min, max over seeds

Jobs run in parallel, one per CPU core. Finished models are skipped, so it can be stopped and rerun.
"""

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SINGLE_WIDTHS = [64, 8, 4, 3, 2]
PAIR_WIDTHS = [8, 6, 5, 4, 3, 2]


def run(cmd, log_path):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    with open(log_path, "w") as log:
        return subprocess.run([sys.executable] + cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, env=env).returncode


def p(*parts):
    return os.path.join(HERE, *parts)


def train_jobs(seed, steps):
    out = os.path.join("runs", f"seed{seed}")
    jobs = []
    for d in SINGLE_WIDTHS:
        f = os.path.join(out, f"single_d{d}")
        jobs.append((["train.py", "--process", "mess3", "--d_model", str(d), "--steps", str(steps or 20000),
                      "--seed", str(seed), "--log_every", "5000", "--out", f + ".pt"], f, seed))
    for regime in ["mixture", "parallel"]:
        for d in PAIR_WIDTHS:
            f = os.path.join(out, f"{regime}_K2_d{d}")
            jobs.append((["train.py", "--regime", regime, "--K", "2", "--d_model", str(d), "--steps",
                          str(steps or 15000), "--seed", str(seed), "--log_every", "5000", "--out", f + ".pt"], f, seed))
    return jobs


def train(job):
    cmd, f, seed = job
    name = f"seed {seed} {os.path.basename(f)}"
    if os.path.exists(p(f + ".done")):
        return f"skipped {name} (already trained)"
    if run(cmd, p(f + ".log")) != 0:
        return f"FAILED {name}: see {f}.log"
    open(p(f + ".done"), "w").close()
    if "--regime" in cmd:
        caps = os.path.join("results", f"seed{seed}", "caps")
        os.makedirs(p(caps), exist_ok=True)
        cap = os.path.join(caps, os.path.basename(f) + "_cap")
        if run(["capacity.py", "--ckpt", f + ".pt", "--out", cap], p(f + "_cap.log")) != 0:
            return f"FAILED analysis of {name}: see {f}_cap.log"
    return f"trained {name}"


def figures(seed):
    runs = os.path.join("runs", f"seed{seed}")
    res = os.path.join("results", f"seed{seed}")
    os.makedirs(p(res), exist_ok=True)
    # Two-process models trained before this script existed may lack their analysis file.
    for f in glob.glob(p(runs, "*_K2_d*.pt")):
        cap = p(res, "caps", os.path.basename(f)[:-3] + "_cap.json")
        if not os.path.exists(cap):
            os.makedirs(os.path.dirname(cap), exist_ok=True)
            run(["capacity.py", "--ckpt", os.path.relpath(f, HERE), "--out", cap[:-5]], f[:-3] + "_cap.log")
    steps = [
        (["simplex_figures.py", "single", "--seed", str(seed), "--ckpts",
          *[os.path.join(runs, f"single_d{d}.pt") for d in SINGLE_WIDTHS], "--out", os.path.join(res, "simplex_single")],
         "single"),
        (["simplex_figures.py", "pair", "--seed", str(seed),
          "--mixture", *[os.path.join(runs, f"mixture_K2_d{d}.pt") for d in PAIR_WIDTHS],
          "--parallel", *[os.path.join(runs, f"parallel_K2_d{d}.pt") for d in PAIR_WIDTHS],
          "--out", os.path.join(res, "simplex_pair")], "pair"),
    ]
    out = []
    for cmd, name in steps:
        log = p(runs, f"figure_{name}.log")
        ok = run(cmd, log) == 0
        out.append(f"seed {seed}: {'made' if ok else 'FAILED'} {name} figure"
                   + ("" if ok else f" (see {os.path.relpath(log, HERE)})"))
    return out


def summarise(seeds):
    """Collect every per-seed number into results/metrics.csv (mean, min, max over seeds)."""
    vals = defaultdict(list)
    for s in seeds:
        res = p("results", f"seed{s}")
        if os.path.exists(os.path.join(res, "simplex_single.json")):
            for r in json.load(open(os.path.join(res, "simplex_single.json"))):
                for k in ["r2", "beyond", "loss_above_optimum", "shuffled_r2"]:
                    if k in r:
                        vals[("single", "-", "-", r["d_model"], k)].append(r[k])
        if os.path.exists(os.path.join(res, "simplex_pair.json")):
            for r in json.load(open(os.path.join(res, "simplex_pair.json"))):
                for k in ["r2", "beyond"]:
                    vals[("pair", r["regime"], r["process"], r["d_model"], k)].append(r[k])
        for c in glob.glob(os.path.join(res, "caps", "*_cap.json")):
            cap = json.load(open(c))
            vals[("pair", cap["regime"], "both", cap["d_model"], "loss_above_optimum")].append(cap["loss_ratio"] - 1)
    rows = []
    for (exp, regime, proc, d, metric), v in sorted(vals.items(), key=lambda kv: (kv[0][0], kv[0][1], str(kv[0][2]), -kv[0][3], kv[0][4])):
        rows.append({"experiment": exp, "regime": regime, "process": proc, "d_model": d, "metric": metric,
                     "mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)), "n_seeds": len(v)})
    with open(p("results", "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def get(exp, regime, proc, d, metric):
        v = vals.get((exp, regime, proc, d, metric))
        return f"{np.mean(v):.2f} [{np.min(v):.2f}, {np.max(v):.2f}]" if v else "-"

    print(f"\nSummary over seeds {seeds}: mean [min, max]")
    print("One process:   width   R²                    beyond last token     loss above optimum (%)")
    for d in SINGLE_WIDTHS:
        v = vals.get(("single", "-", "-", d, "loss_above_optimum"), [np.nan])
        print(f"               {d:>5}   {get('single', '-', '-', d, 'r2'):<20}  {get('single', '-', '-', d, 'beyond'):<20}  "
              f"{100 * np.mean(v):.2f} [{100 * np.min(v):.2f}, {100 * np.max(v):.2f}]")
    print(f"  shuffle control R² (width 64): {get('single', '-', '-', 64, 'shuffled_r2')}")
    print("Two processes: beyond last token for A / B")
    for regime in ["mixture", "parallel"]:
        for d in PAIR_WIDTHS:
            print(f"  {regime:>8} width {d}:  A {get('pair', regime, 'A', d, 'beyond'):<20}  B {get('pair', regime, 'B', d, 'beyond')}")
    print("saved results/metrics.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=None, help="override training steps for every model")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    jobs = []
    for s in args.seeds:
        os.makedirs(p("runs", f"seed{s}"), exist_ok=True)
        jobs += train_jobs(s, args.steps)
    jobs.sort(key=lambda j: "_d64" not in j[1])  # start the slowest models first
    print(f"{len(jobs)} models over seeds {args.seeds}, {args.jobs} at a time", flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for fut in as_completed([pool.submit(train, j) for j in jobs]):
            print(fut.result(), flush=True)

    with ThreadPoolExecutor(max_workers=min(args.jobs, 2 * len(args.seeds))) as pool:
        for fut in as_completed([pool.submit(figures, s) for s in args.seeds]):
            print("\n".join(fut.result()), flush=True)

    first = args.seeds[0]
    for name in ["simplex_single.png", "simplex_pair.png"]:
        src = p("results", f"seed{first}", name)
        if os.path.exists(src):
            shutil.copy(src, p("results", name))
    caps = sorted(glob.glob(p("results", "seed*", "caps", "*_cap.json")))
    code = run(["packing.py", "--caps", *[os.path.relpath(c, HERE) for c in caps], "--out", os.path.join("results", "packing")],
               p("runs", "figure_packing.log"))
    print(("made" if code == 0 else "FAILED") + " packing figure" + ("" if code == 0 else " (see runs/figure_packing.log)"))
    summarise(args.seeds)


if __name__ == "__main__":
    main()
