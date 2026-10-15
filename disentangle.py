"""Does the model work out which process it is in, and does it learn one model before splitting into two?

Two experiments on the sparse regime with one shared set of tokens (shared_K2_d*), where the model is
never told which process produced a sequence and must infer it from the statistics of the tokens.

    python disentangle.py probe     # uses the models run_simplex.py already trained (a few minutes)
    python disentangle.py train     # trains shared-token models, saving snapshots during training
    python disentangle.py develop   # analyses those snapshots
    python disentangle.py all       # train, then develop, then probe

1. probe. The exact posterior probability that process A is running, pi_A, is computed for every
   prefix (composite.Composite.process_posterior). At every layer a ridge probe is fitted from the
   residual stream to pi_A on half the sequences and scored on the other half:
     r2         held-out R^2 of the probe for pi_A.
     accuracy   how often the probe's guess (pi_A > 1/2) names the true process, per position, against
                the Bayes-optimal accuracy (the exact pi_A's guess). Bayes accuracy is below 1 because
                within 10 tokens the two processes are often hard to tell apart.
   The separate-token models are included as a control: there the first token reveals the process.
   Output: results/disentangle/probe.png and probe.json.

2. develop. Shared-token models are retrained (same seeds and settings as run_simplex.py) with
   snapshots at roughly log-spaced steps, into runs/develop/. For every snapshot and layer:
     loss above optimum, the overlap of the A and B belief planes, the offset between their patches,
     each process's belief R^2, and the R^2 of the pi_A probe.
   If the model first learns one averaged process and later splits it into two, the A and B planes
   should start identical (overlap near 1) and separate, and the pi_A probe should switch on, at some
   point in training, probably together with a drop in loss.
   Output: results/disentangle/develop.png and develop.json.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from capacity import analyse_layer, collect_resid, model_loss, optimal_loss_on, r2_score
from composite import build
from fitting import choose_ridge, fit_affine
from model import load_checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "disentangle")
DEV_RUNS = os.path.join(HERE, "runs", "develop")
DEV_WIDTHS = [8, 4]
SEEDS = [0, 1, 2]
STEPS = 15000
SNAPSHOTS = [25, 50, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 10000, 12500]
INK, MUTED, RULE = "#1f1f1e", "#5f5e5a", "#c3c2b7"
WIDTH_COLOURS = {8: "#1baf7a", 6: "#2a78d6", 5: "#8a56c9", 4: "#eb6834", 3: "#b8860b"}


def width_of(path):
    return int(re.search(r"_d(\d+)", os.path.basename(path)).group(1))


def sample(comp, n, L, seed):
    rng = np.random.default_rng(seed)
    tokens, labels = comp.sample(n, L, rng)
    return tokens, labels, rng


def probe_posterior(A_seq, pi, half, rng):
    """Ridge probe from activations (N, L, d) to pi (N, L). Returns held-out R^2 and predictions (N, L)."""
    N, L, d = A_seq.shape
    Atr, ytr = A_seq[half].reshape(-1, d), pi[half].reshape(-1, 1)
    ridge = choose_ridge(Atr, ytr, rng)
    W, c = fit_affine(Atr, ytr, ridge)
    pred = (A_seq.reshape(-1, d) @ W.T + c).reshape(N, L)
    return r2_score(pi[~half].reshape(-1, 1), pred[~half].reshape(-1, 1)), pred


def analyse_model(path, n_seq=3000, seed=0, full=False):
    """Posterior probe at every layer; with full=True also belief planes and loss (for snapshots)."""
    model, ckpt = load_checkpoint(path)
    comp = build(ckpt["extra"]["spec"])
    L = model.cfg.n_ctx
    tokens, labels, rng = sample(comp, n_seq, L + 1, seed)
    inputs = tokens[:, :-1]
    pi = comp.process_posterior(inputs)[..., 0]  # P(process A | prefix)
    resid = collect_resid(model, inputs)
    half = rng.random(len(inputs)) < 0.5
    isA = labels == 0
    out = {"path": os.path.relpath(path, HERE), "vocab": comp.vocab, "d_model": model.cfg.d_model,
           "step": ckpt["extra"].get("step"),
           "bayes_accuracy": [float(((pi[~half, t] > 0.5) == isA[~half]).mean()) for t in range(L)],
           "mean_abs_confidence": [float(np.abs(pi[:, t] - 0.5).mean() * 2) for t in range(L)],
           "layers": []}
    beliefs = comp.factor_beliefs(inputs, labels) if full else None
    for l, A in enumerate(resid):
        r2, pred = probe_posterior(A, pi, half, rng)
        layer = {"pi_r2": r2,
                 "accuracy": [float(((pred[~half, t] > 0.5) == isA[~half]).mean()) for t in range(L)]}
        if full:
            res = analyse_layer(A, beliefs, comp, rng, half)
            layer.update({"belief_r2": res["r2"], "overlap": float(res["overlap"][0][1]),
                          "offset_sep": float(res["offset_sep"][0][1])})
        out["layers"].append(layer)
    if full:
        out["loss_above_optimum"] = model_loss(model, comp, tokens) / optimal_loss_on(comp, tokens) - 1
    return out


# ------------------------------------------------------------------------------------- probe

def probe(args):
    os.makedirs(OUT, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(HERE, "runs", "seed*", "shared_K2_d*.pt")) +
                   glob.glob(os.path.join(HERE, "runs", "seed*", "mixture_K2_d*.pt")))
    paths = [p for p in paths if "_step" not in p and width_of(p) >= 3]
    if not paths:
        raise SystemExit("no trained models found: run python run_simplex.py first")
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda p: analyse_model(p), paths))
    with open(os.path.join(OUT, "probe.json"), "w") as f:
        json.dump(results, f, indent=1)

    shared = [r for r in results if r["vocab"] == "shared"]
    separate = [r for r in results if r["vocab"] != "shared"]
    widths = sorted({r["d_model"] for r in shared}, reverse=True)
    L = len(shared[0]["bayes_accuracy"])
    pos = np.arange(1, L + 1)

    print("Probe for P(process A | prefix), shared tokens: held-out R^2 by layer, mean over seeds")
    for w in widths:
        rs = [r for r in shared if r["d_model"] == w]
        r2 = np.mean([[l["pi_r2"] for l in r["layers"]] for r in rs], 0)
        print(f"  width {w}: " + "  ".join(f"layer {i}: {v:.2f}" for i, v in enumerate(r2)))
    rs = [r for r in separate]
    if rs:
        r2 = np.mean([r["layers"][-1]["pi_r2"] for r in rs])
        print(f"  control, separate tokens, final layer, all widths: {r2:.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    ax = axes[0]
    bayes = np.mean([r["bayes_accuracy"] for r in shared], 0)
    ax.plot(pos, bayes, color=INK, linewidth=2.2, label="Bayes-optimal (exact posterior)")
    for w in widths:
        rs = [r for r in shared if r["d_model"] == w]
        best = np.mean([max(r["layers"], key=lambda l: l["pi_r2"])["accuracy"] for r in rs], 0)
        ax.plot(pos, best, color=WIDTH_COLOURS.get(w, MUTED), linewidth=1.5, marker="o", markersize=3,
                label=f"probe, width {w} (best layer)")
    ax.axhline(0.5, color=RULE, linewidth=1, linestyle="--")
    ax.set_title("Which process is running? Shared tokens", loc="left", fontsize=11, color=INK)
    ax.set_xlabel("position in sequence (tokens seen)")
    ax.set_ylabel("accuracy of guessing the process")
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[1]
    for w in widths:
        rs = [r for r in shared if r["d_model"] == w]
        r2 = np.array([[l["pi_r2"] for l in r["layers"]] for r in rs])
        layers = np.arange(r2.shape[1])
        ax.plot(layers, r2.mean(0), color=WIDTH_COLOURS.get(w, MUTED), marker="o", linewidth=1.5, label=f"width {w}")
        ax.fill_between(layers, r2.min(0), r2.max(0), color=WIDTH_COLOURS.get(w, MUTED), alpha=0.15, linewidth=0)
    ax.set_ylim(-0.05, 1.02)
    ax.set_xticks(layers)
    ax.set_xticklabels(["embed"] + [f"after {i}" for i in layers[1:]])
    ax.set_title("Probe for P(A | prefix) across layers, shared tokens", loc="left", fontsize=11, color=INK)
    ax.set_xlabel("residual stream (after block)")
    ax.set_ylabel("held-out R² (min–max over seeds shaded)")
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[2]
    conf = np.mean([r["mean_abs_confidence"] for r in shared], 0)
    ax.plot(pos, conf, color=INK, linewidth=2)
    ax.set_ylim(0, 1)
    ax.set_title("How sure the exact posterior is", loc="left", fontsize=11, color=INK)
    ax.set_xlabel("position in sequence (tokens seen)")
    ax.set_ylabel("mean of |2 P(A | prefix) − 1|")
    for a in axes:
        a.grid(alpha=0.3, color=RULE)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "probe.png"), dpi=150)
    print(f"Bayes accuracy by position: " + " ".join(f"{v:.2f}" for v in bayes))
    print(f"saved {os.path.relpath(os.path.join(OUT, 'probe.png'), HERE)} and probe.json")


# ------------------------------------------------------------------------------------- train

def train(args):
    os.makedirs(DEV_RUNS, exist_ok=True)
    jobs = []
    for seed in args.seeds:
        for d in args.widths:
            out = os.path.join("runs", "develop", f"shared_K2_d{d}_seed{seed}.pt")
            if os.path.exists(os.path.join(HERE, out[:-3] + ".done")):
                continue
            jobs.append(([sys.executable, "train.py", "--regime", "mixture", "--vocab", "shared", "--K", "2",
                          "--d_model", str(d), "--steps", str(args.steps), "--seed", str(seed), "--log_every", "2500",
                          "--snapshots", *[str(t) for t in SNAPSHOTS if t < args.steps], "--out", out], out))
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")

    def run(job):
        cmd, out = job
        with open(os.path.join(HERE, out[:-3] + ".log"), "w") as log:
            code = subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        if code == 0:
            open(os.path.join(HERE, out[:-3] + ".done"), "w").close()
        return f"{'trained' if code == 0 else 'FAILED'} {out}"

    print(f"{len(jobs)} models to train, {args.jobs} at a time", flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for fut in as_completed([pool.submit(run, j) for j in jobs]):
            print(fut.result(), flush=True)


# ------------------------------------------------------------------------------------- develop

def develop(args):
    os.makedirs(OUT, exist_ok=True)
    snaps = []
    for seed in args.seeds:
        for d in args.widths:
            stem = os.path.join(DEV_RUNS, f"shared_K2_d{d}_seed{seed}")
            for p in glob.glob(stem + "_step*.pt") + glob.glob(stem + ".pt"):
                snaps.append((seed, d, p))
    if not snaps:
        raise SystemExit("no snapshots found: run python disentangle.py train first")
    print(f"analysing {len(snaps)} snapshots, {args.jobs} at a time", flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(analyse_model, p, 2000, 0, True): (seed, d) for seed, d, p in snaps}
        results = []
        for fut in as_completed(futs):
            r = fut.result()
            r["seed"], r["width"] = futs[fut]
            results.append(r)
    results.sort(key=lambda r: (r["width"], r["seed"], r["step"]))
    with open(os.path.join(OUT, "develop.json"), "w") as f:
        json.dump(results, f, indent=1)

    widths = sorted({r["width"] for r in results}, reverse=True)
    n_layers = len(results[0]["layers"])
    layer_colours = plt.cm.viridis(np.linspace(0, 0.9, n_layers))
    rows = [("loss above optimum (%)", lambda r: [100 * r["loss_above_optimum"]], False),
            ("overlap of A and B belief planes", lambda r: [l["overlap"] for l in r["layers"]], True),
            ("offset between A and B patches (spreads)", lambda r: [l["offset_sep"] for l in r["layers"]], True),
            ("probe R² for P(A | prefix)", lambda r: [l["pi_r2"] for l in r["layers"]], True),
            ("belief R², A and B (final layer)", lambda r: r["layers"][-1]["belief_r2"], False)]
    fig, axes = plt.subplots(len(rows), len(widths), figsize=(6.5 * len(widths), 2.6 * len(rows)),
                             squeeze=False, sharex=True)
    for c, w in enumerate(widths):
        by_seed = defaultdict(list)
        for r in results:
            if r["width"] == w:
                by_seed[r["seed"]].append(r)
        for ri, (title, get, per_layer) in enumerate(rows):
            ax = axes[ri, c]
            for seed, rs in by_seed.items():
                steps = [r["step"] for r in rs]
                vals = np.array([get(r) for r in rs])
                for j in range(vals.shape[1]):
                    if per_layer:
                        colour, label = layer_colours[j], ("embed" if j == 0 else f"after block {j}")
                    elif vals.shape[1] == 2:
                        colour, label = ["#2a78d6", "#eb6834"][j], f"process {'AB'[j]}"
                    else:
                        colour, label = INK, None
                    ax.plot(steps, vals[:, j], color=colour, linewidth=1.2, alpha=0.85,
                            label=label if seed == min(by_seed) else None)
            ax.set_xscale("log")
            if ri == 0:
                ax.set_yscale("log")
                ax.set_title(f"Shared tokens, width {w}", loc="left", fontsize=11, color=INK)
            ax.set_ylabel(title, fontsize=9)
            ax.grid(alpha=0.3, color=RULE)
            ax.spines[["top", "right"]].set_visible(False)
            if c == len(widths) - 1 and ri in (1, 4):
                ax.legend(frameon=False, fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
        axes[-1, c].set_xlabel("training step (one line per seed)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "develop.png"), dpi=150)

    print("\nFinal layer over training, mean over seeds: step, loss above optimum, plane overlap, offset, pi probe R^2")
    for w in widths:
        print(f"  width {w}")
        by_step = defaultdict(list)
        for r in results:
            if r["width"] == w:
                by_step[r["step"]].append(r)
        for step in sorted(by_step):
            rs = by_step[step]
            best_pi = np.mean([max(l["pi_r2"] for l in r["layers"]) for r in rs])
            print(f"    {step:>6}  {100 * np.mean([r['loss_above_optimum'] for r in rs]):7.2f}%"
                  f"  overlap {np.mean([r['layers'][-1]['overlap'] for r in rs]):.2f}"
                  f"  offset {np.mean([r['layers'][-1]['offset_sep'] for r in rs]):.2f}"
                  f"  pi R^2 final {np.mean([r['layers'][-1]['pi_r2'] for r in rs]):.2f}, best layer {best_pi:.2f}")
    print(f"saved {os.path.relpath(os.path.join(OUT, 'develop.png'), HERE)} and develop.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["probe", "train", "develop", "all"])
    ap.add_argument("--widths", type=int, nargs="+", default=DEV_WIDTHS, help="train/develop: residual widths")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--steps", type=int, default=STEPS, help="train: training steps (lower for a quick check)")
    args = ap.parse_args()
    torch.set_num_threads(1)
    if args.what in ("train", "all"):
        train(args)
    if args.what in ("develop", "all"):
        develop(args)
    if args.what in ("probe", "all"):
        probe(args)


if __name__ == "__main__":
    main()
