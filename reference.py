"""Theoretical reference models for the sparse regime with shared tokens, compared with the trained models.

    python reference.py      # a few minutes; uses runs/seed*/shared_K2_d*.pt and, if present, runs/develop/

With one shared set of tokens, the two processes together are a single hidden Markov model with 6 hidden
states, (process, state). This script computes two reference points for what a model of that data could hold.

1. The exact joint belief. The posterior over the 6 states,
       J = (pi * eta_A, (1 - pi) * eta_B),   pi = P(A | prefix),
   lives in a 5-dimensional simplex (the join of the two triangles). It is the best possible belief state:
   every exact prediction follows from it. The question is how much of it a trained model carries.
   A ridge probe from the residual stream to J gives a held-out R^2. To read that number, the same probe is
   fitted from idealised representations that hold only part of J:
     prediction        the 3 next-token probabilities (2 free numbers): all the output needs.
     prediction + pi   the prediction plus the process posterior (3 free numbers): what section 5 suggests
                       the width-8 models hold.
     averaged belief   the belief of the best single 3-state model (below).
   If the model held the full joint belief, its R^2 would approach 1; if it held only the prediction and pi,
   it would sit near the "prediction + pi" reference.

2. The best single averaged model. A single 3-state hidden Markov model (same edge-emitting form as Mess3)
   is fitted to the mixed data by maximum likelihood (Baum-Welch, several random starts). Its loss above the
   optimum is what a model loses by treating the two processes as one. If training first learns one
   averaged process and only later splits it, early training snapshots should sit near this level before
   going below it. The last-token predictor (bigram) and the uniform predictor are shown for scale.

Output: results/reference/reference.png, reference.json, and a printed summary.
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from capacity import collect_resid, r2_score
from composite import build, default_spec
from fitting import choose_ridge, fit_affine
from model import load_checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "reference")
L = 10  # context length of the trained models
INK, MUTED, RULE = "#1f1f1e", "#5f5e5a", "#c3c2b7"
WIDTH_COLOURS = {8: "#1baf7a", 6: "#2a78d6", 5: "#8a56c9", 4: "#eb6834", 3: "#b8860b"}


# ------------------------------------------------------------------------------ exact quantities

def joint_quantities(comp, tokens):
    """For prefixes tokens[:, :t+1]: joint belief J (N, L, 6), process posterior pi (N, L), prediction (N, L, 3)."""
    post = comp.process_posterior(tokens)  # (N, L, 2)
    etas = [c.beliefs_along(tokens)[0] for c in comp.components]  # each (N, L, 3)
    J = np.concatenate([post[..., k:k + 1] * etas[k] for k in range(comp.K)], axis=-1)
    pred = comp.next_token_dist(tokens)
    return J, post[..., 0], pred


# ------------------------------------------------------------------------------ averaged 3-state model

def forward_beliefs(T, init, seqs):
    """Edge-emitting HMM T[x][i, j]. Beliefs after each prefix (N, L, S), predictions (N, L, X)."""
    N, Lx = seqs.shape
    eta = np.tile(init, (N, 1))
    B = np.zeros((N, Lx, T.shape[1]))
    for t in range(Lx):
        v = np.einsum("ns,nsr->nr", eta, T[seqs[:, t]])
        eta = v / v.sum(1, keepdims=True)
        B[:, t] = eta
    P = np.einsum("nls,xsr->nlx", B, T)
    return B, P


def baum_welch(seqs, S=3, X=3, iters=150, restarts=8, seed=0):
    """Maximum-likelihood edge-emitting HMM with S states for the sequences (N, L). Returns (T, init, loglik)."""
    rng = np.random.default_rng(seed)
    N, Lx = seqs.shape
    best = None
    for _ in range(restarts):
        T = rng.random((X, S, S)) + 0.5
        T /= T.sum((0, 2), keepdims=True)
        init = np.full(S, 1.0 / S)
        for _ in range(iters):
            # forward (scaled)
            alpha = np.zeros((N, Lx + 1, S))
            alpha[:, 0] = init
            c = np.zeros((N, Lx))
            for t in range(Lx):
                a = np.einsum("ns,nsr->nr", alpha[:, t], T[seqs[:, t]])
                c[:, t] = a.sum(1)
                alpha[:, t + 1] = a / c[:, t:t + 1]
            # backward (scaled with the same constants)
            beta = np.ones((N, Lx + 1, S))
            for t in range(Lx - 1, -1, -1):
                beta[:, t] = np.einsum("nsr,nr->ns", T[seqs[:, t]], beta[:, t + 1]) / c[:, t:t + 1]
            # expected transition counts per token
            counts = np.zeros((X, S, S))
            for t in range(Lx):
                xi = alpha[:, t, :, None] * T[seqs[:, t]] * beta[:, t + 1, None, :] / c[:, t, None, None]
                np.add.at(counts, seqs[:, t], xi)
            gamma0 = alpha[:, 0] * beta[:, 0]
            init = gamma0.sum(0) / gamma0.sum()
            T = counts / np.maximum(counts.sum((0, 2), keepdims=True), 1e-300)
        ll = float(np.log(c).sum())
        if best is None or ll > best[2]:
            best = (T, init, ll)
    return best


def cross_entropy(P, tokens):
    """Mean -log P of the next token; P[:, t] predicts tokens[:, t + 1]."""
    p = np.take_along_axis(P[:, :-1], tokens[:, 1:, None], -1)[..., 0]
    return float(-np.log(np.clip(p, 1e-300, 1)).mean())


# ------------------------------------------------------------------------------ probes

def probe_r2(F, Y, half, rng):
    """Held-out R^2 of a ridge probe from features F (N, L, f) to targets Y (N, L, y)."""
    Ftr, Ytr = F[half].reshape(-1, F.shape[-1]), Y[half].reshape(-1, Y.shape[-1])
    W, c = fit_affine(Ftr, Ytr, choose_ridge(Ftr, Ytr, rng))
    Fte, Yte = F[~half].reshape(-1, F.shape[-1]), Y[~half].reshape(-1, Y.shape[-1])
    return r2_score(Yte, Fte @ W.T + c)


def model_probes(path, tokens, J, avgB, half, seed):
    model, ckpt = load_checkpoint(path)
    rng = np.random.default_rng(seed)
    resid = collect_resid(model, tokens)
    joint = [probe_r2(A, J, half, rng) for A in resid]
    avg = [probe_r2(A, avgB, half, rng) for A in resid]
    m = re.search(r"seed(\d+)", path)
    return {"path": os.path.relpath(path, HERE), "d_model": model.cfg.d_model, "step": ckpt["extra"].get("step"),
            "seed": int(m.group(1)) if m else None, "joint_r2": joint, "averaged_r2": avg}


# ------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_fit", type=int, default=20000, help="sequences for fitting the averaged model")
    ap.add_argument("--n_eval", type=int, default=4000, help="sequences for losses and probes")
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)
    comp = build(default_spec("mixture", 2, "shared"))

    # Reference losses on the same evaluation sequences (length L + 1: L prefixes, each predicting the next token).
    rng = np.random.default_rng(1)
    fit_seqs, _ = comp.sample(args.n_fit, L + 1, rng)
    T, init, _ = baum_welch(fit_seqs, restarts=args.restarts)
    ev, _ = comp.sample(args.n_eval, L + 1, np.random.default_rng(2))
    opt = cross_entropy(comp.next_token_dist(ev), ev)
    avg_B, avg_P = forward_beliefs(T, init, ev)
    prev, cur = ev[:, :-1].ravel(), ev[:, 1:].ravel()
    C = np.zeros((3, 3))
    np.add.at(C, (fit_seqs[:, :-1].ravel(), fit_seqs[:, 1:].ravel()), 1)
    C /= C.sum(1, keepdims=True)
    bigram = float(-np.log(C[prev, cur]).mean())
    losses = {"optimal": opt, "averaged_3_state": cross_entropy(avg_P, ev), "last_token": bigram,
              "uniform": float(np.log(3))}
    above = {k: v / opt - 1 for k, v in losses.items()}
    print("Reference predictors on shared-token sequences (loss above the Bayes optimum):")
    for k, v in above.items():
        print(f"  {k:>17}: {100 * v:6.2f}%   ({losses[k]:.4f} nats/token)")

    # Exact quantities on the probe sequences (inputs = first L tokens).
    inputs = ev[:, :-1]
    J, pi, pred = joint_quantities(comp, inputs)
    avgB = avg_B[:, :-1]
    half = np.random.default_rng(3).random(len(inputs)) < 0.5
    prng = np.random.default_rng(4)
    refs = {"prediction": probe_r2(pred, J, half, prng),
            "prediction + pi": probe_r2(np.concatenate([pred, pi[..., None]], -1), J, half, prng),
            "averaged belief": probe_r2(avgB, J, half, prng)}
    print("\nHow much of the joint belief each idealised representation carries (R^2 of a linear probe):")
    for k, v in refs.items():
        print(f"  {k:>16}: {v:.3f}")

    final = sorted(p for p in glob.glob(os.path.join(HERE, "runs", "seed*", "shared_K2_d*.pt")) if "_step" not in p)
    snaps = sorted(glob.glob(os.path.join(HERE, "runs", "develop", "shared_K2_d*_seed*.pt")))
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        fin = list(pool.map(lambda p: model_probes(p, inputs, J, avgB, half, 5), final))
        dev = list(pool.map(lambda p: model_probes(p, inputs, J, avgB, half, 5), snaps))
    by_w = defaultdict(list)
    for r in fin:
        by_w[r["d_model"]].append(r)
    print("\nTrained shared-token models: R^2 of a probe for the joint belief (final layer / best layer), mean over seeds")
    for w in sorted(by_w, reverse=True):
        rs = by_w[w]
        print(f"  width {w}: {np.mean([r['joint_r2'][-1] for r in rs]):.3f} / {np.mean([max(r['joint_r2']) for r in rs]):.3f}"
              f"   (averaged-model belief: {np.mean([r['averaged_r2'][-1] for r in rs]):.3f})")

    # Development: loss and probes over training, from disentangle.py's results if present.
    dev_json = os.path.join(HERE, "results", "disentangle", "develop.json")
    dev_loss = defaultdict(lambda: defaultdict(list))
    if os.path.exists(dev_json):
        for r in json.load(open(dev_json)):
            dev_loss[r["width"]][r["step"]].append(r["loss_above_optimum"])
    dev_by = defaultdict(lambda: defaultdict(list))
    for r in dev:
        w = int(re.search(r"_d(\d+)_", os.path.basename(r["path"])).group(1))
        dev_by[w][r["step"]].append(r)
    if dev_loss:
        print("\nTraining snapshots: loss above optimum against the averaged model "
              f"({100 * above['averaged_3_state']:.2f}%) and the last-token predictor ({100 * above['last_token']:.2f}%)")
        for w in sorted(dev_loss, reverse=True):
            steps = sorted(dev_loss[w])
            cross = next((s for s in steps if np.mean(dev_loss[w][s]) < above["averaged_3_state"]), None)
            print(f"  width {w}: first step below the averaged model: {cross}; at step {steps[0]}: "
                  f"{100 * np.mean(dev_loss[w][steps[0]]):.2f}%")

    # Figure
    fig, axes = plt.subplots(1, 3 if dev_loss else 1, figsize=(17 if dev_loss else 6.5, 4.8), squeeze=False)
    ax = axes[0, 0]
    widths = sorted(by_w, reverse=True)
    ax.bar(range(len(widths)), [np.mean([max(r["joint_r2"]) for r in by_w[w]]) for w in widths],
           color=[WIDTH_COLOURS.get(w, MUTED) for w in widths])
    ax.set_xticks(range(len(widths)))
    ax.set_xticklabels([f"width {w}" for w in widths])
    for k, ls in zip(refs, ["--", "-", ":"]):
        ax.axhline(refs[k], color=INK, linestyle=ls, linewidth=1.2, label=f"{k} only: {refs[k]:.2f}")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("R² of a linear probe for the joint belief")
    ax.set_title("How much of the 6-state joint belief the model carries", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    if dev_loss:
        ax = axes[0, 1]
        for w in sorted(dev_loss, reverse=True):
            steps = sorted(dev_loss[w])
            ax.plot(steps, [100 * np.mean(dev_loss[w][s]) for s in steps], marker="o", markersize=3,
                    color=WIDTH_COLOURS.get(w, MUTED), label=f"trained model, width {w}")
        for k, ls in [("averaged_3_state", "-"), ("last_token", "--")]:
            ax.axhline(100 * above[k], color=INK, linestyle=ls, linewidth=1.2,
                       label={"averaged_3_state": "best single 3-state model", "last_token": "last token only"}[k])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("training step")
        ax.set_ylabel("loss above optimum (%)")
        ax.set_title("Training against one averaged process", loc="left", fontsize=10, color=INK)
        ax.legend(frameon=False, fontsize=8)
        ax = axes[0, 2]
        for w in sorted(dev_by, reverse=True):
            steps = sorted(dev_by[w])
            for key, ls, lab in [("joint_r2", "-", "joint belief"), ("averaged_r2", "--", "averaged-model belief")]:
                ax.plot(steps, [np.mean([max(r[key]) for r in dev_by[w][s]]) for s in steps], linestyle=ls,
                        color=WIDTH_COLOURS.get(w, MUTED), label=f"{lab}, width {w}")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("training step")
        ax.set_ylabel("probe R² (best layer)")
        ax.set_title("Which belief the model carries during training", loc="left", fontsize=10, color=INK)
        ax.legend(frameon=False, fontsize=8)
    for a in axes[0]:
        a.grid(alpha=0.3, color=RULE)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "reference.png"), dpi=150)
    with open(os.path.join(OUT, "reference.json"), "w") as f:
        json.dump({"losses": losses, "loss_above_optimum": above, "references": refs,
                   "averaged_model": {"T": T.tolist(), "init": init.tolist()}, "final_models": fin,
                   "snapshots": dev}, f, indent=1)
    print(f"\nsaved {os.path.relpath(os.path.join(OUT, 'reference.png'), HERE)} and reference.json")


if __name__ == "__main__":
    main()
