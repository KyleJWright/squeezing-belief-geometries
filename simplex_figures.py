"""Belief-simplex images: the Shai et al. figure, and what happens to it when the model is squeezed.

    # One Mess3 process at several residual widths (reproduction, control, too-small models)
    python simplex_figures.py single --ckpts runs/seed0/single_d64.pt runs/seed0/single_d8.pt ... \
        --out results/seed0/simplex_single

    # Two processes at several widths, sparse (mixture) and dense (parallel) side by side
    python simplex_figures.py pair --mixture runs/seed0/mixture_K2_d*.pt --parallel runs/seed0/parallel_K2_d*.pt \
        --out results/seed0/simplex_pair

How each panel is made (the procedure of Shai et al., Fig. 5):
  1. Compute the exact belief over the 3 hidden states after every prefix (single mode: every
     distinct prefix of all 3^10 sequences, as in the paper; pair mode: sampled sequences).
  2. Take the residual stream at the final layer.
  3. Fit an affine map from activations to beliefs on half of the prefixes (ridge, see fitting.py).
  4. Plot the OTHER half's predicted beliefs in the probability triangle, coloured by true belief.

Shuffle control (single mode, widest model). As in the paper, the same fit is repeated after
randomly re-pairing activations with beliefs. If the fractal were something a linear map from a
high-dimensional space could draw out of any activations, it would reappear; instead the points
collapse to the centre of the triangle.

Each panel title reports:
  R²                 held-out fraction of belief variance the affine map recovers.
  beyond last token  about 70% of a Mess3 belief is fixed by the last token alone (the three large
                     lobes of the fractal); the fine structure is memory of earlier tokens. This is
                     the share of what a last-token lookup cannot explain that the model captures:
                     (R² - R²_last) / (1 - R²_last). Near 1: the full fractal. 0 or below (shown as
                     0): no memory beyond the last token, and the image collapses to clusters.
The .json output also records the loss above the Bayes optimum and canonical correlations between
the activations and the belief's two simplex coordinates.
"""

import argparse
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from composite import build
from fitting import choose_ridge, fit_affine
from model import load_checkpoint
from processes import get_process

CORNERS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
E = np.linalg.qr(np.array([[1.0, -1.0, 0.0], [1.0, 1.0, -2.0]]).T)[0]  # tangent basis of the simplex
INK, MUTED, RULE = "#1f1f1e", "#5f5e5a", "#c3c2b7"


# ------------------------------------------------------------------------ data

def width_of(path):
    m = re.search(r"_d(\d+)\.pt$", path)
    return int(m.group(1)) if m else -1


def final_residual(model, tokens, batch=4096):
    out = []
    with torch.no_grad():
        for i in range(0, len(tokens), batch):
            _, rs = model(torch.from_numpy(tokens[i:i + batch]), return_resid=True)
            out.append(rs[-1].numpy().astype(np.float32))
    return np.concatenate(out)


def enumerate_prefixes(hmm, model):
    """Every distinct prefix of every possible sequence: activations A (N, d), beliefs Y (N, 3) and
    the prefix's last token (N,), one row per distinct prefix of length 1..n_ctx."""
    L = model.cfg.n_ctx
    seqs, B, _ = hmm.possible_sequences(L)
    R = final_residual(model, seqs)
    A, Y, last = [], [], []
    for t in range(L):
        _, idx = np.unique(seqs[:, :t + 1], axis=0, return_index=True)
        A.append(R[idx, t])
        Y.append(B[idx, t])
        last.append(seqs[idx, t])
    return np.concatenate(A), np.concatenate(Y), np.concatenate(last)


def loss_above_optimum(model, hmm, rng, n=20000):
    """Model cross-entropy relative to the Bayes-optimal predictor, scored on the same sampled
    sequences (pairing the two cancels most sampling noise). Returns loss/optimal - 1."""
    toks = hmm.sample(n, model.cfg.n_ctx + 1, rng)
    with torch.no_grad():
        logits = model(torch.from_numpy(toks[:, :-1]))
        ce = float(F.cross_entropy(logits.reshape(-1, hmm.X), torch.from_numpy(toks[:, 1:]).reshape(-1)))
    P = hmm.next_token_probs(hmm.beliefs_along(toks[:, :-1])[0])
    p = np.take_along_axis(P, toks[:, 1:, None], axis=-1)[..., 0]
    return ce / float(-np.log(np.clip(p, 1e-300, 1)).mean()) - 1


# ------------------------------------------------------------------------ fitting

def canonical_correlations(A, U, reg=1e-6):
    """Canonical correlations between activations A (N, d) and belief coordinates U (N, 2)."""
    A = A - A.mean(0)
    U = U - U.mean(0)
    Caa, Cuu, Cau = A.T @ A / len(A), U.T @ U / len(U), A.T @ U / len(A)
    ia = np.linalg.inv(np.linalg.cholesky(Caa + reg * np.trace(Caa) / len(Caa) * np.eye(len(Caa))))
    iu = np.linalg.inv(np.linalg.cholesky(Cuu + 1e-12 * np.eye(len(Cuu))))
    return np.clip(np.linalg.svd(ia @ Cau @ iu.T, compute_uv=False), 0, 1)


def decode(A, Y, half, rng, last, shuffle=False):
    """Fit belief ~ affine(A) on the rows in `half`, predict the rest.

    shuffle=True re-pairs the fitting half's activations and beliefs at random first (the control).
    Returns held-out predictions, held-out true beliefs, and metrics.
    """
    Atr, Ytr = A[half], Y[half]
    if shuffle:
        Ytr = Ytr[rng.permutation(len(Ytr))]
    ridge = choose_ridge(Atr, Ytr, rng)
    W, c = fit_affine(Atr, Ytr, ridge)
    te = ~half
    pred = A[te] @ W.T + c
    ss_tot = ((Y[te] - Y[te].mean(0)) ** 2).sum()
    r2 = float(1 - ((Y[te] - pred) ** 2).sum() / ss_tot)
    # Baseline: predict the belief from the last token alone (its mean belief on the fitting half).
    means = {v: Y[half & (last == v)].mean(0) for v in np.unique(last[half])}
    base = np.array([means.get(v, Y[half].mean(0)) for v in last[te]])
    r2_tok = float(1 - ((Y[te] - base) ** 2).sum() / ss_tot)
    m = {"r2": r2, "r2_last_token": r2_tok, "beyond": float((r2 - r2_tok) / max(1 - r2_tok, 1e-9))}
    if not shuffle:
        m["rho"] = canonical_correlations(A, (Y - 1 / 3) @ E).tolist()
    return pred, Y[te], m


# ------------------------------------------------------------------------ drawing

def draw(ax, pred, true, title, rng, n_max=30000, s=0.5):
    if len(pred) > n_max:
        keep = rng.choice(len(pred), n_max, replace=False)
        pred, true = pred[keep], true[keep]
    tri = np.vstack([CORNERS, CORNERS[:1]])
    ax.plot(tri[:, 0], tri[:, 1], color=RULE, linewidth=0.8, zorder=0)
    xy = pred @ CORNERS
    ax.scatter(xy[:, 0], xy[:, 1], c=np.clip(true, 0, 1), s=s, linewidths=0, rasterized=True)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.22, 1.08)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9.5, color=INK)


def metric_title(width, m):
    return f"d_model = {width}\nR² {m['r2']:.2f} · beyond last token {max(m['beyond'], 0):.2f}"


# ------------------------------------------------------------------------ one process

def single(args, rng):
    ckpts = sorted(args.ckpts, key=width_of, reverse=True)
    fig, axes = plt.subplots(2, 4, figsize=(14.4, 7.6))
    axes = axes.ravel()
    records = []
    slots = [1] + list(range(3, 3 + len(ckpts) - 1))  # widest model next to the ground truth, control after it
    for i, path in enumerate(ckpts):
        model, ckpt = load_checkpoint(path)
        hmm = get_process(ckpt["process"])
        A, Y, last = enumerate_prefixes(hmm, model)
        half = rng.random(len(Y)) < 0.5
        pred, true, m = decode(A, Y, half, rng, last)
        m["loss_above_optimum"] = loss_above_optimum(model, hmm, rng)
        if i == 0:
            draw(axes[0], Y, Y, "Ground truth\n(exact beliefs, every distinct prefix)", rng)
            spred, strue, sm = decode(A, Y, half, rng, last, shuffle=True)
            draw(axes[2], spred, strue, f"Control: beliefs shuffled before fitting\n"
                                        f"d_model = {model.cfg.d_model}, R² {sm['r2']:.2f}", rng)
            m["shuffled_r2"] = sm["r2"]
        draw(axes[slots[i]], pred, true, metric_title(model.cfg.d_model, m), rng)
        records.append({"ckpt": path, "d_model": model.cfg.d_model, **m})
        print(f"d_model {model.cfg.d_model:>3}: R² {m['r2']:.3f}, last token alone {m['r2_last_token']:.3f}, "
              f"beyond last token {m['beyond']:.3f}, loss above optimum {100 * m['loss_above_optimum']:.2f}%"
              + (f", shuffled R² {m['shuffled_r2']:.3f}" if "shuffled_r2" in m else ""))
    for ax in axes[3 + len(ckpts) - 1:]:
        ax.axis("off")
    axes[-1].text(0.02, 0.5, "Each panel: residual stream decoded into the\nprobability triangle, coloured by the true\n"
                  "belief. Held-out prefixes only.\n\nR²: belief variance recovered.\n"
                  "Beyond last token: share of the fine\nstructure (memory of earlier tokens) that\nthe model captures.",
                  transform=axes[-1].transAxes, fontsize=9, color=MUTED, va="center")
    fig.suptitle("One Mess3 process: the belief simplex in the final residual stream, as the residual width shrinks",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out + ".png", dpi=170)
    with open(args.out + ".json", "w") as f:
        json.dump(records, f, indent=1)
    print(f"saved {args.out}.png and {args.out}.json")


# ------------------------------------------------------------------------ two processes

def panels_for(path, args, rng):
    model, ckpt = load_checkpoint(path)
    comp = build(ckpt["extra"]["spec"])
    tokens, labels = comp.sample(args.n_seq, model.cfg.n_ctx, rng)
    beliefs = comp.factor_beliefs(tokens, labels)
    R = final_residual(model, tokens)
    half_seq = rng.random(len(tokens)) < 0.5
    out = []
    for k, (B, mask) in enumerate(beliefs):
        A = R[mask].reshape(-1, R.shape[-1])
        Y = B[mask].reshape(-1, 3)
        last = comp.sub_tokens(tokens[mask], k).reshape(-1)
        half = np.repeat(half_seq[mask], model.cfg.n_ctx)
        out.append(decode(A, Y, half, rng, last))
    return model.cfg.d_model, comp, out


def pair(args, rng):
    runs = {"mixture": sorted(args.mixture, key=width_of, reverse=True),
            "parallel": sorted(args.parallel, key=width_of, reverse=True)}
    for ps in runs.values():
        if ps and build(load_checkpoint(ps[0])[1]["extra"]["spec"]).K != 2:
            raise SystemExit("pair mode expects models trained with --K 2")
    widths = sorted({width_of(p) for ps in runs.values() for p in ps}, reverse=True)
    nrow = len(widths) + 1
    fig, axes = plt.subplots(nrow, 5, figsize=(13, 2.75 * nrow), gridspec_kw={"width_ratios": [1, 1, 0.12, 1, 1]})
    for ax in axes[:, 2]:
        ax.axis("off")
    records = []
    comp = None
    for c0, regime in [(0, "mixture"), (3, "parallel")]:
        by_width = {width_of(p): p for p in runs[regime]}
        for r, w in enumerate(widths, start=1):
            if w not in by_width:
                axes[r, c0].axis("off")
                axes[r, c0 + 1].axis("off")
                continue
            d, comp, res = panels_for(by_width[w], args, rng)
            for k, (pred, true, m) in enumerate(res):
                draw(axes[r, c0 + k], pred, true, metric_title(d, m), rng)
                records.append({"regime": regime, "d_model": d, "process": "AB"[k], **m})
                print(f"{regime:>8} d_model {d:>2} process {'AB'[k]}: R² {m['r2']:.3f}, "
                      f"last token alone {m['r2_last_token']:.3f}, beyond {m['beyond']:.3f}")
    for c0 in (0, 3):  # ground-truth row: exact beliefs of each process
        for k, hmm in enumerate(comp.components):
            Y = hmm.beliefs_along(hmm.sample(args.n_seq, 10, rng))[0].reshape(-1, 3)
            x, a = comp.spec["params"][k]
            draw(axes[0, c0 + k], Y, Y, f"Ground truth, process {'AB'[k]}\n(x = {x}, α = {a})", rng)
    fig.text(0.28, 0.995, "Sparse: one process per sequence", ha="center", va="top", fontsize=12, color=INK)
    fig.text(0.75, 0.995, "Dense: both processes every token", ha="center", va="top", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(args.out + ".png", dpi=150)
    with open(args.out + ".json", "w") as f:
        json.dump(records, f, indent=1)
    print(f"saved {args.out}.png and {args.out}.json")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("single")
    s.add_argument("--ckpts", nargs="+", required=True)
    p = sub.add_parser("pair")
    p.add_argument("--mixture", nargs="+", required=True)
    p.add_argument("--parallel", nargs="+", required=True)
    for q in (s, p):
        q.add_argument("--out", required=True)
        q.add_argument("--n_seq", type=int, default=4000, help="pair mode: sampled sequences per model")
        q.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    single(args, rng) if args.mode == "single" else pair(args, rng)


if __name__ == "__main__":
    main()
