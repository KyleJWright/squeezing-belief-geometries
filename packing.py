"""Compare the learned arrangement of belief planes with optimal Grassmannian packings.

    python packing.py --caps "results/seed*/caps/*_cap.json" --out results/packing

Each process's belief occupies a 2-dimensional plane in the residual stream: the directions the
activation moves as that belief moves (see capacity.py). How close does a model come to the best
possible arrangement of those planes?

Which space. Every read of the residual stream (each block's input, and the final readout) goes
through a LayerNorm, which first subtracts the mean of the d coordinates. The all-ones direction is
therefore invisible to the model, and the space it can use linearly is the hyperplane orthogonal to
it, of dimension n = d - 1. Planes are measured there (capacity.py projects them first), and the
comparison is with packings in R^n. At width 2 that space is a line, which cannot hold a plane, so
width-2 models are left out.

Packing problem. Place K planes in R^n so the largest pairwise overlap is as small as possible.
With overlap(P, Q) = ||P^T Q||_F^2 / 2 (the mean cos^2 of the two principal angles), this is
maximising the smallest chordal distance, d_c^2 = 2(1 - overlap): the Grassmannian packing problem
(Conway, Hardin & Sloane 1996). Two planes fit with zero overlap once n >= 4.

  rankin    The Rankin / simplex bound: max overlap >= 1 - (n - 2)/n * K/(K - 1). It is attained
            only by special (equi-isoclinic) arrangements, so it can be loose.
  optimal   The best packing found numerically here: gradient descent on a smoothed maximum of
            pairwise overlaps from many random starts.
  learned   The final-layer planes of trained models, among the processes they still represent
            (R^2 >= 0.5), in two metrics on the readable space: Euclidean (filled markers) and the
            metric x^T diag(gamma)^2 y imposed by the output head's LayerNorm scale (hollow markers).

What the comparison can and cannot show. The planes are linear subspaces, but each process's beliefs
form an affine patch: a plane plus an offset. In the sparse regime only one process is active at a
time, so two processes can share directions and be told apart by their offsets; the right panel
shows that separation. Overlap far above the optimum there means the model is not solving a packing
problem, not that it is failing to.
"""

import argparse
import csv
import glob
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

COLOURS = {"mixture": "#2a78d6", "mixture-shared": "#1baf7a", "parallel": "#eb6834"}
NAMES = {"mixture": "sparse, separate tokens", "mixture-shared": "sparse, shared tokens", "parallel": "dense"}
INK, MUTED, RULE = "#1f1f1e", "#5f5e5a", "#c3c2b7"


def rankin(K, n, k=2):
    if K < 2:
        return 0.0
    return max(0.0, 1 - (n - k) / n * K / (K - 1))


def overlaps(Q):
    """Q: (K, n, k) orthonormal frames -> (K, K) overlap matrix."""
    M = torch.einsum("adi,bdj->abij", Q, Q)
    return (M ** 2).sum((-1, -2)) / Q.shape[-1]


def optimal_packing(K, n, k=2, restarts=64, steps=1500, seed=0):
    """Smallest achievable maximum pairwise overlap of K k-planes in R^n (numerical), for n >= k."""
    if k * K <= n:
        return 0.0
    if n == k:
        return 1.0
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(restarts, K, n, k, generator=g, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([X], lr=0.02)
    off = ~torch.eye(K, dtype=torch.bool)
    for step in range(steps):
        O = torch.stack([overlaps(q) for q in torch.linalg.qr(X)[0]])
        beta = 20 + 480 * step / steps  # anneal the soft maximum towards the true maximum
        loss = (torch.logsumexp(beta * O[:, off], dim=-1) / beta).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        O = torch.stack([overlaps(q) for q in torch.linalg.qr(X)[0]])
        return float(O[:, off].max(-1).values.min())


def learned(cap_paths):
    rows = []
    for p in cap_paths:
        with open(p) as f:
            c = json.load(f)
        final = str(max(int(k) for k in c["layers"] if k != "concat"))
        L = c["layers"][final]
        kept = L.get("kept", list(range(c["K"])))
        if len(kept) < 2 or c["d_model"] < 3:
            continue
        O = np.array(L["overlap"])
        pairs = [(i, j) for i in kept for j in kept if i < j]
        regime = c["regime"] + ("-shared" if c["regime"] == "mixture" and c.get("vocab") == "shared" else "")
        row = {"file": p, "regime": regime, "K": c["K"], "d_model": c["d_model"], "n_readable": c["d_model"] - 1,
               "n_kept": len(kept), "max_overlap": float(max(O[i, j] for i, j in pairs)),
               "mean_overlap": float(np.mean([O[i, j] for i, j in pairs]))}
        if "overlap_gamma" in L:
            Og = np.array(L["overlap_gamma"])
            row["max_overlap_gamma"] = float(max(Og[i, j] for i, j in pairs))
        if "overlap_logit" in L:
            Ol = np.array(L["overlap_logit"])
            row["max_overlap_logit"] = float(max(Ol[i, j] for i, j in pairs))
            row["min_plane_agreement"] = float(min(L["plane_agreement"][k] for k in kept))
        if "offset_sep" in L:
            S = np.array(L["offset_sep"])
            row["min_offset_sep"] = float(min(S[i, j] for i, j in pairs))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", nargs="+", required=True, help="*_cap.json files from capacity.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--restarts", type=int, default=64)
    args = ap.parse_args()

    paths = sorted({p for pat in args.caps for p in glob.glob(pat)} or set(args.caps))
    rows = learned(paths)
    if not rows:
        raise SystemExit("no usable *_cap.json files (need width >= 3 and at least 2 represented processes)")

    Ks = sorted({r["n_kept"] for r in rows})
    widths = sorted({r["d_model"] for r in rows})
    opt = {(K, d): optimal_packing(K, d - 1, restarts=args.restarts) for K in Ks for d in range(3, max(widths) + 1)}
    for r in rows:
        r["optimal_max_overlap"] = opt[(r["n_kept"], r["d_model"])]
        r["rankin_bound"] = rankin(r["n_kept"], r["d_model"] - 1)
        r["excess_over_optimal"] = r["max_overlap"] - r["optimal_max_overlap"]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["n_kept"], r["regime"], r["d_model"])))

    print("Optimal packings in the readable space R^(d-1) (largest pairwise overlap; Rankin bound in brackets):")
    for K in Ks:
        print(f"  {K} planes: " + ", ".join(f"width {d}: {opt[(K, d)]:.3f} ({rankin(K, d - 1):.3f})"
                                          for d in range(3, max(widths) + 1)))
    print("\nLearned vs optimal (largest overlap among represented processes, final layer), mean [min, max]:")
    by = defaultdict(list)
    for r in rows:
        by[(r["n_kept"], r["regime"], r["d_model"])].append(r)
    for (K, regime, d), rs in sorted(by.items()):
        m = [r["max_overlap"] for r in rs]
        line = (f"  {regime:>14}  {K} planes  width {d:>2}:  learned {np.mean(m):.3f} [{min(m):.3f}, {max(m):.3f}]"
                f" ({len(m)} models)   optimal {opt[(K, d)]:.3f}")
        if "max_overlap_gamma" in rs[0]:
            g = [r["max_overlap_gamma"] for r in rs]
            line += f"   γ-metric {np.mean(g):.3f} [{min(g):.3f}, {max(g):.3f}]"
        if "max_overlap_logit" in rs[0]:
            line += (f"   logit coords {np.mean([r['max_overlap_logit'] for r in rs]):.3f}"
                     f" (plane agreement {np.mean([r['min_plane_agreement'] for r in rs]):.3f})")
        if "min_offset_sep" in rs[0]:
            line += f"   offset separation {np.mean([r['min_offset_sep'] for r in rs]):.2f}"
        print(line)

    has_offsets = any("min_offset_sep" in r for r in rows)
    ncol = len(Ks) + (1 if has_offsets else 0)
    fig, axes = plt.subplots(1, ncol, figsize=(6.2 * ncol, 6.0), squeeze=False)
    ticks = [t for t in [3, 4, 5, 6, 8, 12, 16, 32, 64] if t <= max(widths)]
    for ax, K in zip(axes[0], Ks):
        ds = np.arange(3, max(widths) + 1)
        ax.plot(ds, [opt[(K, d)] for d in ds], color=INK, linewidth=2, label="optimal packing in ℝᵈ⁻¹ (numerical)")
        dd = np.linspace(3, max(widths), 300)
        ax.plot(dd, [rankin(K, x - 1) for x in dd], color=MUTED, linewidth=1.3, linestyle="--", label="Rankin bound")
        for regime in COLOURS:
            rs = [r for r in rows if r["n_kept"] == K and r["regime"] == regime]
            if rs:
                jitter = 1 + 0.025 * (np.arange(len(rs)) % 3 - 1)
                name = NAMES[regime]
                ax.scatter([r["d_model"] * j for r, j in zip(rs, jitter)], [r["max_overlap"] for r in rs], s=34,
                           color=COLOURS[regime], edgecolor="white", linewidth=0.7, zorder=3,
                           label=f"learned, {name} (Euclidean)")
                if "max_overlap_gamma" in rs[0]:
                    ax.scatter([r["d_model"] * j * 1.04 for r, j in zip(rs, jitter)], [r["max_overlap_gamma"] for r in rs],
                               s=34, facecolor="white", edgecolor=COLOURS[regime], linewidth=1.3, zorder=3,
                               label=f"learned, {name} (γ-metric)")
        ax.set_ylim(-0.03, 1.03)
        ax.axvline(2 * K + 1, color=RULE, linewidth=1, zorder=0)
        ax.set_title(f"{K} belief planes: overlap against the best possible packing", loc="left", fontsize=11, color=INK)
        ax.set_ylabel("largest pairwise overlap (mean cos² of principal angles)")
        ax.set_xlabel(f"residual width d  (readable space ℝᵈ⁻¹; grey line: d − 1 = {2 * K})")
        ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    if has_offsets:
        ax = axes[0, -1]
        for regime in ["mixture", "mixture-shared"]:
            rs = [r for r in rows if "min_offset_sep" in r and r["regime"] == regime]
            if rs:
                jitter = 1 + 0.025 * (np.arange(len(rs)) % 3 - 1)
                ax.scatter([r["d_model"] * j for r, j in zip(rs, jitter)], [r["min_offset_sep"] for r in rs], s=34,
                           color=COLOURS[regime], edgecolor="white", linewidth=0.7, zorder=3, label=NAMES[regime])
        ax.legend(frameon=False, fontsize=8.5)
        ax.axhline(1, color=MUTED, linewidth=1, linestyle="--")
        ax.text(max(widths), 1.05, "centres one spread apart", ha="right", va="bottom", fontsize=8.5, color=MUTED)
        ax.set_ylim(bottom=0)
        ax.set_title("Sparse regime: how far apart the two patches sit", loc="left", fontsize=11, color=INK)
        ax.set_ylabel("distance between process centres ÷ within-process spread")
        ax.set_xlabel("residual width d")
    for ax in axes[0]:
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.minorticks_off()
        ax.grid(alpha=0.3, color=RULE)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=160)
    print(f"\nsaved {args.out}.png and {args.out}.csv")


if __name__ == "__main__":
    main()
