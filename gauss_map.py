"""Is the belief patch flat? The Gauss map of the embedding of beliefs in the residual stream.

    python gauss_map.py            # all trained models of width >= 4 in runs/seed*/ (a few minutes)

The rest of the project treats each belief as filling a flat patch of a 2-plane: a ~ m + M (eta - c).
A linear decoder cannot test this. A surface that bends into an extra direction, a = m + M u + f(u) n,
decodes perfectly by projecting n away. This script measures the bending directly.

Gauss map. At many anchor beliefs u0, the local tangent plane of the embedding u -> a is estimated by a
local affine fit of activation on belief over the k nearest beliefs: a ~ a0 + J (u - u0). The plane
T(u0) = span J is a point of the Grassmannian Gr(2, d-1) (readable space, as in capacity.py). The patch
is flat exactly when T is constant, so the spread of T measures curvature, in the same units as the rest
of the project:

  tilt          angle between each local plane and the global plane V = span M, as an RMS principal
                angle, arccos(sqrt(overlap)), in degrees; mean over anchors.
  tilt_flat     the same for a synthetic flat patch: the global affine fit plus the model's own residuals,
                shuffled between points. Finite neighbourhoods and unrelated activation variance make
                local planes wobble even for a flat patch; this is that floor. Curvature is the excess
                tilt - tilt_flat.
  quadratic     fraction of activation variance gained by adding quadratic terms in the belief to the
                fit, split into the part inside the global plane (a reparametrisation of the plane,
                e.g. log(eta N) instead of eta: not curvature) and the part outside it (bending: curvature).

The Gauss map ignores how the patch is parametrised; it only sees which plane it is tangent to. Local
planes are only estimated where the nearby beliefs spread in two directions (the fractal has thin
filaments); other anchors are skipped and counted.

  bend          the size of all the bending, at every scale: the conditional mean E[a | u], estimated by
                averaging activations over nearby beliefs, minus the global affine fit, split into the part
                outside the belief plane(s) (bend out) and inside (bend in), each as a fraction of the
                variance the affine fit explains. bend out (flat) is the same for the synthetic flat patch.
                Left blank for dense models whose two planes already fill the readable space.

Models: one process, and two processes per process, all on sampled sequences
(sparse with separate or shared tokens: each process on its own sequences; dense: both beliefs jointly,
with the local fit on all four coordinates and each process's plane read from its own columns). Width 3
is left out: its readable space is a plane, so every local plane is the whole space. Processes whose belief
does not decode linearly (held-out R^2 < 0.5, e.g. the dense process B at narrow widths) are left out of
the summary plots: a plane fitted to a belief the model has dropped is noise.

Output: results/gauss/gauss.png (curvature maps and summary), gauss.csv, gauss.json.
"""

import argparse
import csv
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

from capacity import E, collect_resid, readable
from composite import build
from fitting import fit_affine
from model import load_checkpoint
from processes import get_process
from simplex_figures import CORNERS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "gauss")
INK, MUTED, RULE = "#1f1f1e", "#5f5e5a", "#c3c2b7"
REGIME_COLOURS = {"single": INK, "mixture": "#2a78d6", "mixture-shared": "#1baf7a", "parallel": "#eb6834"}
REGIME_NAMES = {"single": "one process", "mixture": "sparse, separate tokens", "mixture-shared": "sparse, shared tokens",
                "parallel": "dense"}


def orth(M):
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    return U[:, s > 1e-8 * max(s.max(), 1e-30)]


def overlap(Q1, Q2):
    return float(np.linalg.norm(Q1.T @ Q2) ** 2 / max(min(Q1.shape[1], Q2.shape[1]), 1))


def rms_angle_deg(ov):
    return float(np.degrees(np.arccos(np.sqrt(np.clip(ov, 0, 1)))))


def farthest_points(U, n, rng):
    """n anchor indices spread over the point cloud U (farthest-point sampling)."""
    idx = [int(rng.integers(len(U)))]
    d = np.linalg.norm(U - U[idx[0]], axis=1)
    for _ in range(n - 1):
        idx.append(int(d.argmax()))
        d = np.minimum(d, np.linalg.norm(U - U[idx[-1]], axis=1))
    return np.array(idx)


def local_jacobians(U, A, anchors, k, min_spread=0.05):
    """Local affine fit A ~ a0 + J (U - u0) over the k nearest points (in belief coordinates) of each anchor.

    Returns a list of J (d, p) or None where the neighbourhood does not spread in every belief direction:
    the ratio of the smallest to the largest standard deviation of the local beliefs must be >= min_spread.
    """
    out = []
    for i in anchors:
        dist = np.linalg.norm(U - U[i], axis=1)
        nn = np.argpartition(dist, min(k, len(U) - 1))[:k]
        Uc = U[nn] - U[nn].mean(0)
        ev = np.linalg.eigvalsh(Uc.T @ Uc / len(nn))
        if ev[0] <= 0 or np.sqrt(ev[0] / ev[-1]) < min_spread:
            out.append(None)
            continue
        Ac = A[nn] - A[nn].mean(0)
        J = np.linalg.lstsq(Uc, Ac, rcond=None)[0].T  # (d, p)
        out.append(J)
    return out


def quadratic_gain(U, A, Q):
    """Variance gained by quadratic terms in U, as fractions of total activation variance: (in-plane, out)."""
    p = U.shape[1]
    quad = np.stack([U[:, i] * U[:, j] for i in range(p) for j in range(i, p)], axis=1)
    W1, c1 = fit_affine(U, A)
    W2, c2 = fit_affine(np.concatenate([U, quad], 1), A)
    delta = (np.concatenate([U, quad], 1) @ W2.T + c2) - (U @ W1.T + c1)
    delta -= delta.mean(0)
    total = ((A - A.mean(0)) ** 2).sum()
    inside = delta @ Q @ Q.T
    return float((inside ** 2).sum() / total), float(((delta - inside) ** 2).sum() / total)


def knn_mean(U, X, queries, k, chunk=256):
    """Mean of X over the k nearest points (in U) of each query point: an estimate of E[X | u]."""
    out = np.empty((len(queries), X.shape[1]))
    for s in range(0, len(queries), chunk):
        q = queries[s:s + chunk]
        dist = ((U[None, :, :] - U[q][:, None, :]) ** 2).sum(-1)
        nn = np.argpartition(dist, k, axis=1)[:, :k]
        out[s:s + chunk] = X[nn].mean(1)
    return out


def bending(U, A, fit, Q, queries, k):
    """Deviation of the conditional mean E[a | u] from the global affine fit, at all scales, as a fraction
    of the variance the affine fit explains: (out of the belief planes' span, inside it)."""
    dev = knn_mean(U, A, queries, k) - fit[queries]
    inside = dev @ Q @ Q.T
    explained = ((fit - fit.mean(0)) ** 2).sum(1).mean()
    return float(((dev - inside) ** 2).sum(1).mean() / explained), float((inside ** 2).sum(1).mean() / explained)


def analyse(U, A, groups, rng, n_anchors, k):
    """U (N, p) belief coordinates, A (N, d) readable activations; groups: list of column slices, one per
    process. Returns per-process results."""
    W, c = fit_affine(U, A)  # A ~ U W^T + c, W: (d, p)
    fit = U @ W.T + c
    resid = A - fit
    flat = fit + resid[rng.permutation(len(U))]
    anchors = farthest_points(U, n_anchors, rng)
    Js = local_jacobians(U, A, anchors, k)
    Js_flat = local_jacobians(U, flat, anchors, k)
    # Quadratic terms in all belief coordinates; "in plane" means inside the span of all the belief planes.
    q_in, q_out = quadratic_gain(U, A, orth(W))
    queries = rng.choice(len(U), size=min(2000, len(U)), replace=False)
    kb = min(200, max(20, len(U) // 100))
    span = orth(W)
    if span.shape[1] >= np.linalg.matrix_rank(A - A.mean(0), tol=1e-6 * np.abs(A).max()):
        # The belief planes already span the whole readable space (dense, width <= 5): nothing is "out of plane".
        bend_out = bend_in = bend_out_flat = q_in = q_out = float("nan")
    else:
        bend_out, bend_in = bending(U, A, fit, span, queries, kb)
        bend_out_flat, _ = bending(U, flat, fit, span, queries, kb)
    linear_r2 = float(1 - (resid ** 2).sum() / ((A - A.mean(0)) ** 2).sum())
    half = rng.random(len(U)) < 0.5
    out = []
    for g in groups:
        Wd, cd = fit_affine(A[half], U[half][:, g])  # linear decode of this belief, for "is it represented?"
        err = U[~half][:, g] - (A[~half] @ Wd.T + cd)
        decode_r2 = float(1 - (err ** 2).sum() / ((U[~half][:, g] - U[~half][:, g].mean(0)) ** 2).sum())
        V = orth(W[:, g])
        tilt = [rms_angle_deg(overlap(orth(J[:, g]), V)) if J is not None else np.nan for J in Js]
        tilt_flat = [rms_angle_deg(overlap(orth(J[:, g]), V)) if J is not None else np.nan for J in Js_flat]
        planes = [orth(J[:, g]) for J in Js if J is not None]
        pair = [rms_angle_deg(overlap(planes[i], planes[j])) for i in range(len(planes)) for j in range(i)]
        out.append({"anchors": U[anchors][:, g].tolist(), "tilt_per_anchor": tilt,
                    "tilt": float(np.nanmean(tilt)), "tilt_flat": float(np.nanmean(tilt_flat)),
                    "max_angle_between_local_planes": float(max(pair)) if pair else float("nan"),
                    "skipped_anchors": int(sum(J is None for J in Js)),
                    "quadratic_in_plane": q_in, "quadratic_out_of_plane": q_out,
                    "bend_out_of_plane": bend_out, "bend_in_plane": bend_in, "bend_out_of_plane_flat": bend_out_flat,
                    "linear_r2": linear_r2, "decode_r2": decode_r2})
    return out


def regime_of(comp):
    return comp.regime + ("-shared" if comp.regime == "mixture" and comp.vocab == "shared" else "")


def model_rows(path, n_seq, n_anchors, k_frac, seed):
    model, ckpt = load_checkpoint(path)
    d = model.cfg.d_model
    P = readable(d)
    rng = np.random.default_rng(seed)
    rows = []
    seed_id = int(re.search(r"seed(\d+)", path).group(1)) if re.search(r"seed(\d+)", path) else 0
    base = {"file": os.path.relpath(path, HERE), "seed": seed_id, "d_model": d}
    spec = ckpt["extra"].get("spec")
    if spec is None:  # one process, on sampled sequences like the two-process models
        hmm = get_process(ckpt.get("process", "mess3"))
        tokens = hmm.sample(n_seq, model.cfg.n_ctx, rng)
        U = (hmm.beliefs_along(tokens)[0].reshape(-1, 3) - 1 / 3) @ E
        A = collect_resid(model, tokens)[-1].reshape(-1, d).astype(float) @ P
        res = analyse(U, A, [slice(0, 2)], rng, n_anchors, max(200, int(k_frac * len(U))))
        rows.append({**base, "regime": "single", "process": "-", **res[0]})
        return rows
    comp = build(spec)
    L = model.cfg.n_ctx
    tokens, labels = comp.sample(n_seq, L, rng)
    beliefs = comp.factor_beliefs(tokens, labels)
    A_seq = collect_resid(model, tokens)[-1] @ P
    regime = regime_of(comp)
    if comp.regime == "parallel":
        U = np.concatenate([(B.reshape(-1, 3) - 1 / 3) @ E for B, _ in beliefs], 1)
        res = analyse(U, A_seq.reshape(-1, d), [slice(2 * j, 2 * j + 2) for j in range(comp.K)], rng, n_anchors,
                      max(200, int(k_frac * len(U))))
        for j, r in enumerate(res):
            rows.append({**base, "regime": regime, "process": "AB"[j], **r})
    else:
        for j, (B, mask) in enumerate(beliefs):
            U = (B[mask].reshape(-1, 3) - 1 / 3) @ E
            r = analyse(U, A_seq[mask].reshape(-1, d), [slice(0, 2)], rng, n_anchors, max(200, int(k_frac * len(U))))[0]
            rows.append({**base, "regime": regime, "process": "AB"[j], **r})
    return rows


def triangle(ax, title):
    ax.plot(*np.vstack([CORNERS, CORNERS[:1]]).T, color=RULE, linewidth=1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9, color=INK)


def to_xy(U2):
    eta = U2 @ E.T + 1 / 3
    return eta @ CORNERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="*", help="default: runs/seed*/ single, mixture, shared, parallel models")
    ap.add_argument("--min_width", type=int, default=4)
    ap.add_argument("--n_seq", type=int, default=6000, help="sampled sequences per two-process model")
    ap.add_argument("--anchors", type=int, default=64)
    ap.add_argument("--k_frac", type=float, default=0.1,
                    help="neighbours per local tangent-plane fit, as a fraction of the points (min 200)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)

    paths = args.ckpts or sorted(p for pat in ["single_d*.pt", "mixture_K2_d*.pt", "shared_K2_d*.pt", "parallel_K2_d*.pt"]
                                 for p in glob.glob(os.path.join(HERE, "runs", "seed*", pat)))
    paths = [p for p in paths if "_step" not in p and int(re.search(r"_d(\d+)", os.path.basename(p)).group(1)) >= args.min_width]
    if not paths:
        raise SystemExit("no trained models found: run python run_simplex.py first")
    print(f"{len(paths)} models, {args.jobs} at a time", flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = [r for rs in pool.map(lambda p: model_rows(p, args.n_seq, args.anchors, args.k_frac, 0), paths) for r in rs]
    with open(os.path.join(OUT, "gauss.json"), "w") as f:
        json.dump(rows, f, indent=1)
    fields = ["regime", "process", "d_model", "seed", "tilt", "tilt_flat", "max_angle_between_local_planes",
              "bend_out_of_plane", "bend_out_of_plane_flat", "bend_in_plane", "quadratic_in_plane",
              "quadratic_out_of_plane", "linear_r2", "decode_r2", "skipped_anchors", "file"]
    with open(os.path.join(OUT, "gauss.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["regime"], r["process"], -r["d_model"], r["seed"])))

    by = defaultdict(list)
    for r in rows:
        by[(r["regime"], r["process"], r["d_model"])].append(r)
    print("\nGauss map of the belief patch, final layer, mean over seeds (decode R2 < 0.5: belief not represented;"
          " ignore that row)")
    print(f"{'regime':>15} {'proc':>4} {'width':>5}   {'tilt (deg)':>10} {'flat floor':>10} {'excess':>7}"
          f"   {'bend out':>8} {'(flat)':>7} {'bend in':>8}   {'quad out':>8}   {'decode R2':>9}")
    for (regime, proc, d), rs in sorted(by.items(), key=lambda kv: (list(REGIME_NAMES).index(kv[0][0]), kv[0][1], -kv[0][2])):
        t, tf = np.mean([r["tilt"] for r in rs]), np.mean([r["tilt_flat"] for r in rs])
        print(f"{regime:>15} {proc:>4} {d:>5}   {t:>10.1f} {tf:>10.1f} {t - tf:>7.1f}"
              f"   {100 * np.mean([r['bend_out_of_plane'] for r in rs]):>7.1f}%"
              f" {100 * np.mean([r['bend_out_of_plane_flat'] for r in rs]):>6.1f}%"
              f" {100 * np.mean([r['bend_in_plane'] for r in rs]):>7.1f}%"
              f"   {100 * np.mean([r['quadratic_out_of_plane'] for r in rs]):>7.2f}%"
              f"   {np.mean([r['decode_r2'] for r in rs]):>9.2f}")

    # Figure: curvature maps for a few models (first seed; process A, or the single process), then the
    # excess tilt and the out-of-plane bending by width, mean over seeds.
    s0 = min(x["seed"] for x in rows)
    want = [("single", 64), ("single", 8), ("single", 4), ("mixture", 8), ("mixture", 4),
            ("mixture-shared", 8), ("mixture-shared", 4), ("parallel", 8), ("parallel", 4)]
    maps = [m for m in want if any(r["regime"] == m[0] and r["d_model"] == m[1] and r["seed"] == s0 for r in rows)]
    ncol = len(maps)
    fig = plt.figure(figsize=(max(2.2 * ncol, 12), 7.6))
    gs = fig.add_gridspec(2, ncol, height_ratios=[1, 1.5], hspace=0.35)
    vmax = np.nanpercentile([t for r in rows for t in r["tilt_per_anchor"]], 95)
    short = {"single": "one process", "mixture": "sparse, separate", "mixture-shared": "sparse, shared", "parallel": "dense"}
    sc = None
    for i, (regime, d) in enumerate(maps):
        ax = fig.add_subplot(gs[0, i])
        r = next(x for x in rows if x["regime"] == regime and x["d_model"] == d and x["seed"] == s0
                 and x["process"] in ("-", "A"))
        triangle(ax, f"{short[regime]}\nwidth {d}" + ("" if r["process"] == "-" else ", A"))
        xy = to_xy(np.array(r["anchors"]))
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=r["tilt_per_anchor"], cmap="viridis", vmin=0, vmax=vmax, s=14)
    cbar = fig.colorbar(sc, ax=fig.axes[:ncol], fraction=0.015, pad=0.01)
    cbar.set_label("local tilt from\nglobal plane (deg)", fontsize=8)

    ticks = sorted({r["d_model"] for r in rows})
    panels = [("excess tilt over flat floor (deg)", lambda r: r["tilt"] - r["tilt_flat"],
               "Gauss map: how far local tangent planes tilt from the global plane"),
              ("out-of-plane bending (% of explained variance)", lambda r: 100 * r["bend_out_of_plane"],
               "How much the patch bends out of its plane, at all scales")]
    half = ncol // 2
    for j, (ylabel, get, title) in enumerate(panels):
        ax = fig.add_subplot(gs[1, j * half:(j + 1) * half if j == 0 else ncol])
        for regime in REGIME_NAMES:
            for proc, marker in [("-", "o"), ("A", "o"), ("B", "s")]:
                rs = [r for r in rows if r["regime"] == regime and r["process"] == proc and r["decode_r2"] >= 0.5]
                if not rs:
                    continue
                ws = sorted({r["d_model"] for r in rs})
                ys = [np.nanmean([get(r) for r in rs if r["d_model"] == w]) if any(
                    np.isfinite(get(r)) for r in rs if r["d_model"] == w) else np.nan for w in ws]
                ax.plot(ws, ys, marker=marker, color=REGIME_COLOURS[regime], linewidth=1.3,
                        linestyle="-" if proc in ("-", "A") else "--",
                        label=REGIME_NAMES[regime] + ("" if proc == "-" else f", process {proc}"))
        ax.axhline(0, color=RULE, linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("residual width d")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.grid(alpha=0.3, color=RULE)
        ax.spines[["top", "right"]].set_visible(False)
        if j == 1:
            ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.subplots_adjust(left=0.05, right=0.84, top=0.93, bottom=0.08, wspace=0.35)
    fig.savefig(os.path.join(OUT, "gauss.png"), dpi=150)
    print(f"\nsaved {os.path.relpath(os.path.join(OUT, 'gauss.png'), HERE)}, gauss.csv and gauss.json")


if __name__ == "__main__":
    main()
