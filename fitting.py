"""Affine regression helpers shared by the analysis scripts."""

import numpy as np


RIDGE_GRID = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


def fit_affine(A, Y, ridge=0.0):
    """Affine fit Y ~ A W^T + c with an optional ridge penalty. Returns W (S, d), c (S,).

    The penalty is ridge * mean(eigenvalue of the activation covariance), so it is scale-free.
    Why it matters: the interventions use the pseudo-inverse of W. An unregularised fit on
    nearly degenerate directions (e.g. layer 0, which takes only a few distinct values per
    position) gets huge coefficients, and its pseudo-inverse then gives edits that barely move
    the activation. A small ridge keeps W well conditioned.
    """
    mu_a, mu_y = A.mean(0), Y.mean(0)
    Ac, Yc = A - mu_a, Y - mu_y
    G = Ac.T @ Ac / len(A)
    lam = ridge * np.trace(G) / G.shape[0]
    Wt = np.linalg.solve(G + (lam + 1e-12) * np.eye(G.shape[0]), Ac.T @ Yc / len(A))
    W = Wt.T
    return W, mu_y - W @ mu_a


def mse(A, Y, W, c):
    return float(((A @ W.T + c - Y) ** 2).mean())


def choose_ridge(A, Y, rng):
    """Pick the ridge penalty that minimises held-out MSE on a random 50/50 split."""
    perm = rng.permutation(len(A))
    tr, te = perm[: len(A) // 2], perm[len(A) // 2:]
    scores = {r: mse(A[te], Y[te], *fit_affine(A[tr], Y[tr], r)) for r in RIDGE_GRID}
    return min(scores, key=scores.get)
