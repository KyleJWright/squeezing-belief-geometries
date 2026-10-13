"""Mess3 hidden Markov processes and exact Bayesian belief updates.

Conventions follow Shai et al. (2024), appendix A.3:

    T[x][i, j] = Pr(emit token x, move to hidden state j | currently in hidden state i)

A belief eta is a probability vector over hidden states (a point in the triangle for Mess3).
After seeing token x, the optimal observer updates it to

    eta' = eta T[x] / (eta T[x] 1)

and the probability of the next token x is eta T[x] 1.
"""

import itertools

import numpy as np


class HMM:
    def __init__(self, name, T, tokens):
        self.name = name
        self.T = np.asarray(T, dtype=float)  # shape (X, S, S)
        self.tokens = list(tokens)
        self.X, self.S = self.T.shape[0], self.T.shape[1]
        total = self.T.sum(axis=0)
        assert np.allclose(total.sum(axis=1), 1.0), "rows of sum_x T[x] must sum to 1"
        # Stationary distribution: left eigenvector of sum_x T[x] with eigenvalue 1.
        w, v = np.linalg.eig(total.T)
        pi = np.real(v[:, np.argmin(np.abs(w - 1.0))])
        self.pi = pi / pi.sum()
        # Joint (token, next state) table per current state, for sampling.
        self._flat_cdf = self.T.transpose(1, 0, 2).reshape(self.S, self.X * self.S).cumsum(axis=1)

    def sample(self, n, length, rng):
        """n token sequences of the given length, starting from the stationary distribution."""
        s = rng.choice(self.S, size=n, p=self.pi)
        out = np.empty((n, length), dtype=np.int64)
        for t in range(length):
            u = rng.random(n)[:, None]
            idx = np.minimum((u > self._flat_cdf[s]).sum(axis=1), self.X * self.S - 1)
            out[:, t] = idx // self.S
            s = idx % self.S
        return out

    def beliefs_along(self, seqs):
        """Exact beliefs after each prefix.

        seqs: (N, L) token ids. Returns beliefs (N, L, S), where beliefs[n, t] is the belief after
        seeing seqs[n, :t+1], and prefix probabilities (N, L) (0 means impossible).
        """
        seqs = np.asarray(seqs)
        N, L = seqs.shape
        eta = np.tile(self.pi, (N, 1))
        p = np.ones(N)
        B = np.zeros((N, L, self.S))
        P = np.zeros((N, L))
        for t in range(L):
            v = np.einsum("ns,nsr->nr", eta, self.T[seqs[:, t]])
            z = v.sum(axis=1)
            p = p * z
            eta = v / np.where(z > 0, z, 1.0)[:, None]
            B[:, t] = eta
            P[:, t] = p
        return B, P

    def next_token_probs(self, beliefs):
        """Optimal next-token distribution for beliefs of shape (..., S): (..., X)."""
        return np.einsum("...s,xsr->...x", beliefs, self.T)

    def possible_sequences(self, length):
        """All sequences of the given length with nonzero probability, with their beliefs and probs."""
        seqs = np.array(list(itertools.product(range(self.X), repeat=length)), dtype=np.int64)
        B, P = self.beliefs_along(seqs)
        keep = P[:, -1] > 0
        return seqs[keep], B[keep], P[keep]

    def optimal_loss(self, length):
        """Cross-entropy (nats per token) of the Bayes-optimal predictor, as in training: inputs
        x_0..x_{L-1}, target at position t is x_{t+1}, given x_0..x_t."""
        seqs, B, P = self.possible_sequences(length)
        w = P[:, -1] / P[:, -1].sum()
        total = 0.0
        for t in range(length):
            nxt = self.next_token_probs(B[:, t]).clip(1e-300)
            total += (w * -(nxt * np.log(nxt)).sum(axis=1)).sum()
        return total / length


def mess3(x=0.05, alpha=0.85, name="mess3"):
    """Mess3 with parameters (x, alpha). The defaults are the process in Shai et al. (A.3).

    T[k][i, j] = (alpha if j == k else beta) * (y if i == j else x),  beta = (1 - alpha)/2,  y = 1 - 2x.
    """
    beta, y = (1 - alpha) / 2, 1 - 2 * x
    T = np.zeros((3, 3, 3))
    for k in range(3):
        for i in range(3):
            for j in range(3):
                T[k, i, j] = (alpha if j == k else beta) * (y if i == j else x)
    return HMM(name, T, tokens="ABC")


def get_process(name):
    if name != "mess3":
        raise ValueError(f"unknown process {name!r}")
    return mess3()
