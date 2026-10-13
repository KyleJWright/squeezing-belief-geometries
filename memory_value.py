"""How much is each Mess3 process's memory worth, in nats of next-token loss?

    python memory_value.py

For each process in the two-process experiments, compares three predictors of the next token:
  no context    the stationary token distribution (uniform for Mess3)
  last token    the best predictor that sees only the previous token
  optimal       the Bayes-optimal predictor, which uses the exact belief (the whole prefix)
The gap between "last token" and "optimal" is what a model loses by keeping only the last token,
i.e. by dropping the fine structure of the belief simplex.
"""

import numpy as np

from composite import MESS3_VARIANTS
from processes import mess3


def memory_value(x, alpha, n=50000, length=11, seed=0):
    rng = np.random.default_rng(seed)
    h = mess3(x, alpha)
    seqs = h.sample(n, length, rng)
    B, _ = h.beliefs_along(seqs[:, :-1])
    P = h.next_token_probs(B)  # P[:, t] predicts seqs[:, t + 1]
    # Positions with a previous token, so all three predictors are scored on the same tokens.
    prev, cur = seqs[:, 1:-1].ravel(), seqs[:, 2:].ravel()
    optimal = -np.log(np.take_along_axis(P[:, 1:], seqs[:, 2:, None], -1)).mean()
    C = np.zeros((h.X, h.X))
    np.add.at(C, (prev, cur), 1)
    C /= C.sum(1, keepdims=True)
    last = -np.log(C[prev, cur]).mean()
    marg = np.bincount(cur, minlength=h.X) / len(cur)
    none = -(marg * np.log(marg)).sum()
    return none, last, optimal


def main():
    print(f"{'process':>8} {'x':>5} {'alpha':>6} {'no context':>11} {'last token':>11} {'optimal':>8}"
          f" {'value of last token':>20} {'value of memory':>16}")
    for name, (x, a) in zip("AB", MESS3_VARIANTS[:2]):
        none, last, opt = memory_value(x, a)
        print(f"{name:>8} {x:>5} {a:>6} {none:>11.4f} {last:>11.4f} {opt:>8.4f} {none - last:>20.4f} {last - opt:>16.4f}")


if __name__ == "__main__":
    main()
