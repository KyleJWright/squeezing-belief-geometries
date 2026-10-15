"""Train the small transformer on next-token prediction.

One Mess3 process:
    python train.py --process mess3 --d_model 64 --steps 20000 --out runs/single_d64.pt

Two Mess3 processes sharing one residual stream:
    python train.py --regime mixture  --K 2 --d_model 4 --steps 15000 --out runs/mixture_K2_d4.pt
    python train.py --regime parallel --K 2 --d_model 4 --steps 15000 --out runs/parallel_K2_d4.pt

Only the residual-stream width (--d_model) is varied; attention head size and MLP width stay
fixed, so the residual stream is the bottleneck. The log reports loss as a ratio to the
Bayes-optimal loss (1.00 = optimal). Fresh data is sampled every step. Checkpoints are saved
periodically, so intermediate snapshots can be analysed while training continues.
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from composite import build, default_spec
from model import Config, Transformer, save_checkpoint
from processes import get_process


class SingleProcess:
    """Adapter so a single HMM looks like a Composite to the training loop."""

    def __init__(self, hmm):
        self.hmm, self.name, self.d_vocab = hmm, hmm.name, hmm.X

    def sample(self, n, length, rng):
        return self.hmm.sample(n, length, rng), None

    def optimal_loss(self, n_ctx):
        return self.hmm.optimal_loss(n_ctx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--process", choices=["mess3"], help="single-process mode")
    ap.add_argument("--regime", choices=["mixture", "parallel"], help="multi-process mode")
    ap.add_argument("--K", type=int, default=2, help="number of Mess3 processes (multi-process mode)")
    ap.add_argument("--vocab", choices=["disjoint", "shared"], default="disjoint", help="mixture only")
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--d_head", type=int, default=8)
    ap.add_argument("--d_mlp", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_ctx", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=1000)
    ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument("--snapshots", type=int, nargs="*", default=[],
                    help="also save a separate checkpoint <out>_step<N>.pt at each of these steps")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if (args.process is None) == (args.regime is None):
        ap.error("give exactly one of --process or --regime")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    if args.process:
        data, spec, label = SingleProcess(get_process(args.process)), None, args.process
    else:
        spec = default_spec(args.regime, args.K, args.vocab)
        data, label = build(spec), "composite"
    opt_loss = data.optimal_loss(args.n_ctx)
    print(f"{data.name}: vocab {data.d_vocab}, optimal loss {opt_loss:.4f} nats/token, d_model {args.d_model}")

    cfg = Config(d_vocab=data.d_vocab, n_ctx=args.n_ctx, d_model=args.d_model, n_layers=args.n_layers,
                 d_head=args.d_head, d_mlp=args.d_mlp)
    model = Transformer(cfg).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.05)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters on {args.device}")

    t0, running = time.time(), []
    for step in range(1, args.steps + 1):
        seqs, _ = data.sample(args.batch, args.n_ctx + 1, rng)
        seqs = torch.from_numpy(seqs).to(args.device)
        logits = model(seqs[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, data.d_vocab), seqs[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        running.append(loss.item())
        if step % args.log_every == 0:
            mean = float(np.mean(running))
            running = []
            print(f"step {step:>7}  loss {mean:.4f}  ratio to optimal {mean / opt_loss:.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if step in args.snapshots:
            save_checkpoint(args.out[:-3] + f"_step{step}.pt", model.cpu(), label,
                            {"step": step, "optimal_loss": opt_loss, "spec": spec, "final_loss": float(loss.item())})
            model.to(args.device)
        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(args.out, model.cpu(), label,
                            {"step": step, "optimal_loss": opt_loss, "spec": spec, "final_loss": float(loss.item())})
            model.to(args.device)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
