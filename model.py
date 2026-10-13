"""A small pre-LayerNorm transformer with access to the residual stream.

Architecture as in Shai et al. (2024), appendix A.6: 4 layers, residual width d_model (64 in the
paper), 1 attention head per layer with head size 8, MLP width 256, ReLU, context length 10,
LayerNorm, causal attention. Only d_model is varied in this project, so the residual stream is the
part being squeezed.

forward(tokens, return_resid=True) also returns the residual stream at layer indices
0 (after the embedding) through n_layers (the final residual stream, before the last LayerNorm).
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    d_vocab: int
    n_ctx: int = 10
    d_model: int = 64
    n_layers: int = 4
    n_heads: int = 1
    d_head: int = 8
    d_mlp: int = 256


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        inner = cfg.n_heads * cfg.d_head
        self.qkv = nn.Linear(cfg.d_model, 3 * inner)
        self.out = nn.Linear(inner, cfg.d_model)

    def forward(self, x):
        B, T, _ = x.shape
        h, dh = self.cfg.n_heads, self.cfg.d_head
        q, k, v = self.qkv(x).split(h * dh, dim=-1)
        q, k, v = (t.view(B, T, h, dh).transpose(1, 2) for t in (q, k, v))
        z = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(z.transpose(1, 2).reshape(B, T, h * dh))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_mlp), nn.ReLU(), nn.Linear(cfg.d_mlp, cfg.d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class Transformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.d_vocab, cfg.d_model)
        self.pos = nn.Parameter(torch.randn(cfg.n_ctx, cfg.d_model) * 0.02)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.d_vocab)

    def forward(self, tokens, return_resid=False):
        x = self.embed(tokens) + self.pos[: tokens.shape[1]]
        resid = [x]
        for block in self.blocks:
            x = block(x)
            resid.append(x)
        logits = self.unembed(self.ln_final(x))
        return (logits, resid) if return_resid else logits


def save_checkpoint(path, model, process_name, extra=None):
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict(),
                "process": process_name, "extra": extra or {}}, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = Transformer(Config(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
