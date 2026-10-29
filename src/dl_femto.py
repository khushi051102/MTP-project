"""Phase 4.2: small sequence models for FEMTO (leave-one-bearing-out, multi-seed).

Input: sequences (n, 30, 8) of log-ratio features; target: rul_frac in [0, 1]
(1.0 until the first predicting time, then falling to 0). Headline metric: macro
MAE over the post-FPT snapshots of each held-out bearing (no-skill baseline = 0.5).
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import SEED
from .dl import count_params, get_device
from .utils import set_seed


# ------------------------------------------------------------------ models
class GRUReg(nn.Module):
    """One GRU layer; the last hidden state summarises the 30 snapshots."""

    def __init__(self, n_in=8, hidden=32):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                       # x: (batch, time, features)
        _, h = self.gru(x)                      # h: (1, batch, hidden)
        return torch.sigmoid(self.head(h[-1])).squeeze(-1)   # (batch,), inside (0, 1)


class TCNReg(nn.Module):
    """Three temporal convolutions (conv over time), then average over time."""

    def __init__(self, n_in=8, ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_in, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, 5, padding=2, stride=2), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.ReLU())
        self.head = nn.Linear(ch, 1)

    def forward(self, x):
        z = self.net(x.transpose(1, 2)).mean(dim=-1)        # (batch, ch)
        return torch.sigmoid(self.head(z)).squeeze(-1)


class TransReg(nn.Module):
    """Tiny transformer encoder: learned positions + 2 self-attention layers."""

    def __init__(self, n_in=8, d=32, heads=4, layers=2, ff=64, seq_len=30):
        super().__init__()
        self.inp = nn.Linear(n_in, d)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d))
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        z = self.enc(self.inp(x) + self.pos)                # (batch, time, d)
        return torch.sigmoid(self.head(z.mean(dim=1))).squeeze(-1)


MODELS = {"GRU": GRUReg, "TCN": TCNReg, "Transformer": TransReg}


# ---------------------------------------------------------------- training
def train_regressor(model, X, y, w, epochs=15, lr=2e-3, wd=1e-2, bs=256, seed=SEED, device=None):
    """Weighted L1 loss (degrading snapshots count `w` times more than healthy ones,
    because the headline metric is measured on the degrading ones). Fixed epochs."""
    device = device or get_device()
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = model.to(device)
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32),
                       torch.as_tensor(w, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=bs, shuffle=True, generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(dl))
    for _ in range(epochs):
        model.train()
        for xb, yb, wb in dl:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad()
            loss = (wb * (model(xb) - yb).abs()).sum() / wb.sum()
            loss.backward()
            opt.step()
            sched.step()
    return model


@torch.no_grad()
def predict_reg(model, X, bs=1024):
    model.eval()
    dev = next(model.parameters()).device
    out = [model(torch.as_tensor(X[i:i + bs], dtype=torch.float32, device=dev)).cpu().numpy()
           for i in range(0, len(X), bs)]
    return np.concatenate(out)


# -------------------------------------------------------------------- LOBO
def run_femto_models(X, y, meta, models=("GRU", "TCN", "Transformer"), seeds=(1,),
                     epochs=15, w_post=4.0, bearings=None, verbose=True):
    """Leave-one-bearing-out. Features are standardised with the TRAINING bearings'
    mean/std only. Returns a long table: seed, model, bearing, MAE_post, MAE_all, params."""
    post_all = meta["post_fpt"].to_numpy().astype(bool)
    names = bearings or sorted(meta["bearing"].unique())
    rows = []
    for seed in seeds:
        for b in names:
            te = (meta["bearing"] == b).to_numpy()
            tr = ~te
            mu = X[tr].mean(axis=(0, 1), keepdims=True)
            sd = X[tr].std(axis=(0, 1), keepdims=True) + 1e-6
            Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
            w = np.where(post_all[tr], w_post, 1.0)
            for name in models:
                set_seed(seed)                      # seeds the initial weights
                model = MODELS[name]()
                train_regressor(model, Xtr, y[tr], w, epochs=epochs, seed=seed)
                p = predict_reg(model, Xte)
                err = np.abs(p - y[te])
                rows.append({"seed": seed, "model": name, "bearing": b,
                             "MAE_post": float(err[post_all[te]].mean()), "MAE_all": float(err.mean()),
                             "n_post": int(post_all[te].sum()), "params": count_params(model)})
            if verbose:
                print(f"seed {seed}  held-out {b} done", flush=True)
    return pd.DataFrame(rows)
