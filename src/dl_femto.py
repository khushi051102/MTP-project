"""Phase 4: sequence models for FEMTO (leave-one-bearing-out, multi-seed).

Small models (GRU, TCN, Transformer) are the edge candidates; GRU-L and
Transformer-L are ~100x larger versions of the same designs, trained with the
SAME recipe (epochs, lr, weight decay) so that only model size changes.

Input: sequences (n, 30, 8) of log-ratio features; target: rul_frac in [0, 1]
(1.0 until the first predicting time, then falling to 0). Headline metric: macro
MAE over the post-FPT snapshots of each held-out bearing (no-skill baseline = 0.5).
"""
import subprocess
import time

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
    """GRU layer(s); the last layer's final hidden state summarises the 30 snapshots."""

    def __init__(self, n_in=8, hidden=32, layers=1):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                       # x: (batch, time, features)
        _, h = self.gru(x)                      # h: (layers, batch, hidden)
        return torch.sigmoid(self.head(h[-1])).squeeze(-1)   # last layer -> (batch,), inside (0, 1)


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


MODELS = {
    # small models (the edge candidates)
    "GRU": GRUReg, "TCN": TCNReg, "Transformer": TransReg,
    # heavier versions of the same designs (the "heavy pipeline" side of the comparison)
    "GRU-L": lambda: GRUReg(hidden=256, layers=2),
    "Transformer-L": lambda: TransReg(d=128, heads=8, layers=4, ff=512),
}


# ------------------------------------------------------------ bookkeeping
def git_commit():
    """Short hash of the current commit, so every results file can be traced to its code."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _sync(device):
    """GPU work is asynchronous: wait for it to finish before reading the clock."""
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


# ---------------------------------------------------------------- training
def train_regressor(model, X, y, w, epochs=15, lr=2e-3, wd=1e-2, bs=256, seed=SEED,
                    device=None, return_history=False):
    """Weighted L1 loss (degrading snapshots count `w` times more than healthy ones,
    because the headline metric is measured on the degrading ones). Fixed epochs.

    return_history=False -> returns the model (old behaviour, so other callers still work).
    return_history=True  -> returns (model, epoch_losses), where epoch_losses[e] is the
                            mean training loss over the batches of epoch e.
    """
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
    epoch_losses = []
    for _ in range(epochs):
        model.train()
        running, n_batches = 0.0, 0
        for xb, yb, wb in dl:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad()
            loss = (wb * (model(xb) - yb).abs()).sum() / wb.sum()
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()          # .item() copies one number to the CPU
            n_batches += 1
        epoch_losses.append(running / n_batches)
    if return_history:
        return model, epoch_losses
    return model


@torch.no_grad()
def predict_reg(model, X, bs=1024):
    model.eval()
    dev = next(model.parameters()).device
    out = [model(torch.as_tensor(X[i:i + bs], dtype=torch.float32, device=dev)).cpu().numpy()
           for i in range(0, len(X), bs)]
    return np.concatenate(out)


# -------------------------------------------------------------------- LOBO
def _fold(X, meta, b):
    """Train/test split for held-out bearing b, standardised with TRAINING bearings only."""
    te = (meta["bearing"] == b).to_numpy()
    tr = ~te
    mu = X[tr].mean(axis=(0, 1), keepdims=True)
    sd = X[tr].std(axis=(0, 1), keepdims=True) + 1e-6
    return tr, te, (X[tr] - mu) / sd, (X[te] - mu) / sd


def run_femto_models(X, y, meta, models=("GRU", "TCN", "Transformer"), seeds=(1,),
                     epochs=15, w_post=4.0, bearings=None, verbose=True):
    """Leave-one-bearing-out. Features are standardised with the TRAINING bearings'
    mean/std only. Returns a long table: seed, model, bearing, MAE_post, MAE_all, params,
    final training loss, training time. The commit hash is stored in df.attrs["commit"]."""
    post_all = meta["post_fpt"].to_numpy().astype(bool)
    names = bearings or sorted(meta["bearing"].unique())
    device = get_device()
    rows = []
    for seed in seeds:
        for b in names:
            tr, te, Xtr, Xte = _fold(X, meta, b)
            w = np.where(post_all[tr], w_post, 1.0)
            for name in models:
                set_seed(seed)                      # seeds the initial weights
                model = MODELS[name]()
                _sync(device)
                t0 = time.perf_counter()
                model, hist = train_regressor(model, Xtr, y[tr], w, epochs=epochs, seed=seed,
                                              device=device, return_history=True)
                _sync(device)
                train_s = time.perf_counter() - t0
                p = predict_reg(model, Xte)
                err = np.abs(p - y[te])
                rows.append({"seed": seed, "model": name, "bearing": b,
                             "MAE_post": float(err[post_all[te]].mean()), "MAE_all": float(err.mean()),
                             "n_post": int(post_all[te].sum()), "params": count_params(model),
                             "final_train_loss": hist[-1], "train_s": train_s})
            if verbose:
                print(f"seed {seed}  held-out {b} done", flush=True)
    df = pd.DataFrame(rows)
    df.attrs["commit"] = git_commit()
    return df


# ---------------------------------------------------------- reproducibility
def repro_check(X, y, meta, model_name="GRU-L", bearing="Bearing1_1", seed=1, epochs=2, w_post=4.0):
    """Train the same model on the same fold twice with the same seed and compare the
    per-epoch training losses. Identical (or differences around 1e-5 or smaller on a GPU)
    means the seeding is complete; clearly different means something is not seeded."""
    post_all = meta["post_fpt"].to_numpy().astype(bool)
    tr, _, Xtr, _ = _fold(X, meta, bearing)
    w = np.where(post_all[tr], w_post, 1.0)
    runs = []
    for _ in range(2):
        set_seed(seed)
        model = MODELS[model_name]()
        _, hist = train_regressor(model, Xtr, y[tr], w, epochs=epochs, seed=seed, return_history=True)
        runs.append(hist)
    max_diff = max(abs(a - b) for a, b in zip(*runs))
    return runs[0], runs[1], max_diff
