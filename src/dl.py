"""Phase 4: small PyTorch models, a shared training loop and cost measurements.

Works on CPU or GPU (Colab). Everything that decides the result (seeds, epochs,
scaling) is an explicit argument so it can be written down in the assumptions log.
"""
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from .config import SEED
from .dataset import split_by_load
from .utils import set_seed


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ models
class CNN1D(nn.Module):
    """Small 1D CNN for raw vibration windows, input (batch, length) or (batch, 1, length).

    conv -> batch-norm -> ReLU -> max-pool, four times, then global average pooling
    and one linear layer. The first kernel is wide (64 samples = 5 ms at 12 kHz, about
    one fault-impact period) with stride 8, so it acts like a learned filter bank.
    """

    def __init__(self, n_classes: int = 10, width: int = 16):
        super().__init__()
        w = width

        def block(cin, cout, k, stride=1, pool=True):
            layers = [nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2, bias=False),
                      nn.BatchNorm1d(cout), nn.ReLU()]
            if pool:
                layers.append(nn.MaxPool1d(2))
            return layers

        self.features = nn.Sequential(*block(1, w, 64, stride=8), *block(w, 2 * w, 3),
                                      *block(2 * w, 4 * w, 3), *block(4 * w, 4 * w, 3, pool=False))
        self.head = nn.Linear(4 * w, n_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        z = self.features(x)
        z = z.mean(dim=-1)          # global average pooling over time
        return self.head(z)


# ---------------------------------------------------------------- training
def train_classifier(model, X_tr, y_tr, epochs=20, lr=2e-3, wd=1e-2, bs=128,
                     seed=SEED, device=None, verbose=True):
    """Fixed number of epochs, AdamW + one-cycle learning-rate schedule.
    No early stopping on the test fold: the epoch count is chosen in advance."""
    device = device or get_device()
    set_seed(seed)                                  # controls the order of the batches
    torch.backends.cudnn.deterministic = True       # same results on repeated GPU runs
    torch.backends.cudnn.benchmark = False
    model = model.to(device)
    ds = TensorDataset(torch.as_tensor(X_tr, dtype=torch.float32), torch.as_tensor(y_tr, dtype=torch.long))
    dl = DataLoader(ds, batch_size=bs, shuffle=True, generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(dl))
    lossf = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(xb)
        if verbose and (ep % 5 == 4 or ep == epochs - 1):
            print(f"  epoch {ep + 1:3d}  train loss {total / len(ds):.4f}", flush=True)
    return model


@torch.no_grad()
def predict(model, X, bs=512, device=None):
    device = device or next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.as_tensor(X[i:i + bs], dtype=torch.float32, device=device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


# ------------------------------------------------------------------- costs
def count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def model_size_kb(model) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024


@torch.no_grad()
def latency_ms(model, example, n_warmup=20, n_runs=200, threads=1):
    """Median and 95th-percentile time to score ONE input on the CPU with `threads` threads.
    Only numbers measured on your own laptop belong in the cost table, not Colab's."""
    prev = torch.get_num_threads()
    torch.set_num_threads(threads)
    m = model.to("cpu").eval()
    x = torch.as_tensor(example[:1], dtype=torch.float32)
    for _ in range(n_warmup):
        m(x)
    times = []
    for _ in range(n_runs):
        t = time.perf_counter()
        m(x)
        times.append((time.perf_counter() - t) * 1e3)
    torch.set_num_threads(prev)
    return float(np.median(times)), float(np.percentile(times, 95))


# ------------------------------------------------------------------- CWRU
def run_cwru_cnn(X, info, epochs=20, width=16, seed=SEED, verbose=True):
    """Leave-one-load-out CNN. The input is divided by ONE scalar (the training
    windows' standard deviation), not standardised per window, because amplitude
    (RMS) is one of the strongest class cues. Returns (per-fold table, last model)."""
    y = pd.factorize(info["label"], sort=True)[0]
    rows, model = [], None
    for test_load in range(4):
        tr, _, te = split_by_load(info, test_load, None)
        scale = float(X[tr].std())
        set_seed(seed)                                   # seeds the INITIAL WEIGHTS (created on the next line)
        model = CNN1D(n_classes=int(y.max()) + 1, width=width)
        if verbose:
            print(f"fold: test load {test_load} HP  (train {len(tr)}, test {len(te)})", flush=True)
        train_classifier(model, X[tr] / scale, y[tr], epochs=epochs, seed=seed, verbose=verbose)
        pred = predict(model, X[te] / scale).argmax(axis=1)
        rows.append({"test_load": test_load, "accuracy": accuracy_score(y[te], pred),
                     "macro_F1": f1_score(y[te], pred, average="macro")})
    return pd.DataFrame(rows), model
