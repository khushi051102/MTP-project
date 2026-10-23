"""Phase 2: build the windowed CWRU dataset and split it WITHOUT leakage.

KEY FACT: every (class, load) pair in CWRU is exactly ONE recording. A random
window-level split therefore puts neighbouring (even overlapping) windows of the
same recording in both train and test, and the model is graded on data it has
effectively seen. The only honest split unit is the recording, which here means
the motor LOAD (0/1/2/3 HP).
"""
import numpy as np
import pandas as pd

from .config import CWRU_RAW, DATA_PROCESSED
from .cwru_meta import build_metadata
from .data_loader import load_cwru_file
from .signal_proc import HOP, WIN_LEN, make_windows


def _label(fault_type, diameter):
    return "normal" if fault_type == "normal" else f"{fault_type}_{diameter:.3f}"


def build_cwru_windows(raw_root=CWRU_RAW, win_len=WIN_LEN, hop=HOP):
    """Returns (X, info). X: float32 (n_windows, win_len) at 12 kHz.
    info: one row per window with file_id, fault_type, diameter_in, load_hp,
    label (10-class string), y (int class id), win_idx."""
    meta = build_metadata()
    X_parts, info_parts = [], []
    for r in meta.itertuples():
        rec = load_cwru_file(raw_root / r.relpath, int(r.fs_native))   # -> 12 kHz
        w = make_windows(rec.signal, win_len, hop).astype(np.float32)
        X_parts.append(w)
        info_parts.append(pd.DataFrame({
            "file_id": r.file_id,
            "fault_type": r.fault_type,
            "diameter_in": r.diameter_in,
            "load_hp": r.load_hp,
            "label": _label(r.fault_type, r.diameter_in),
            "win_idx": np.arange(len(w)),
        }))
    X = np.concatenate(X_parts)
    info = pd.concat(info_parts, ignore_index=True)
    classes = sorted(info["label"].unique())
    info["y"] = info["label"].map({c: i for i, c in enumerate(classes)})
    return X, info


def save_windows(X, info, path=None):
    path = path or (DATA_PROCESSED / "cwru_windows.npz")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X)
    info.to_csv(str(path).replace(".npz", "_info.csv"), index=False)
    return path


def load_windows(path=None):
    path = path or (DATA_PROCESSED / "cwru_windows.npz")
    X = np.load(path)["X"]
    info = pd.read_csv(str(path).replace(".npz", "_info.csv"))
    return X, info


def split_by_load(info, test_load, val_load=None):
    """Leave-one-load-out split. Returns integer index arrays (train, val, test).
    Whole recordings go to one side only; asserts prove it."""
    loads = set(info["load_hp"].unique())
    test_loads = {test_load}
    val_loads = {val_load} if val_load is not None else set()
    if test_loads & val_loads:
        raise ValueError("val_load must differ from test_load")
    train_loads = loads - test_loads - val_loads

    def idx(ls):
        return np.flatnonzero(info["load_hp"].isin(ls).to_numpy())

    train, val, test = idx(train_loads), idx(val_loads), idx(test_loads)
    files = lambda i: set(info.iloc[i]["file_id"])
    assert not (files(train) & files(test)), "LEAK: a file is in train and test"
    assert not (files(train) & files(val)), "LEAK: a file is in train and val"
    assert not (files(val) & files(test)), "LEAK: a file is in val and test"
    return train, val, test


def class_counts(info, idx):
    return info.iloc[idx]["label"].value_counts().sort_index()
