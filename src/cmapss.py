"""C-MAPSS turbofan data: loading, RUL labels, condition-wise scaling, windows.

Files (data/raw/cmapss): train_FD00x.txt, test_FD00x.txt, RUL_FD00x.txt.
Columns: unit, cycle, 3 operating settings, 21 sensors (26 in total).
FD001/FD003: one operating condition. FD002/FD004: six conditions.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .config import DATA_RAW, SEED

OP = ["op1", "op2", "op3"]
SENSORS = [f"s{i}" for i in range(1, 22)]
COLS = ["unit", "cycle"] + OP + SENSORS
N_CONDITIONS = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}   # from the dataset readme
MIN_UNIQUE = 5   # a sensor with fewer distinct values is a flickering flag, not a signal


def load_cmapss(fd: str, root=None):
    """Returns (train_df, test_df, rul_end). rul_end[i] is the TRUE remaining life
    of test engine i+1 at the last cycle we observe for it."""
    root = root or (DATA_RAW / "cmapss")
    train = pd.read_csv(root / f"train_{fd}.txt", sep=r"\s+", header=None, names=COLS)
    test = pd.read_csv(root / f"test_{fd}.txt", sep=r"\s+", header=None, names=COLS)
    rul_end = pd.read_csv(root / f"RUL_{fd}.txt", sep=r"\s+", header=None)[0].to_numpy()
    if test["unit"].nunique() != len(rul_end):
        raise ValueError("test engines and RUL rows do not match")
    return train, test, rul_end


def add_rul(train: pd.DataFrame, cap: int = 125) -> pd.DataFrame:
    """RUL label for training data: cycles left until the engine's last cycle,
    clipped at `cap`. Why clip: early in life an engine is healthy and its
    sensors carry no information about WHEN it will fail, so asking the model
    to tell 300 from 200 cycles left only teaches it noise."""
    df = train.copy()
    last = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = (last - df["cycle"]).clip(upper=cap)
    return df


class ConditionScaler:
    """Standardise each sensor WITHIN its operating condition.

    In FD002/FD004 the sensors swing mostly because the engine changes altitude
    and speed, not because it degrades. Scaling each condition separately
    removes that swing so what remains is degradation. Fit on TRAIN only.
    """

    def __init__(self, n_conditions: int):
        self.n_conditions = n_conditions

    def fit(self, train: pd.DataFrame):
        if self.n_conditions > 1:
            self.km_ = KMeans(self.n_conditions, n_init=10, random_state=SEED).fit(train[OP])
        cond = self._cond(train)
        g = train[SENSORS].groupby(cond)
        self.mean_, self.std_ = g.mean(), g.std(ddof=0)
        # sensors that stay (almost) constant after scaling carry no information
        scaled = self._scale(train, cond)
        self.keep_ = [s for s in SENSORS if scaled[s].std() > 1e-6 and train[s].nunique() >= MIN_UNIQUE]
        return self

    def _cond(self, df):
        if self.n_conditions == 1:
            return pd.Series(0, index=df.index)
        return pd.Series(self.km_.predict(df[OP]), index=df.index)

    def _scale(self, df, cond):
        out = df.copy()
        mean = self.mean_.loc[cond].set_index(df.index)
        std = self.std_.loc[cond].set_index(df.index).replace(0, 1.0)
        out[SENSORS] = (df[SENSORS] - mean) / std
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._scale(df, self._cond(df))


def last_cycle_rows(test: pd.DataFrame, rul_end, cap=None):
    """One row per test engine (its last observed cycle) and the true RUL there.
    cap=None keeps the official values; cap=125 matches the clipped training
    target. Decide which you report and write it down."""
    last = test.sort_values(["unit", "cycle"]).groupby("unit").tail(1).reset_index(drop=True)
    y = np.asarray(rul_end, float)
    return last, (np.minimum(y, cap) if cap else y)


def make_sequences(df: pd.DataFrame, features, seq_len: int = 30, label=True, last_only=False):
    """Sliding windows (n, seq_len, n_features) per engine. Windows never cross
    engines. Engines shorter than seq_len are padded at the start by repeating
    their first row. label=True also returns the RUL at each window's end."""
    Xs, ys, units = [], [], []
    for unit, g in df.sort_values(["unit", "cycle"]).groupby("unit"):
        f = g[features].to_numpy(dtype=np.float32)
        if len(f) < seq_len:
            f = np.vstack([np.repeat(f[:1], seq_len - len(f), axis=0), f])
            r = np.concatenate([np.full(seq_len - len(g), np.nan), g["rul"].to_numpy()]) if label else None
        else:
            r = g["rul"].to_numpy() if label else None
        ends = [len(f) - 1] if last_only else range(seq_len - 1, len(f))
        for e in ends:
            Xs.append(f[e - seq_len + 1:e + 1])
            units.append(unit)
            if label:
                ys.append(r[e])
    X = np.stack(Xs)
    return (X, np.asarray(ys, np.float32), np.asarray(units)) if label else (X, np.asarray(units))
