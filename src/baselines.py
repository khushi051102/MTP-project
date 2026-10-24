"""Phase 3: classical-ML baselines.

CWRU (classification):  features -> LogReg / Tree / RandomForest / (XGBoost)
    evaluated two ways: a random window split (LEAKY) and leave-one-load-out (HONEST).
C-MAPSS (RUL regression): scaled sensors + rolling stats -> LinReg / Tree / RF / (XGBoost)
    evaluated on the official test set (last cycle of each test engine).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .cmapss import (N_CONDITIONS, ConditionScaler, add_rul, load_cmapss)
from .config import SEED
from .dataset import split_by_load
from .metrics import mae, nasa_score, rmse
from .signal_proc import time_features

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:          # pip install xgboost
    XGBClassifier = XGBRegressor = None


# ============================================================ CWRU features
def cwru_features(X: np.ndarray, fs: int = 12_000, n_bands: int = 16) -> pd.DataFrame:
    """27 features per window: 10 time-domain + 16 log band energies + spectral centroid.
    Band energies split 0..fs/2 into equal slices; the log compresses the huge
    dynamic range between strong and weak bands so linear models can use it."""
    X = np.asarray(X, dtype=np.float64)
    tf = time_features(X)
    Xc = X - X.mean(axis=1, keepdims=True)
    win = np.hanning(X.shape[1])
    mag = np.abs(np.fft.rfft(Xc * win, axis=1)) * 2.0 / win.sum()
    freqs = np.fft.rfftfreq(X.shape[1], d=1.0 / fs)
    edges = np.linspace(0, len(freqs), n_bands + 1).astype(int)
    bands = np.stack([np.log10((mag[:, a:b] ** 2).sum(axis=1) + 1e-12)
                      for a, b in zip(edges[:-1], edges[1:])], axis=1)
    centroid = (mag * freqs).sum(axis=1) / (mag.sum(axis=1) + 1e-12)
    out = tf.copy()
    for i in range(n_bands):
        out[f"band_{i:02d}"] = bands[:, i]
    out["centroid"] = centroid
    return out


def _cls_models(seed):
    m = {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed),
    }
    if XGBClassifier is not None:
        m["XGBoost"] = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                     n_jobs=-1, random_state=seed, verbosity=0)
    return m


def run_cwru_baselines(F: pd.DataFrame, info: pd.DataFrame, seed: int = SEED, n_random: int = 4):
    """Returns a table: model x protocol -> accuracy / macro-F1 (mean +- std).
    y is re-coded to 0..K-1 because XGBoost requires it."""
    y = pd.factorize(info["label"], sort=True)[0]
    Xv = F.to_numpy()
    rows = []
    for name in _cls_models(seed):
        accs, f1s = [], []
        sss = StratifiedShuffleSplit(n_splits=n_random, test_size=0.3, random_state=seed)
        for tr, te in sss.split(Xv, y):
            mdl = _cls_models(seed)[name].fit(Xv[tr], y[tr])
            p = mdl.predict(Xv[te])
            accs.append(accuracy_score(y[te], p)); f1s.append(f1_score(y[te], p, average="macro"))
        rows.append((name, "random window split (leaky)", accs, f1s))
        accs, f1s = [], []
        for test_load in range(4):
            tr, _, te = split_by_load(info, test_load, None)
            mdl = _cls_models(seed)[name].fit(Xv[tr], y[tr])
            p = mdl.predict(Xv[te])
            accs.append(accuracy_score(y[te], p)); f1s.append(f1_score(y[te], p, average="macro"))
        rows.append((name, "leave-one-load-out (honest)", accs, f1s))
    return pd.DataFrame({
        "model": [r[0] for r in rows], "protocol": [r[1] for r in rows],
        "accuracy": [f"{np.mean(r[2]):.3f} +- {np.std(r[2]):.3f}" for r in rows],
        "macro_F1": [f"{np.mean(r[3]):.3f} +- {np.std(r[3]):.3f}" for r in rows],
    })


# ========================================================== C-MAPSS features
def cmapss_features(df: pd.DataFrame, feats, window: int = 10) -> pd.DataFrame:
    """Current value + rolling mean + rolling std over the last `window` cycles of
    the SAME engine. The rolling stats give the model a short memory (trend/noise)."""
    d = df.sort_values(["unit", "cycle"])
    g = d.groupby("unit")[feats]
    mean = g.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    std = g.rolling(window, min_periods=2).std().reset_index(level=0, drop=True).fillna(0.0)
    return pd.concat([d[feats], mean.add_suffix("_m"), std.add_suffix("_sd")], axis=1)


def _reg_models(seed):
    m = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=20, random_state=seed),
        "RandomForest": RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                              n_jobs=-1, random_state=seed),
    }
    if XGBRegressor is not None:
        m["XGBoost"] = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                    n_jobs=-1, random_state=seed, verbosity=0)
    return m


def run_cmapss_baselines(fd: str, cap: int = 125, window: int = 10, root=None, seed: int = SEED,
                         scale: str = "condition"):
    train, test, rul_end = load_cmapss(fd, root)
    train = add_rul(train, cap)
    # scale="global" treats all rows as ONE condition (ablation: what does per-condition scaling buy?)
    sc = ConditionScaler(1 if scale == "global" else N_CONDITIONS[fd]).fit(train)   # fit on TRAIN only
    tr_s, te_s = sc.transform(train), sc.transform(test)
    Ftr = cmapss_features(tr_s, sc.keep_, window)
    Fte = cmapss_features(te_s, sc.keep_, window)
    ytr = train.loc[Ftr.index, "rul"].to_numpy()
    last_idx = te_s.sort_values(["unit", "cycle"]).groupby("unit").tail(1).index
    Xte = Fte.loc[last_idx].to_numpy()
    y_true = np.minimum(np.asarray(rul_end, float), cap)       # same cap as training
    rows = []
    for name, mdl in _reg_models(seed).items():
        mdl.fit(Ftr.to_numpy(), ytr)
        pred = np.clip(mdl.predict(Xte), 0, cap)
        rows.append({"dataset": fd, "model": name, "RMSE": round(rmse(y_true, pred), 2),
                     "MAE": round(mae(y_true, pred), 2), "NASA_score": round(nasa_score(y_true, pred), 1)})
    return pd.DataFrame(rows)
