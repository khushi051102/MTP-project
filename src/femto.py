"""FEMTO / PRONOSTIA bearings (IEEE PHM 2012): loading and health-indicator features.

Layout (data/raw/femto): training_set/BearingX_Y/acc_00001.csv ... (also test_set,
validation_set). Each acc file is one snapshot: ~2560 rows of
    hour, minute, second, microsecond, horizontal_accel, vertical_accel
(comma separated, no header), recorded at 25.6 kHz every ~10 s.
Separators are detected per file in case some bearings use ';'.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import DATA_PROCESSED, DATA_RAW

FS = 25_600          # Hz, to be CHECKED against the timestamps with check_fs()


def _read_acc(path) -> np.ndarray:
    with open(path) as f:
        first = f.readline()
    sep = ";" if ";" in first else ","
    return pd.read_csv(path, header=None, sep=sep).to_numpy(dtype=np.float64)


def check_fs(path) -> dict:
    """Infer the sampling rate from the TOTAL time span of one snapshot.

    Do not use the median of consecutive timestamp differences: the stamps are
    rounded to whole microseconds (and to 5 significant digits in some files),
    so single differences are too coarse; the span over ~2560 rows is not.
    """
    a = _read_acc(path)
    t = a[:, 0] * 3600 + a[:, 1] * 60 + a[:, 2] + a[:, 3] * 1e-6
    span = float(t[-1] - t[0])
    return {"rows": len(a), "span_s": round(span, 6),
            "implied_fs_hz": round((len(a) - 1) / span, 1)}


def bearing_features(folder) -> pd.DataFrame:
    """One row per snapshot: time since start (s) and vibration indicators per axis."""
    files = sorted(Path(folder).glob("acc_*.csv"))
    if not files:
        raise FileNotFoundError(f"no acc_*.csv in {folder}")
    rows = []
    for i, f in enumerate(files):
        a = _read_acc(f)
        t0 = a[0, 0] * 3600 + a[0, 1] * 60 + a[0, 2] + a[0, 3] * 1e-6
        row = {"idx": i, "t0": t0, "n_rows": len(a)}
        for name, x in (("h", a[:, 4]), ("v", a[:, 5])):
            x = x - x.mean()
            rms = float(np.sqrt(np.mean(x ** 2)))
            row[f"rms_{name}"] = rms
            row[f"kurt_{name}"] = float(stats.kurtosis(x, fisher=False))
            row[f"peak_{name}"] = float(np.max(np.abs(x)))
            row[f"crest_{name}"] = float(np.max(np.abs(x)) / rms)
        rows.append(row)
    df = pd.DataFrame(rows)
    t = np.unwrap(df["t0"].to_numpy(), period=86400.0)     # clock wraps at midnight
    df["t"] = t - t[0]
    return df.drop(columns="t0")


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Run-to-failure bearings only: RUL (seconds) = time of last snapshot - t.
    ASSUMPTION: the last snapshot of a training bearing is its failure time."""
    out = df.copy()
    out["rul_s"] = out["t"].iloc[-1] - out["t"]
    return out


def first_predicting_time(df: pd.DataFrame, col: str = "rms_h", ref=(100, 400),
                          k: float = 3.0, run: int = 5, smooth: int = 1):
    """Index of the first snapshot after `ref` where `col` exceeds
    mean + k*std of the healthy reference window ref=(start, stop) for `run`
    consecutive snapshots. Returns None if it never happens.

    The reference window should come AFTER the run-in period (many bearings are
    noisier in the first minutes) and BEFORE any degradation; choose it by
    looking at the plot. `smooth` > 1 applies a rolling median first, which
    removes isolated spikes. ref, k, run and smooth are all assumptions.
    """
    x = pd.Series(df[col].to_numpy())
    if smooth > 1:
        x = x.rolling(smooth, min_periods=1).median()
    x = x.to_numpy()
    r0, r1 = ref
    base = x[r0:r1]
    thr = base.mean() + k * base.std()
    above = x > thr
    for i in range(r1, len(x) - run + 1):
        if above[i:i + run].all():
            return int(i)
    return None


def build_femto_features(root=None, sets=("training_set",), save=True) -> pd.DataFrame:
    """Features for every bearing in the given sets, cached to data/processed."""
    root = Path(root or (DATA_RAW / "femto"))
    parts = []
    for s in sets:
        for folder in sorted((root / s).glob("Bearing*")):
            df = bearing_features(folder)
            df.insert(0, "bearing", folder.name)
            df.insert(0, "set", s)
            parts.append(df)
            print(f"{s}/{folder.name}: {len(df)} snapshots", flush=True)
    out = pd.concat(parts, ignore_index=True)
    if save:
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        out.to_csv(DATA_PROCESSED / "femto_features.csv", index=False)
    return out
