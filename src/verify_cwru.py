"""Verify metadata against the official CWRU tables and test the sampling rate.

Run from the project root:   python -m src.verify_cwru
"""
import numpy as np
import pandas as pd

from .config import CWRU_RAW, RESULTS
from .cwru_meta import build_metadata
from .cwru_reference import CLASS_TO_FREQ, MULT, NOMINAL_RPM, REF_FILES
from .data_loader import load_cwru_file


# ---------------------------------------------------------------- part 1
def verify_labels(meta: pd.DataFrame) -> pd.DataFrame:
    """Compare every metadata row with the official table. Returns the mismatches."""
    bad = []
    for r in meta.itertuples():
        key = (r.fault_type, None if pd.isna(r.diameter_in) else r.diameter_in)
        official = REF_FILES.get(key)
        if official is None:
            bad.append((r.relpath, "class/diameter not in official table"))
            continue
        if official[r.load_hp] != r.file_id:
            bad.append((r.relpath, f"official file for {key}, {r.load_hp} HP is {official[r.load_hp]}"))
        if NOMINAL_RPM[r.load_hp] != r.nominal_rpm:
            bad.append((r.relpath, f"nominal rpm should be {NOMINAL_RPM[r.load_hp]}"))
    return pd.DataFrame(bad, columns=["file", "problem"])


# ---------------------------------------------------------------- part 2
def expected_fault_freq(fault_type: str, rpm: float):
    """Expected defect frequency (Hz) for a class at a given shaft speed."""
    name = CLASS_TO_FREQ.get(fault_type)
    return None if name is None else MULT[name] * rpm / 60.0


# ---------------------------------------------------------------- part 3
def harmonic_score(sig: np.ndarray, fs: float, shaft_hz: float,
                   n_harm: int = 4, half_bw: float = 1.5) -> float:
    """How strongly does the spectrum show peaks at 1X..nX shaft speed IF the
    sampling rate were `fs`?  Peak height is measured relative to the median
    spectral level in 5-300 Hz, so a value near 1 means 'no visible peak'."""
    x = sig - sig.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    floor = np.median(spec[(freqs > 5) & (freqs < 300)])
    ratios = []
    for h in range(1, n_harm + 1):
        m = (freqs > h * shaft_hz - half_bw) & (freqs < h * shaft_hz + half_bw)
        ratios.append(spec[m].max() / floor)
    return float(np.mean(ratios))


def sampling_rate_test(sig: np.ndarray, rpm: float, candidates=(48_000, 12_000)):
    """Score each candidate sampling rate; the true one should place the shaft
    harmonics at rpm/60 Hz. Returns (scores dict, verdict string)."""
    shaft_hz = rpm / 60.0
    scores = {fs: harmonic_score(sig, fs, shaft_hz) for fs in candidates}
    a, b = candidates
    if scores[a] > 2 * scores[b]:
        verdict = f"looks like {a} Hz"
    elif scores[b] > 2 * scores[a]:
        verdict = f"looks like {b} Hz"
    else:
        verdict = "INCONCLUSIVE"
    return scores, verdict


def run_verify(raw_root=CWRU_RAW, save=True) -> pd.DataFrame:
    meta = build_metadata()

    print("== 1. file numbers / rpm vs official tables ==")
    bad = verify_labels(meta)
    print("all 40 rows match the official tables" if bad.empty else bad.to_string(index=False))

    print("\n== 2. expected defect frequencies (official multipliers x measured rpm/60) ==")
    shaft, expected = [], []
    for r in meta.itertuples():
        path = raw_root / r.relpath
        if not path.exists():
            shaft.append(np.nan); expected.append(np.nan); continue
        rec = load_cwru_file(path, r.fs_native, target_fs=r.fs_native)  # no decimation
        rpm = rec.rpm if rec.rpm is not None else r.nominal_rpm
        shaft.append(rpm / 60.0)
        f = expected_fault_freq(r.fault_type, rpm)
        expected.append(np.nan if f is None else f)
    meta["shaft_hz"] = shaft
    meta["expected_fault_hz"] = expected
    print(meta[["relpath", "load_hp", "shaft_hz", "expected_fault_hz"]].round(2).to_string(index=False))

    print("\n== 3. sampling-rate test (positive control = a 12 kHz fault file) ==")
    tests = [("inner_race/105.mat", 12_000)] + [(f"normal/{n}.mat", None) for n in (97, 98, 99, 100)]
    for rel, known_fs in tests:
        path = raw_root / rel
        if not path.exists():
            print(rel, "missing"); continue
        row = meta[meta["relpath"] == rel].iloc[0]
        rec = load_cwru_file(path, row.fs_native, target_fs=row.fs_native)
        rpm = rec.rpm if rec.rpm is not None else row.nominal_rpm
        scores, verdict = sampling_rate_test(rec.signal, rpm)
        tag = f"(known {known_fs} Hz)" if known_fs else "(unknown)"
        print(f"{rel:22s} {tag:18s} score@48k={scores[48000]:.1f}  score@12k={scores[12000]:.1f}  -> {verdict}")

    if save:
        RESULTS.mkdir(exist_ok=True)
        meta.to_csv(RESULTS / "cwru_verified.csv", index=False)
    return meta


if __name__ == "__main__":
    run_verify()
