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
def top_half_energy(sig: np.ndarray) -> float:
    """Energy fraction above per-sample frequency 0.25 (top half of the band).

    Per-sample frequency p means true frequency p * fs, so p in (0.25, 0.5] is
    12-24 kHz if fs = 48k but only 3-6 kHz if fs = 12k. Bearing/structural
    resonances live at 3-6 kHz, so a REAL 12 kHz recording keeps a large share
    of its energy in that band (105.mat, known 12 kHz: ~40 %), while a 48 kHz
    recording low-passed near 12 kHz by the acquisition filter shows a hard
    cliff there (<1 %).

    Replaces the old shaft-comb test, which could not discriminate: a 30-120 Hz
    comb sits far below both Nyquist limits, so it lights up under EITHER
    assumed fs and the verdict degenerated to noise-skirt comparison.
    """
    x = sig - sig.mean()
    S = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    ps = np.fft.rfftfreq(x.size)          # cycles per SAMPLE, fs-independent
    return float(S[ps > 0.25].sum() / S.sum())


def sampling_rate_test(sig: np.ndarray, rpm: float, candidates=(48_000, 12_000),
                       hi: float = 0.05, lo: float = 0.01):
    """Decide between candidate sampling rates from the signal itself.

    Returns (E_top from top_half_energy, verdict). Calibrated on two ground
    truths: 105.mat (known 12 kHz) E_top = 0.40 -> '12000'; official normal
    files 97-100 (48 kHz per the CWRU download page) E_top < 0.01 -> '48000'.
    `rpm` is kept for call compatibility; the test no longer needs it.
    Limitation: a hypothetical 12 kHz recording with no energy above 3 kHz
    would read '48000'; audit_cwru's duration check cross-guards that case.
    """
    e = top_half_energy(sig)
    if e >= hi:
        verdict = f"looks like {candidates[1]} Hz"
    elif e <= lo:
        verdict = f"looks like {candidates[0]} Hz"
    else:
        verdict = "INCONCLUSIVE"
    return e, verdict


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
        e_top, verdict = sampling_rate_test(rec.signal, rpm)
        tag = f"(known {known_fs} Hz)" if known_fs else "(unknown)"
        print(f"{rel:22s} {tag:18s} E(top half)={e_top:.3f}  -> {verdict}")

    if save:
        RESULTS.mkdir(exist_ok=True)
        meta.to_csv(RESULTS / "cwru_verified.csv", index=False)
    return meta


if __name__ == "__main__":
    run_verify()
