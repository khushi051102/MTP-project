"""Phase 1: windowing, time-domain features, spectrum and envelope analysis.

Every function takes a 1-D signal at a KNOWN sampling rate `fs` (12 kHz after
data_loader.load_cwru_file). Nothing here knows about files or labels.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.signal.windows import hann

WIN_LEN = 2048   # samples per window: 0.17 s at 12 kHz (~5 shaft turns, ~29 BPFI impulses)
HOP = 1024       # 50 % overlap


# ----------------------------------------------------------------- windowing
def make_windows(sig: np.ndarray, win_len: int = WIN_LEN, hop: int = HOP) -> np.ndarray:
    """Cut a 1-D signal into overlapping windows -> array (n_windows, win_len).
    The incomplete tail is dropped, so no window is zero-padded."""
    if sig.ndim != 1:
        raise ValueError("sig must be 1-D")
    if sig.size < win_len:
        raise ValueError("signal shorter than one window")
    view = np.lib.stride_tricks.sliding_window_view(sig, win_len)
    return view[::hop].copy()


# ------------------------------------------------------- time-domain features
def time_features(windows: np.ndarray) -> pd.DataFrame:
    """One row of scalar features per window.

    kurtosis uses Pearson's definition (a Gaussian signal = 3), which is why
    healthy vibration sits near 3 and impulsive faults rise well above it.
    """
    w = np.asarray(windows, dtype=np.float64)
    rms = np.sqrt(np.mean(w ** 2, axis=1))
    peak = np.max(np.abs(w), axis=1)
    mean_abs = np.mean(np.abs(w), axis=1)
    # "t_" prefix: names like kurtosis/std/mean would collide with DataFrame
    # methods, so df.kurtosis would silently return a method, not the column.
    return pd.DataFrame({
        "t_mean": w.mean(axis=1),
        "t_std": w.std(axis=1),
        "t_rms": rms,
        "t_peak": peak,
        "t_p2p": w.max(axis=1) - w.min(axis=1),
        "t_skew": stats.skew(w, axis=1),
        "t_kurtosis": stats.kurtosis(w, axis=1, fisher=False),
        "t_crest": peak / rms,               # peak / RMS
        "t_shape": rms / mean_abs,           # RMS / mean |x|
        "t_impulse": peak / mean_abs,        # peak / mean |x|
    })


# ------------------------------------------------------------------ spectrum
def amplitude_spectrum(sig: np.ndarray, fs: float):
    """One-sided amplitude spectrum with a Hann window.

    Hann tapers the window edges to zero, which cuts spectral leakage (a pure
    tone would otherwise smear across bins because the window cuts the wave
    mid-cycle). Dividing by sum(window) keeps amplitudes in signal units.
    """
    x = sig - sig.mean()
    w = hann(x.size, sym=False)
    mag = np.abs(np.fft.rfft(x * w)) * 2.0 / w.sum()
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    return freqs, mag


# ------------------------------------------------------------------ envelope
def envelope_spectrum(sig: np.ndarray, fs: float, band=(2000.0, 5500.0), order: int = 4):
    """Envelope spectrum: band-pass -> Hilbert envelope -> remove mean -> FFT.

    Why: a bearing defect hits once per pass and each impact rings a high
    frequency structural resonance. The repetition rate (BPFI/BPFO/...) shows
    up as the AMPLITUDE MODULATION of that ringing, not as energy at the
    defect frequency itself, so we demodulate first.
    `band` must lie below fs/2; it is an assumption you must log and test.
    """
    low, high = band
    if not (0 < low < high < fs / 2):
        raise ValueError(f"band {band} must satisfy 0 < low < high < fs/2 = {fs/2}")
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, sig - sig.mean())      # zero-phase: no time shift
    env = np.abs(hilbert(filtered))
    env = env - env.mean()
    w = hann(env.size, sym=False)
    mag = np.abs(np.fft.rfft(env * w)) * 2.0 / w.sum()
    freqs = np.fft.rfftfreq(env.size, d=1.0 / fs)
    return freqs, mag


# -------------------------------------------------------------- peak finding
def peak_to_floor(freqs, mag, f0: float, tol_frac: float = 0.03,
                  floor_range=(10.0, 500.0)) -> float:
    """Height of the largest peak within +/- tol_frac of f0, divided by the
    median level in `floor_range`. Values near 1 mean 'nothing there'.

    tol_frac exists because CWRU speed was recorded by hand, so the true
    defect frequency can differ from rpm/60 x multiplier by a few percent.
    """
    lo, hi = f0 * (1 - tol_frac), f0 * (1 + tol_frac)
    band = (freqs >= lo) & (freqs <= hi)
    floor = np.median(mag[(freqs >= floor_range[0]) & (freqs <= floor_range[1])])
    return float(mag[band].max() / floor)


# --------------------------------------------------------------------- STFT
def spectrogram_db(sig: np.ndarray, fs: float, nperseg: int = 256, overlap: float = 0.75):
    """Short-time Fourier transform magnitude in dB -> (freqs, times, S_db).

    Trade-off (time-frequency uncertainty): a short segment locates impacts in
    time but blurs frequency (resolution = fs/nperseg); a long segment does the
    opposite. Hann window per segment, same reason as in amplitude_spectrum.
    """
    from scipy.signal import spectrogram
    f, t, S = spectrogram(sig - sig.mean(), fs=fs, window="hann", nperseg=nperseg,
                          noverlap=int(nperseg * overlap), mode="magnitude")
    return f, t, 20 * np.log10(S + 1e-12)
