"""Load one CWRU .mat file as a clean 1-D drive-end signal at a common rate."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.io import loadmat
from scipy.signal import decimate

from .config import TARGET_FS


@dataclass
class Recording:
    signal: np.ndarray        # 1-D, float64, sampled at `fs`
    fs: int                   # sampling rate of `signal` (after any decimation)
    rpm: Optional[float]      # measured shaft speed stored inside the file
    n_native: int             # sample count BEFORE decimation


def _find_key(d: dict, token: str, required: bool = True,
              prefix: Optional[str] = None) -> Optional[str]:
    """Return the single key containing `token` (e.g. 'DE_time', 'RPM').

    CWRU keys carry a file number ('X097_DE_time'), so we cannot hard-code them.
    Rule:
      1. exactly one key contains `token`  -> return it;
      2. several keys contain `token`      -> if `prefix` (e.g. 'X098', the
         file's own number) is given, keep only keys starting with it, and
         return the key only if exactly one survives;
      3. anything else (zero when required, or still several) -> raise.
    Some CWRU files hold signals of more than one recording, so the file's own
    number is the only safe way to pick the right one. We never guess.
    """
    matches = [k for k in d if token in k and not k.startswith("__")]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        if required:
            raise KeyError(f"no key containing '{token}'; keys: {list(d)}")
        return None

    # several candidates: use the file's own number to disambiguate
    if prefix is not None:
        narrowed = [k for k in matches if k.startswith(prefix)]
        if len(narrowed) == 1:
            return narrowed[0]
        raise KeyError(
            f"several keys contain '{token}': {matches}; after filtering by "
            f"prefix '{prefix}' {len(narrowed)} remain: {narrowed}"
        )
    raise KeyError(f"several keys contain '{token}': {matches} (no prefix given)")


def load_cwru_file(path, fs_native: int, target_fs: int = TARGET_FS) -> Recording:
    path = Path(path)
    if not path.stem.isdigit():
        raise ValueError(f"expected a numeric file name like '98.mat', got '{path.name}'")
    prefix = f"X{int(path.stem):03d}"        # 98 -> 'X098', matches the keys inside the file

    d = loadmat(str(path))

    key = _find_key(d, "DE_time", prefix=prefix)
    sig = np.asarray(d[key], dtype=np.float64).ravel()   # (N,1) column -> (N,)
    n_native = sig.size

    rpm_key = _find_key(d, "RPM", required=False, prefix=prefix)
    rpm = float(np.asarray(d[rpm_key]).ravel()[0]) if rpm_key else None

    if fs_native != target_fs:
        if fs_native % target_fs != 0:
            raise ValueError(f"{fs_native} is not an integer multiple of {target_fs}")
        # decimate = low-pass (anti-alias) filter THEN keep every q-th sample.
        # Plain sig[::q] would fold frequencies above target_fs/2 into the band.
        sig = decimate(sig, fs_native // target_fs, zero_phase=True)

    return Recording(signal=sig, fs=target_fs, rpm=rpm, n_native=n_native)