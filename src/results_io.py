"""Save and reload result tables so a new Colab session never has to retrain.

save_results({"allres": allres, ...}, folder)  -> one CSV per table + manifest.json
load_results(folder)                          -> dict of DataFrames, same names
verify_results({"allres": allres, ...}, folder)-> reload from disk and compare with memory

Summary tables often keep labels in the index (e.g. model names after a groupby) and
may have non-text column names (e.g. seeds 1, 2, 3 after unstack). CSV keeps neither,
so `normalise` moves a meaningful index into columns and turns column names into text;
the manifest remembers which columns were the index, and load_results puts it back.
"""
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _has_default_index(df):
    idx = df.index
    return isinstance(idx, pd.RangeIndex) and idx.start == 0 and idx.step == 1 and idx.name is None


def normalise(df):
    """Return (table ready for CSV, list of index column names or None)."""
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = ["|".join(str(p) for p in c if str(p) != "") for c in out.columns]
    out.columns = [str(c) for c in out.columns]
    index_cols = None
    if not _has_default_index(df):
        names = [n if n is not None else f"index_{i}" for i, n in enumerate(out.index.names)]
        out.index.names = names
        out = out.reset_index()
        index_cols = [str(n) for n in names]
        out.columns = [str(c) for c in out.columns]
    return out, index_cols


def save_results(tables, folder):
    """tables: dict name -> DataFrame. Existing files with the same name are overwritten;
    the manifest keeps entries for tables saved earlier under other names."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    man_path = folder / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    commit, now = git_commit(), datetime.now().isoformat(timespec="seconds")
    for name, df in tables.items():
        if isinstance(df, pd.Series):
            df = df.to_frame()
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{name} is {type(df).__name__}, not a DataFrame")
        flat, index_cols = normalise(df)
        flat.to_csv(folder / f"{name}.csv", index=False)
        manifest[name] = {"rows": len(flat), "columns": list(flat.columns), "index": index_cols,
                          "commit": df.attrs.get("commit", commit), "saved": now}
    man_path.write_text(json.dumps(manifest, indent=2))
    print(f"saved {len(tables)} tables to {folder} (commit {commit})")


def load_results(folder, names=None, verbose=True):
    """Load every table listed in the manifest (or only `names`), restore the index,
    and check row counts against the manifest."""
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    out = {}
    for name in names or manifest:
        info = manifest[name]
        df = pd.read_csv(folder / f"{name}.csv")
        assert len(df) == info["rows"], f"{name}: {len(df)} rows, manifest says {info['rows']}"
        if info.get("index"):
            df = df.set_index(info["index"])
        df.attrs["commit"] = info["commit"]
        out[name] = df
        if verbose:
            print(f"{name:20s} rows={len(df):6d}  commit={info['commit']}  saved={info['saved']}")
    return out


def verify_results(tables, folder):
    """Reload each table from disk and compare it with the copy in memory.
    Prints OK or the first difference found. Returns True only if all match."""
    loaded = load_results(folder, names=list(tables), verbose=False)
    all_ok = True
    for name, df in tables.items():
        if isinstance(df, pd.Series):
            df = df.to_frame()
        mem, _ = normalise(df)
        disk, _ = normalise(loaded[name]) if not _has_default_index(loaded[name]) else (loaded[name], None)
        try:
            pd.testing.assert_frame_equal(mem.reset_index(drop=True), disk.reset_index(drop=True),
                                          check_dtype=False, check_exact=False, rtol=1e-9, atol=1e-12)
            print(f"{name:20s} OK      shape={mem.shape}")
        except AssertionError as e:
            all_ok = False
            print(f"{name:20s} CHECK   {str(e).splitlines()[0]}")
    return all_ok


def save_arrays(path, **arrays):
    """Cache of input arrays (e.g. Xf, yf) so the data pipeline need not be rerun."""
    np.savez_compressed(path, **arrays)


def load_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
