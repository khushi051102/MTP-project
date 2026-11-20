"""Save and reload result tables so a new Colab session never has to retrain.

save_results({"allres": allres, ...}, folder)  -> one CSV per table + manifest.json
load_results(folder)                          -> dict of DataFrames, same names

The manifest records, for every table: rows, columns, the git commit and the time it
was saved, so every number in the thesis can be traced back to the code that made it.
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


def save_results(tables, folder):
    """tables: dict name -> DataFrame. Existing files with the same name are overwritten;
    the manifest keeps entries for tables saved earlier under other names."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    man_path = folder / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    commit, now = git_commit(), datetime.now().isoformat(timespec="seconds")
    for name, df in tables.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{name} is {type(df).__name__}, not a DataFrame")
        df.to_csv(folder / f"{name}.csv", index=False)
        manifest[name] = {"rows": len(df), "columns": list(df.columns),
                          "commit": df.attrs.get("commit", commit), "saved": now}
    man_path.write_text(json.dumps(manifest, indent=2))
    print(f"saved {len(tables)} tables to {folder} (commit {commit})")


def load_results(folder, names=None):
    """Load every table listed in the manifest (or only `names`) and check row counts."""
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    out = {}
    for name in names or manifest:
        df = pd.read_csv(folder / f"{name}.csv")
        info = manifest[name]
        assert len(df) == info["rows"], f"{name}: {len(df)} rows, manifest says {info['rows']}"
        df.attrs["commit"] = info["commit"]
        out[name] = df
        print(f"{name:15s} rows={len(df):6d}  commit={info['commit']}  saved={info['saved']}")
    return out


def save_arrays(path, **arrays):
    """Cache of input arrays (e.g. Xf, yf) so the data pipeline need not be rerun."""
    np.savez_compressed(path, **arrays)


def load_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
