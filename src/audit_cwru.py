"""Phase 0 audit: do the files, sampling rates and loads match the metadata?

Run from the project root:   python -m src.audit_cwru
"""
import pandas as pd

from .config import CWRU_RAW, RESULTS
from .cwru_meta import build_metadata
from .data_loader import load_cwru_file

DURATION_RANGE_S = (4.0, 13.0)   # sanity range, NOT a rule: some files are ~5 s
RPM_TOLERANCE = 30               # measured vs nominal rpm, loose on purpose


def run_audit(raw_root=CWRU_RAW, save=True) -> pd.DataFrame:
    meta = build_metadata()

    # 1) every listed file must exist
    meta["exists"] = [(raw_root / p).exists() for p in meta["relpath"]]
    missing = meta.loc[~meta["exists"], "relpath"].tolist()

    # 2) every file on disk must be listed (otherwise it is an unlabeled file)
    on_disk = {f"{p.parent.name}/{p.name}" for p in raw_root.glob("*/*.mat")}
    unlisted = sorted(on_disk - set(meta["relpath"]))

    # 3) measure what the files really contain
    n_native, rpm_meas = [], []
    for row in meta.itertuples():
        if not row.exists:
            n_native.append(None)
            rpm_meas.append(None)
            continue
        rec = load_cwru_file(raw_root / row.relpath, row.fs_native)
        n_native.append(rec.n_native)
        rpm_meas.append(rec.rpm)
    meta["n_native"] = n_native
    meta["rpm_measured"] = rpm_meas
    meta["duration_s"] = meta["n_native"] / meta["fs_native"]
    meta["rpm_diff"] = meta["rpm_measured"] - meta["nominal_rpm"]

    # 4) flags
    lo, hi = DURATION_RANGE_S
    meta["flag_duration"] = (meta["duration_s"] < lo) | (meta["duration_s"] > hi)
    meta["flag_rpm"] = meta["rpm_diff"].abs() > RPM_TOLERANCE

    # 5) within a group, measured rpm must fall as load rises 0 -> 3 HP
    keys = ["fault_type", "diameter_in", "position"]
    meta["flag_rpm_order"] = False
    for _, g in meta.groupby(keys, dropna=False):
        g = g.sort_values("load_hp")
        if g["rpm_measured"].notna().all():
            ok = (g["rpm_measured"].diff().dropna() < 0).all()
            meta.loc[g.index, "flag_rpm_order"] = not ok

    print(f"listed: {len(meta)}  missing: {len(missing)}  unlisted on disk: {len(unlisted)}")
    if missing:
        print("MISSING:", missing)
    if unlisted:
        print("UNLISTED:", unlisted)
    cols = ["relpath", "load_hp", "fs_native", "n_native", "duration_s",
            "nominal_rpm", "rpm_measured", "rpm_diff"]
    print(meta[cols].to_string(index=False))
    flagged = meta[meta[["flag_duration", "flag_rpm", "flag_rpm_order"]].any(axis=1)]
    print(f"\nFLAGGED ROWS: {len(flagged)}")
    if len(flagged):
        print(flagged[["relpath", "duration_s", "rpm_diff",
                       "flag_duration", "flag_rpm", "flag_rpm_order"]].to_string(index=False))

    if save:
        RESULTS.mkdir(exist_ok=True)
        meta.to_csv(RESULTS / "cwru_audit.csv", index=False)
    return meta


if __name__ == "__main__":
    run_audit()
