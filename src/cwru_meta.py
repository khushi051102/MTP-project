"""CWRU metadata: which file number is which class / diameter / load.

File numbers were checked against the official CWRU tables on 2026-10-23
(40 of 40 rows match, using verify_cwru.py).

VERIFIED 2026-10-25: the Normal-baseline sampling rate is 48 kHz, no longer an
assumption. Evidence (verify_cwru.py part 3, official files): all four normal
files keep < 0.3 % of their energy above per-sample frequency 0.25 (i.e. above
12 kHz -> impossible for a 12 kHz recording) while the known-12 kHz fault file
105.mat keeps 40 % in its top band (3-6 kHz). Sample counts agree: 97.mat has
243,938 samples = 5.08 s at 48 kHz (would be an implausible 20.3 s at 12 kHz).
"""
import pandas as pd

from .config import PROJECT_ROOT

LOADS_HP = [0, 1, 2, 3]
NOMINAL_RPM = [1797, 1772, 1750, 1730]   # nominal motor speed at each load
OUTER_RACE_POSITION = "@6"               # ASSUMPTION: only the @6 o'clock files are used
FS_NORMAL = 48_000                       # ASSUMPTION: normal baseline recorded at 48 kHz
FS_FAULT = 12_000                        # drive-end fault files at 12 kHz

# (fault_type, fault diameter in inches, four file numbers for loads 0..3 HP)
GROUPS = [
    ("normal",     None,  [97, 98, 99, 100]),
    ("inner_race", 0.007, [105, 106, 107, 108]),
    ("inner_race", 0.014, [169, 170, 171, 172]),
    ("inner_race", 0.021, [209, 210, 211, 212]),
    ("ball",       0.007, [118, 119, 120, 121]),
    ("ball",       0.014, [185, 186, 187, 188]),
    ("ball",       0.021, [222, 223, 224, 225]),
    ("outer_race", 0.007, [130, 131, 132, 133]),
    ("outer_race", 0.014, [197, 198, 199, 200]),
    ("outer_race", 0.021, [234, 235, 236, 237]),
]


def build_metadata() -> pd.DataFrame:
    rows = []
    for fault_type, diameter, files in GROUPS:
        for num, load, rpm in zip(files, LOADS_HP, NOMINAL_RPM):
            rows.append({
                "file_id": num,
                "fault_type": fault_type,
                "diameter_in": diameter,
                "position": OUTER_RACE_POSITION if fault_type == "outer_race" else None,
                "load_hp": load,
                "nominal_rpm": rpm,
                "fs_native": FS_NORMAL if fault_type == "normal" else FS_FAULT,
                # forward slashes + relative path -> same CSV works on Ubuntu and Windows
                "relpath": f"{fault_type}/{num}.mat",
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = PROJECT_ROOT / "data" / "metadata.csv"
    meta = build_metadata()
    meta.to_csv(out, index=False)
    print(f"wrote {len(meta)} rows to {out}")
