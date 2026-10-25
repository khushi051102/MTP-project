"""Reference values transcribed from the OFFICIAL CWRU Bearing Data Center pages.

Sources (accessed 2026-10-23):
  - https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data
  - https://engineering.case.edu/bearingdatacenter/normal-baseline-data
  - https://engineering.case.edu/bearingdatacenter/bearing-information

These were copied from the pages by hand/by tool. Spot-check a few against the
website yourself, then keep this file frozen: it is the "ground truth" that
verify_cwru.py compares your metadata against.
"""

# approximate motor speed at each load (HP), from the data pages
NOMINAL_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}

# (fault_type, diameter in inches) -> file numbers for loads 0,1,2,3 HP.
# Fault files: 12k drive-end table; outer race = the "@6:00" column only.
REF_FILES = {
    ("normal",     None):  [97, 98, 99, 100],
    ("inner_race", 0.007): [105, 106, 107, 108],
    ("inner_race", 0.014): [169, 170, 171, 172],
    ("inner_race", 0.021): [209, 210, 211, 212],
    ("ball",       0.007): [118, 119, 120, 121],
    ("ball",       0.014): [185, 186, 187, 188],
    ("ball",       0.021): [222, 223, 224, 225],
    ("outer_race", 0.007): [130, 131, 132, 133],
    ("outer_race", 0.014): [197, 198, 199, 200],
    ("outer_race", 0.021): [234, 235, 236, 237],
}

# Drive-end bearing 6205-2RS JEM SKF: defect frequencies as MULTIPLES of the
# running speed in Hz (shaft_hz = rpm / 60), from the "Bearing Information" page.
MULT = {
    "BPFI": 5.4152,    # inner ring
    "BPFO": 3.5848,    # outer ring
    "FTF":  0.39828,   # cage (fundamental train frequency)
    "BSF":  2.357,     # rolling element SPIN (matches the section-1.3 geometric formula)
    "BSF2": 4.7135,    # ball DEFECT impact rate = 2xBSF: the ball hits the inner
                       # AND outer race once per revolution, so impacts come in pairs
}

# which defect frequency belongs to which class (ball fault -> 2xBSF, see MULT)
CLASS_TO_FREQ = {"inner_race": "BPFI", "outer_race": "BPFO", "ball": "BSF2"}
