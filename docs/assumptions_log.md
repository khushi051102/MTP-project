# Assumptions log

| ID | Assumption | How I checked it | Status (verified / unverified / risk) |
|----|------------|------------------|----------------------------------------|
| A1 | Drive-end (DE_time) channel is the one analysed | | unverified |
| A2 | 48 kHz, verified by Smith & Randall Table A1| | unverified |
| A3 | Outer-race files used are only the @6 o'clock position | | unverified |
| A4 | File numbers in cwru_meta.py match the official CWRU tables | | verified | 23-oct-2026
| A5 | Measured RPM (inside each .mat) falls as load rises 0 -> 3 HP | audit_cwru.py, flag_rpm_order | unverified |
| A6 | Bearing fault multipliers match my report Equations (8)-(10) | | verified | 23-oct-2026
RPM missing in files 98 and 99, speed hand-recorded so a tolerance band is needed, and the outer-race @6 label inconsistency on the CWRU site.
