# Assumptions log

| ID | Assumption | How I checked it | Status |
|----|------------|------------------|--------|
| A1 | The drive-end channel (DE_time) is the one analysed | Loader asserts exactly one DE_time key; envelope peaks match expected defect frequencies on files 106 and 131 | supported (3 files) |
| A2 | Normal files are 48 kHz, fault files 12 kHz | Smith and Randall Table A1; file 98 spectrum has content to about 20 kHz and no wall at 6 kHz | verified |
| A3 | Only outer-race @6 o'clock files are used | CWRU site labels @6 inconsistently between its pages | risk |
| A4 | File numbers in cwru_meta.py match the official tables | verify_cwru.py, 40 of 40 rows, 2026-10-23 | verified |
| A5 | Measured RPM falls as load rises | audit_cwru.py; Normal group not checked (files 98 and 99 have no RPM) | verified except Normal |
| A6 | Defect multipliers BPFI 5.4152, BPFO 3.5848, FTF 0.39828, BSF 4.7135 | Official Bearing Information page, 2026-10-23 | verified |
| A7 | RPM was recorded by hand, so defect frequencies need a tolerance band of 3 percent | Observed sidebands about 0.4 Hz below expected (about 0.25 percent) | accepted risk |
| A8 | My harmonic sampling-rate test | Disagreed with the paper on the Normal files; cause not established | unreliable, not used |
| A9 | Window 2048 with hop 1024 holds several fault impulses | Fill in the events-per-window numbers once run | verified | (27.3, 18.1, 23.8 events per window).
| A10 | Envelope band 2000-5500 Hz, 4th-order Butterworth, zero-phase | Works on files 106 and 131, fails on ball file 119; band not tuned | verified | supported (spectrogram shows the 2.5–4 kHz resonance).
| A11 | Raw and envelope spectra are compared with dominance, not peak-to-floor | Peak-to-floor is not comparable across spectrum types | verified |
| A12 | CWRU split by load (leave-one-load-out), 10 classes | Random split gives about 1.0; honest split gives 0.91 to 0.996 | verified |
| A13 | Results are not driven by the 4.5-6 kHz decimation artifact | Dropping the top four bands gives 0.982 versus 0.996; other artifacts not excluded | partly checked |
| A14 | C-MAPSS RUL cap of 125 on training and on the test target | Chosen, not derived; state it when comparing with papers | chosen |
| A15 | Six operating conditions in FD002 and FD004 | KMeans clusters match rounded settings one to one | verified |
| A16 | Sensors dropped if constant within a condition or if fewer than 5 distinct values | s6 dropped in FD001 (2 distinct values, correlation -0.12) | chosen |
| A17 | Scaler fit on training data only | By construction in ConditionScaler.fit | verified |
| A18 | FD004 readme says 248 train and 249 test; files say 249 train and 248 test | Row and engine counts, RUL file has 248 rows; files trusted | documented |
| A19 | FEMTO sampled at 25.6 kHz | Dataset documentation; check_fs result pending | verified | 25,601 Hz implied from the snapshot’s timestamp span.
| A20 | validation_set is the full run of each test bearing | Counts match official RUL for 10 of 11 bearings; Bearing1_4 gives 289 snapshots versus official 339 s; official value used | verified with one exception |
| A21 | FEMTO failure time is the last snapshot | Failure was defined by amplitude above 20 g; files run a few percent longer | risk |
| A22 | First predicting time rule (reference window, 3 sigma, 5 consecutive, median smoothing) | Parameters are my choices; Bearing1_1 first gave 5.4 h of 7.78 h | unverified |
| A23 | FEMTO timestamps are monotonic with a 10 s spacing | Plot shows stray horizontal lines; diagnostic pending | risk | confirmed, snapshots 2120 and 2121 of Bearing1_1 have corrupted timestamps. The time axis is idx × 10 s.
the six bearings differ in shape (gradual, abrupt, hump), so FPT must be checked per bearing.
A20 becomes “verified for 10 of 11; Bearing1_4 uses the official 339 s and is cut to 1,172 snapshots”.
A21 stays “risk” (last snapshot treated as failure for the 6 training bearings).
New: time axis is idx × 10 s, and t_stamp keeps the original for the record.
