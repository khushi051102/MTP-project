# Assumptions log

Status words: **verified** (checked against an external source or by a test), **supported** (evidence from my own data, not conclusive), **chosen** (my decision, not derived), **risk** (known weakness), **resolved**, **unreliable**.

## CWRU (bearing fault classification)

| ID | Assumption | How I checked it | Status |
|----|------------|------------------|--------|
| A1 | The drive-end channel (DE_time) is the one analysed | Loader asserts exactly one DE_time key; envelope peaks match the expected defect frequencies on files 106 and 131 | supported (2 files) |
| A2 | Normal files are 48 kHz, fault files 12 kHz | Smith and Randall Table A1; spectrum of file 98 has content up to about 20 kHz and no wall at 6 kHz | verified |
| A3 | Only the outer-race @6 o'clock files are used | The CWRU site labels the @6 position inconsistently between its pages (centered on one, load zone on another) | risk |
| A4 | File numbers in cwru_meta.py match the official tables | verify_cwru.py, 40 of 40 rows, 2026-10-23 | verified |
| A5 | Measured RPM falls as load rises from 0 to 3 HP | audit_cwru.py; the Normal group could not be checked because files 98 and 99 contain no RPM (nominal RPM used for them) | verified except Normal |
| A6 | Defect multipliers BPFI 5.4152, BPFO 3.5848, FTF 0.39828, BSF 4.7135 | Official CWRU Bearing Information page, 2026-10-23 | verified |
| A7 | RPM was recorded by hand, so defect frequencies need a tolerance band (3 percent) | Observed sidebands sit about 0.4 Hz (about 0.25 percent) below the expected values | accepted risk |
| A8 | My harmonic sampling-rate test | Disagreed with the paper on the Normal files; cause not established | unreliable, not used |
| A9 | A window of 2048 samples with hop 1024 holds several fault impulses | About 27.3 (BPFI), 18.1 (BPFO) and 23.8 (BSF) events and 5.0 shaft turns per window | verified |
| A10 | Envelope band 2000-5500 Hz, 4th-order Butterworth, zero-phase | Works on files 106 and 131, fails on ball file 119; spectrogram shows the resonance at about 2.5-4 kHz; band not tuned | supported |
| A11 | Raw and envelope spectra are compared with dominance, not peak-to-floor | Peak-to-floor is not comparable across spectrum types. Dominance raw / envelope: file 106 0.69 / 2.37, file 131 0.13 / 1.35, file 119 0.03 / 0.47 | verified |
| A12 | CWRU split by load (leave-one-load-out), 10 classes | Random window split gives about 1.0 for every model; leave-one-load-out gives RandomForest 0.996, LogReg 0.960, XGBoost 0.941, DecisionTree 0.912 | verified |
| A13 | Results are not driven by the 4.5-6 kHz decimation artifact | Dropping the top four bands gives 0.982 versus 0.996; other recording artifacts not excluded | partly checked |
| A14 | Leave-one-load-out is honest about load changes only | Each class is one specimen and one recording per load, so a new bearing or a new fault is never tested; the 0 HP fold is the weakest (extrapolation at the edge of the load range) | risk |
| A15 | Time-domain features alone are weaker than time plus spectral features | Random Forest, leave-one-load-out: time only 0.941, bands only 0.981, all 0.996. Kurtosis and crest factor alone cannot separate ball 0.007 and outer 0.014 from Normal; RMS and band energies can | supported |

## C-MAPSS (turbofan RUL)

| ID | Assumption | How I checked it | Status |
|----|------------|------------------|--------|
| A16 | RUL cap of 125 on the training target and on the test target | Chosen, not derived; state it when comparing with papers that use the uncapped test RUL | chosen |
| A17 | Six operating conditions in FD002 and FD004 | KMeans clusters match the rounded settings one to one (cluster sizes are unequal in the data itself) | verified |
| A18 | Sensors are dropped if constant within a condition or if they have fewer than 5 distinct values | s6 dropped in FD001 (2 distinct values, correlation -0.12); s9 and s14 kept despite weak whole-life correlation | chosen |
| A19 | The scaler is fit on training data only | By construction in ConditionScaler.fit | verified |
| A20 | FD004 readme says 248 train and 249 test engines; the files say 249 train and 248 test | Row and engine counts; the RUL file has 248 rows; files trusted | documented |
| A21 | Validation engines: GroupShuffleSplit(test_size=0.2, random_state=42), 80 train and 20 validation engines for FD001; sequences of 30 cycles | Overlap asserted empty; shape check (17731, 30, 14) was run on all 100 engines | verified |
| A22 | Baselines are single runs with default hyperparameters on the official test set (last cycle per engine) | RMSE differences below about 1 are probably noise; linear regression is competitive everywhere; NASA score does not follow RMSE | limitation |

## FEMTO (bearing RUL, extended protocol)

| ID | Assumption | How I checked it | Status |
|----|------------|------------------|--------|
| A23 | FEMTO is sampled at 25.6 kHz | check_fs on Bearing1_1: 25,601 Hz implied from the timestamp span of one snapshot | verified |
| A24 | The microsecond column of the test and validation files is written with 5 significant digits | check_fs is only reliable on training bearings; timestamps are used only for ordering | documented |
| A25 | validation_set is the full run of each test bearing | Counts minus test_set counts reproduce the official RUL for 10 of 11 bearings | verified for 10 of 11 |
| A26 | Bearing1_4 uses the official RUL of 339 s and is cut to 1,172 snapshots | validation_set gives 289 snapshots (2,890 s) instead of 339 s; the last kept snapshot therefore has RUL 9 s by design | chosen |
| A27 | Official RUL values (OFFICIAL_RUL) | Transcribed from published tables; 10 of 11 are reproduced independently from my own file counts; the 339 s of Bearing1_4 comes only from the published tables | partly verified |
| A28 | Failure time is the last snapshot of the run (except Bearing1_4) | Failure was defined by amplitude above 20 g and the files run a few percent longer than one paper's estimate of the failure moment | risk |
| A29 | Time axis is idx x 10 s | Timestamps are exactly 10 s apart in 16 of 17 bearings; Bearing1_1 snapshots 2120 and 2121 carry corrupted stamps (t = -52 s). Original kept as t_stamp | resolved |
| A30 | First predicting time (FPT) per bearing | Automatic stay-above rule (baseline 10-30 percent of the run, 3 sigma, rolling median of 9) with manual overrides: 1_7 (2208, jump after a weak drift), 2_1 (874, first sustained step), 2_2 (750, mid-life hump treated as a transient), 2_5 (2303, weak intermittent rise at about 2253 ignored). Offline only: it uses the length of the run | chosen |
| A31 | Only 12.6 percent of snapshots (3,094 of 24,633) are after FPT; 1_1 and 1_3 supply 82 percent of them; eleven bearings have fewer than 60 post-FPT snapshots | Table of post-FPT counts | limitation |
| A32 | Protocol: all 17 full runs (6 training_set + 11 validation_set), leave-one-bearing-out | Results are not comparable to papers that use the official split | chosen |
| A33 | Target rul_frac: 1.0 until FPT, then linear to 0 at the end of the run | Seconds with a cap would label 74 percent of post-FPT snapshots as capped at 1,200 s; uncapped seconds let two bearings dominate the error | chosen |
| A34 | Headline metric: macro MAE_post (per bearing, post-FPT snapshots only); the no-skill baseline "always healthy" scores 0.5 (0.495 for Bearing1_4) | MAE over all snapshots is not used because it flatters any model (87 percent of snapshots are healthy) | verified |
| A35 | Model inputs: 8 features (RMS, kurtosis, peak, crest for horizontal and vertical), log ratio to the bearing's own median over snapshots 100-199, sequences of 30 snapshots ending at idx 200 or later | Baseline window sits before every FPT (earliest 223); weak for Bearing2_5, whose run-in lasts about 500 snapshots | chosen |

## Final FPT values (snapshot index)

1_1: 1352, 1_2: 824, 1_3: 1291, 1_4: 1087, 1_5: 2415, 1_6: 2421, 1_7: 2208, 2_1: 874, 2_2: 750, 2_3: 1950, 2_4: 747, 2_5: 2303, 2_6: 691, 2_7: 223, 3_1: 495, 3_2: 1589, 3_3: 319

## Reproducibility

| ID | Assumption | How I checked it | Status |
|----|------------|------------------|--------|
| A36 | Seeds are pinned with set_seed(42) at the top of every notebook; data/raw is read-only (chmod -R a-w) | Done; raw data and processed data are excluded from git | verified |
