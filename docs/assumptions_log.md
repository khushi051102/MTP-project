# Assumptions log

Status words: **verified** (checked against an external source or by a test), **measured** (a result from my own runs), **supported** (evidence from my own data, not conclusive), **chosen** (my decision, not derived), **risk** (known weakness), **limitation**, **resolved**, **unreliable**.

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

## Phase 3: classical baselines (CWRU and FEMTO)

| ID | Assumption | How I checked it | Status |
|----|------------|------------------|--------|
| A37 | CWRU baseline features: 27 per window (10 time-domain, 16 log band energies of 375 Hz each over 0-6 kHz, spectral centroid). FEMTO baseline inputs: last value, mean, standard deviation and change over the 30-snapshot window for each of 8 features (32 numbers) | Defined in baselines.py and the Phase 3 notebook | chosen |
| A38 | Model settings are not tuned: CWRU LogReg (standardised), tree depth 10, Random Forest 200 trees, XGBoost 200 trees depth 6 learning rate 0.1, SVM RBF with C=10. FEMTO tree depth 8 leaf 20, Random Forest 100 trees with a third of the features per split, leaf 10, half the rows per tree, XGBoost 300 trees depth 5 learning rate 0.05 | The lighter Random Forest was chosen because the first version took over 3 minutes per fold in a one-core test | chosen |
| A39 | CWRU leave-one-load-out accuracy: Random Forest 0.996 +- 0.002 over 5 seeds (0 HP 0.986 +- 0.007), LogReg 0.960, XGBoost 0.941, tree 0.912, SVM 0.909 (0 HP only 0.636). Random window split gives about 1.0 for all | Phase 3 runs | measured |
| A40 | Permutation importance (Random Forest, held-out load): band 10 (3.75-4.1 kHz) is by far the largest, band 15 (5.6-6 kHz) second. Correlated features share importance, so the bars are small | Importance plot; band 15 keeps the filter-artifact concern of A13 open | supported |
| A41 | FEMTO prognosis, macro MAE on post-FPT snapshots, one seed: no-skill 0.500, LinReg 0.241, tree 0.255, Random Forest 0.237, XGBoost 0.249. Spread across bearings is about 0.1, paired Random Forest minus LinReg is -0.004 with standard error 0.021, so there is no reliable winner. Pooled over snapshots: Random Forest 0.120, XGBoost 0.147, tree 0.217, LinReg 0.231, so the ranking depends on how the error is averaged | 17-fold leave-one-bearing-out | measured |
| A42 | Bearings with fewer than 15 post-FPT snapshots (5 of 17) give noisy scores. Bearings 2_5 and 3_3 fail for every model; for 3_3 my guess is that it is the only gradual bearing in condition 3 (not tested) | Per-bearing table | limitation |
| A43 | FEMTO detection (healthy versus post-FPT), alarm after 5 consecutive positives. LogReg recall 0.887, false-alarm rate 0.039, median delay 0.5 snapshots; Random Forest 0.834 / 0.085 / 3.5; XGBoost 0.838 / 0.123 / -12; tree 0.792 / 0.128 / -26.5. One bearing never alarms for any model (probably 2_4, which has only 4 post-FPT snapshots, fewer than the run length) | 17-fold leave-one-bearing-out | measured |
| A44 | Early alarms of LogReg (more than 5 minutes before FPT): 1_1, 1_7 and 2_2 coincide with weak precursors or the hump that I left out of the label; 1_6 and 2_3 coincide with short transient bursts (real false alarms). Longer persistence trades false alarms for missed warnings: run 5/10/20/30 gives early alarms 5/4/4/2, late alarms 0/1/2/4, bearings without alarm 1/3/5/8. Run 5 kept | Alarm-persistence table | measured |
| A45 | The FPT label is derived from horizontal RMS, which is also an input, so part of the detection and RUL accuracy is agreement with my rule. Without rms_h: LinReg 0.241, Random Forest 0.251, recall 0.889, false-alarm rate 0.050 (with all features 0.241, 0.237, 0.887, 0.039). Shape features only: MAE 0.337 and 0.334, recall 0.611, false alarms 0.137. The models are not merely echoing the rule, and amplitude carries most of the information | Ablations | supported |
| A46 | A 30-snapshot window mainly reduces false alarms: last snapshot only gives false-alarm rate 0.097 (recall 0.880, MAE 0.268 and 0.260); adding the standard deviation gives 0.038 (recall 0.888, MAE 0.241 and 0.241); adding the trend changes nothing. The Random Forest used 50 trees in this test, so its numbers differ slightly from A41 | Window-feature ablation | supported |
| A47 | C-MAPSS is secondary: baselines on all four sets, no further importance or window study | Decision | chosen |

## Phase 4: deep learning

| ID | Assumption | How I checked it | Status |
|----|------------|------------------|--------|
| A48 | Training protocol: number of epochs fixed in advance (CNN 20, FEMTO models 15), AdamW with learning rate 2e-3 and weight decay 1e-2, one-cycle schedule, batch size 128 (CNN) and 256 (FEMTO). No early stopping, so nothing is selected on a test fold | Design | chosen |
| A49 | Seeding: the seed is set before the model is created and GPU kernels are deterministic. My first version seeded only the batch order, so the earliest CNN numbers must not be used | Repeating seed 1 gave identical fold accuracies [0.7652, 0.983, 1.0, 0.9864] on the Colab GPU | verified |
| A50 | 1D CNN (21,994 parameters, 85.9 KB) on raw windows, input divided by one global scalar (training standard deviation) so amplitude is kept. 5 seeds: accuracy 0.949 +- 0.025, 0 HP 0.854 +- 0.077 (seed range 0.765-0.955); Random Forest 0.996 +- 0.002. The CNN is not better; its errors are only between ball sizes at 0 HP | Phase 4 runs | measured |
| A51 | FEMTO sequence inputs: 8 log ratios to each bearing's own median over snapshots 100-199, sequences of 30 snapshots ending at snapshot 200 or later (21,233 sequences, 3,094 post-FPT), standardised with training-bearing statistics | The baseline window sits before every FPT (earliest 223); weak for Bearing2_5, whose run-in lasts about 500 snapshots | chosen |
| A52 | Loss: L1 with weight 4 on post-FPT snapshots (untuned), sigmoid output in (0, 1) | Design | chosen |
| A53 | Macro MAE on post-FPT snapshots: GRU (4,065 parameters) 0.251 +- 0.010 over 3 seeds (0.251, 0.242, 0.261); Transformer (18,369) 0.256 +- 0.007 (0.262, 0.257, 0.248); TCN (9,793) 0.329 +- 0.006 (0.329, 0.335, 0.323). Seed spread (about 0.01) is far smaller than the spread across bearings (about 0.11-0.14). The networks match the classical models (0.237-0.255) but do not beat them | Phase 4 runs; seed 2 and 3 values confirmed against the printed table | measured |
| A54 | The GRU is best on most single bearings (5 of 17 across all seven models, seed 1; Bearing1_1: 0.049 against 0.108 for the Random Forest), so memory helps on the long gradual degradation but not on abrupt failures | Per-bearing table, seed 1 | supported |
| A55 | Skipped on purpose: CNN-LSTM, Mamba-style state-space model, MC-dropout, early stopping, deep models on C-MAPSS | Decision, for time | chosen |
| A56 | Colab timings are not used as evidence; latency, memory and size are measured only on my laptop with one CPU thread (Phase 6) | Design | chosen |
| A57 | Heavy variants are my own larger versions of the same designs, trained with the same recipe: GRU-L (2 layers, 256 hidden, 599,297 parameters), Transformer-L (4 layers, width 128, 8 heads, 798,209), CNN width 128 (1,294,090). They are not cloud LLMs | Parameter counts printed by the code match the hand count | chosen |
| A58 | CWRU, leave-one-load-out, 5 seeds: wide CNN accuracy 0.961 +- 0.015 (0 HP 0.883 +- 0.026) against small CNN 0.949 +- 0.025 (0 HP 0.854 +- 0.077) and Random Forest 0.996 +- 0.002. The 59 times larger CNN gains 0.012, which is inside the seed spread; it is more stable, and capacity does not close the gap to the Random Forest | Phase 4 runs | measured |
| A59 | FEMTO, macro MAE on post-FPT snapshots, 3 seeds each: GRU 0.251 +- 0.010 (4,065 parameters), Transformer 0.256 +- 0.007 (18,369), Transformer-L 0.255 +- 0.003 (798,209), GRU-L 0.288 +- 0.020 (599,297), TCN 0.329 +- 0.006 (9,793). The classical models score 0.237-0.255, so no neural model beats them. The GRU with 4k parameters equals Transformer-L with about 200 times more | Final Phase 4 table | measured |
| A60 | The larger GRU is worse than the small one by about 0.037 (about two seed spreads). My guess is overfitting, since 17 bearings exist and two supply 82 percent of the degradation data; not tested | Three seeds only | limitation |
| A61 | Phase 4 conclusion: small models are enough on both datasets; neural networks do not beat Random Forest or linear regression. The claim for the SLM must therefore be "matches at lower cost", not "outperforms", and its comparison table must include the Random Forest and linear baselines | Based on A39, A41, A50, A58 and A59 | chosen |
