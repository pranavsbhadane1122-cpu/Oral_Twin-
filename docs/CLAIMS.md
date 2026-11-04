# OralTwin claims register

**Single source of truth for every factual claim this project makes.** Every
other document — paper, figure caption, README, poster, post — cites this file.
This file cites only the code, the configuration and the logs.

Built on 2026-11-02 **from the repository**, not from any existing document.
Existing documents are treated as suspect; that is why this file exists. Where a
document and an artefact disagree, the artefact wins and the disagreement is
recorded here.

**Rule: no claim without a source line.** Anything that cannot be traced appears
under [UNSOURCED](#unsourced) rather than being stated.

### A note on the source hierarchy

The brief for this register assumed every number traces to a run in
`models/logs/experiments.md`. Several do not. They trace to an **artefact file**
produced by a named script — real evidence, but never written up as a logged
run. That is a weaker provenance than a logged run and a stronger one than
nothing, so those rows are marked **`artefact only`** rather than being filed as
unsourced. The gap is itself a finding; see
[What I expected to find and could not](#what-i-expected-to-find-and-could-not).

---

## 1. Measured numbers

### 1.1 Three-class classifier (Kaggle `oral_diseases` + MIO)

| Value | What it measures | Source | Status |
|---|---|---|---|
| **0.8996** | Share of 697 test photographs the first classifier labelled correctly. | experiments.md 2026-09-09 | **DISCARDED** — see below |
| 0.9039 | Accuracy of an earlier training run. | experiments.md 2026-09-09 | **DISCARDED** — the model file was lost to a Colab runtime recycle before download. No artefact exists; it was never evaluated again. |
| 0.8846 | Same first classifier, re-scored on the later clean 52-image split. | experiments.md 2026-09-10 | **DISCARDED** — leakage-confounded |
| **0.6346** [0.499, 0.752] | Share of 52 test photographs the de-biased classifier labelled correctly. | experiments.md 2026-09-10, CI added 2026-09-11 | Current figure for this model. The model is **PARKED**. |
| 0.7885 [0.660, 0.878] | Share of predictions that survive a small re-framing of the same photo (±8°, 0.92–1.08 scale). | experiments.md 2026-09-10 | **Gate ≥ 0.90 MISSED.** `tests/test_training.py` still fails on this by design. |
| 1.0000 | Same, for the contaminated model. | experiments.md 2026-09-10 | **DISCARDED** — perfect stability on memorised images is what memorisation looks like, not robustness. |

**Why 0.8996 was discarded.** Three independent reasons, each sufficient:

1. **Grad-CAM.** Of ten inspected cases only ~3 had heatmaps on teeth or gum
   tissue; failures were systematic (image corners, lips, the blank background
   of a cartoon illustration). experiments.md 2026-09-10.
2. **The split was not independent.** 6,246 raw images in the three source
   folders reduce to **199 distinct base photographs**; the remainder are
   duplicates, flips, rotations, crops and exposure variants. Almost every
   "test" image was a near-duplicate of a training image. experiments.md
   2026-09-10.
3. **Instability.** `pair_0036` flipped from p(caries)=0.99 to 0.00 under a
   small warp of the same photograph. experiments.md 2026-09-10.

> **0.8996 must be read as dataset separability, not accuracy.** It is the most
> important number in this register precisely because it was once the headline.

**Two claims in the 2026-09-10 entry were themselves wrong** and were corrected
on 2026-09-11 (experiments.md keeps both, the correction supersedes):

| Discarded claim | Replacement |
|---|---|
| Grad-CAM "~3/10 → 6/10 improvement" | **Not like-for-like** — the two figures came from different photograph sets. On the *same* ten photographs: v1 6 pass / 2 marginal / 2 fail, v2 6 / 1 / 3. De-biasing did **not** measurably change where the model looks. |
| "calculus 3/3, gingivitis 3/4, caries 0/3" | Counted by **predicted** class. By **true** class the ten cases are 6 calculus and 4 caries, **no true gingivitis at all**: v2 scores calculus 6/6, caries 0/4. |

### 1.2 Binary referral classifier (Piyarathne)

Source for all rows: **experiments.md 2026-10-06**, "Phase 2 retrain: Piyarathne
binary referral (GATE FAILED)". Test split n = 449, patient-disjoint.

| Value | What it measures | Gate | Verdict |
|---|---|---|---|
| 0.630 [0.585, 0.674] | Share of photographs sorted correctly into refer / do-not-refer. | report honestly | reported |
| **0.487** [0.423, 0.551] | Share of cases that *needed* referral which it did refer. | — | **misses half** |
| 0.774 [0.714, 0.824] | Share of cases not needing referral that it correctly left alone. | — | — |
| 0.691 [0.616, 0.757] | Of those it referred, the share that genuinely needed it (PPV). | — | — |
| 0.596 [0.538, 0.651] | Of those it did not refer, the share that genuinely did not need it (NPV). | — | — |
| **6/10** | Heatmaps falling on oral tissue, scored blind by true class. | ≥ 8/10 | **FAIL** |
| 0.902 [0.871, 0.926] | Warp stability. | ≥ 0.90 | PASS, point estimate only — the interval reaches below the gate |

Confusion: TP 111, FP 50, FN 117, TN 171. **117 of 228 cases needing referral
were not referred.**

> **Updated 2026-11-04 from 0.491 to 0.487.** The re-measurement run found TP
> 111 where the 2026-10-06 run found 112 — **one image of 449 changed side**.
> This is resize or preprocessing nondeterminism, **not a change in the
> model**: the checkpoint is byte-identical and every other figure, including
> the external 0.072, reproduced exactly. Recorded because the figure now
> appears in front of readers and the two runs would otherwise disagree.

By original category: Healthy 0.798 [0.713, 0.863] (n=109), Benign 0.750
[0.662, 0.821] (n=112), OPMD 0.464 [0.398, 0.532] (n=209), OCA 0.789
[0.567, 0.915] (n=19 — **underpowered, interval spans 0.35**).

### 1.3 OPMD subtype analysis — hypothesis refuted

Source: **experiments.md 2026-10-06**. No retraining; a re-score of existing
predictions.

| Value | What it measures | Status |
|---|---|---|
| 0.543 [0.402, 0.678] n=46 | Sensitivity on conditions showing a distinct patch or mass. | — |
| 0.472 [0.399, 0.545] n=176 | Sensitivity on conditions showing fine texture (striae, blanching). | — |
| +7.2 points | The gap between them. | **Criterion not met** — intervals overlap heavily |
| **0.370** [0.215, 0.558] n=27 | Patch group **with frank cancer removed**. | Falls **below** the fine-texture group |

The apparent advantage is carried entirely by frank cancer. The resolution
hypothesis is not merely unsupported — the data point the other way.

### 1.4 External validation (SMART-OM, never trained on)

Source: **experiments.md 2026-10-06**. n = 2,270 images, OPMD vs Normal.

| Value | What it measures | Status |
|---|---|---|
| **0.072** [0.038, 0.131] | Share of OPMD cases referred. 9 of 125. | The headline external figure |
| 0.964 [0.955, 0.971] | Share of Normal cases correctly left alone. | — |
| 0.103 [0.055, 0.185] | Of those referred, the share genuinely OPMD. | — |
| 0.915 [0.902, 0.925] | Overall share sorted correctly. | **MUST NEVER BE QUOTED ALONE** — base-rate artefact |
| 0.947 [0.937, 0.956] | NPV — of those not referred, the share genuinely not OPMD. | **MUST NEVER BE QUOTED ALONE** — see below |
| −41.9 points | Fall from internal 0.491 to external 0.072. | The external interval **excludes** the internal figure |

> **Why 0.915 and 0.947 must never be quoted alone.** SMART-OM is **94.5%
> Normal**. A model that answers "no referral" to almost everything scores
> 0.915 accuracy while being useless at the task.
>
> **The NPV is the same trap and reads as more reassuring.** Answering
> "negative" to every image on a 94.5%-negative set yields an NPV of roughly
> **0.945** by arithmetic alone — the measured 0.947 is barely above it. An NPV
> of 0.947 sounds like "when it says no, it is right 95% of the time"; what it
> mostly reflects is that almost everything is negative. Quote it only beside
> the base rate and the 0.072 sensitivity.
>
> This is the same failure mode as 0.9951 below.

### 1.5 Acquisition confound baseline — `artefact only`

Source: **`data/processed/confound_baseline.txt`**. *No corresponding entry in
experiments.md.*

| Value | What it measures | Status |
|---|---|---|
| **0.9951** | Accuracy of a classifier that sees **no image content** — only file extension and EXIF presence. | **DISCARDED as a performance figure** |
| 0.9919 | Accuracy of always answering "not cancer". | **The base rate that kills it** — 0.9951 beats it by 0.3 points |
| 0.400 recall at 1.000 precision | What the metadata rule actually recovers: 8 of 20 Oral Cancer images, with no false positives. | The real finding |

The rule learned, in full, is `is_png > 0.5 → cancer`. This is evidence for the
paper and a reason to keep the Oral Cancer subset out of headline figures. It is
not a model that is kept.

### 1.6 Alignment and change detection — the modules that passed

Source: **experiments.md 2026-10-06**, "Longitudinal pairs regenerated". 200
simulated visit pairs.

| Value | What it measures | Previously published | Status |
|---|---|---|---|
| 200/200 | Pairs the aligner accepted rather than refused. | 200/200 | unchanged |
| **1.18 px** | Median distance between where the four image corners land under the estimated transform and the true one. | 1.17 px | current |
| 98.5% | Share of pairs under 10 px of corner error. | — | current |
| **−0.387** | How well the confidence score **ranks** which alignments are worse. | **−0.516** | **REGRESSION — misses the ≤ −0.50 target.** The published −0.516 must not be quoted without this note. |
| 100% | Share of injected changes the delta module reported. | 100% | unchanged |
| 3.2% | Share of no-change pairs wrongly flagged on colour. | 4.1% | improved |
| 0.0% | Share of no-change pairs wrongly flagged on area. | 1.0% | improved |
| 1.51 pp | Mean error between measured and injected area change. | 1.75 pp | improved |
| 17/17 | Partly out-of-frame regions correctly labelled non-comparable. | 13/13 | unchanged |
| 30.9% | Edge/compactness false-flag rate. | 35.1% | **Known weak. Excluded from all user-facing output by design.** |

**Alignment accuracy is unchanged and good; what degraded is the confidence
score's ranking ability.** The refusal logic keys off the confidence *floor*,
not the ranking, so the practical effect is limited — but the number changed and
the old one is now wrong.

### 1.7 Segmentation — gate failed

Source: **experiments.md 2026-10-06**, "Phase 3 — U-Net segmentation". Test
split n = 449. CIs are bootstrap percentile intervals over images, not Wilson:
Dice is a continuous per-image quantity, not a success count.

| Value | What it measures | Gate | Verdict |
|---|---|---|---|
| **0.933** [0.926, 0.940] | Overlap between predicted and annotated mouth opening (Dice). | — | The project's one validated segmentation figure |
| 0.883 [0.872, 0.893] | Same, as intersection-over-union. | — | — |
| **0.362** [0.332, 0.392] | Overlap between predicted and annotated area of concern. | ≥ 0.70 | **FAIL** |
| 0.278 [0.253, 0.304] | Same, as IoU. | — | — |
| **5/10** | Predicted masks landing on annotated tissue, judged by looking. | ≥ 8/10 | **FAIL** |

**The failure has a shape, and these four numbers are the evidence:**

| Value | What it shows |
|---|---|
| **0.494** | Dice between the predicted *lesion* mask and the true *cavity* mask — **higher than its 0.362 against the lesion it was asked to find**. |
| **r = 0.777** | Correlation between per-image lesion Dice and how much of the mouth the lesion happens to fill (n=340). |
| 0.201 → 0.778 | Lesion Dice for lesions covering <10% of the cavity (n=144) versus >75% (n=14). |
| **15.5%** | Mean share of the frame painted as lesion on 109 **healthy** images, where the correct answer is nothing. Only 2.8% predicted nothing. |

Train lesion Dice 0.364 against validation 0.347: **no generalisation gap.** This
is not overfitting. The model learned the wrong target consistently, so more
epochs or regularisation would not address it.

**Timing:** 6.87 min/epoch steady state; the full run early-stopped at epoch 27
(best epoch 21) in 2 h 27 min.

### 1.8 Region source A/B — cavity model rejected

Source: **experiments.md 2026-10-07**. Both arms ran the same evaluator
functions with only the config key changed.

| Metric | HSV heuristic | Cavity model | Outcome |
|---|---|---|---|
| median corner error | **1.18 px** | 1.41 px | worse |
| p90 corner error | **2.78 px** | 4.64 px | worse |
| under 10 px | **98.5%** | 95.5% | worse |
| delta MAE | **1.51 pp** | 2.29 pp | worse |
| frame coverage | 91.2% | 70.4% | — |
| median ORB keypoints | 840 | 677 | the mechanism |
| runtime | **0.4 min** | 3.4 min | 8× slower |

All 400 images used the cavity model; **no fallback fired**. The 0.933 Dice is
an **in-domain** figure and did not transfer.

**Valid-region sweep:**

| Source | caught | missed | partial | over-flagged | MAE | n(MAE) |
|---|---|---|---|---|---|---|
| frame_intersection | **17** | **0** | **29** | **12** | 1.51 | 171 |
| cavity | 11 | **6** | 112 | 101 | 1.96 | 88 |
| cavity_and_frame | **17** | **0** | 122 | 105 | 1.19 | 78 |

`cavity_and_frame`'s 1.19 pp MAE is **not** the best result in the table: it is
measured on n=78 against n=171, because that variant labels 122 of 200 pairs
partial instead of 29. The apparent gain is a selection effect.

### 1.9 Cavity spot-check on MIO images

Source: **`models/logs/cavity_spotcheck_verdict.txt`** and experiments.md
2026-10-07.

| Value | What it measures |
|---|---|
| 7/10 | Masks giving a sensible answer for "how much of the frame is mouth". |
| 3/10 | Masks reporting **22–40% where the true answer is ~100%** — extreme close-ups already entirely intraoral. |
| 4/10 | Images where the HSV heuristic returned **exactly 100.0%** — its failure path, indistinguishable in output from a genuine full-frame view. |
| 4 of 10 | Caption reports that print a mouth fraction; the other 6 decline it. |

### 1.10 On-device feasibility — `artefact only`

Source: **`data/processed/deploy/pipeline_benchmark.txt`**. *No corresponding
entry in experiments.md.* Desktop CPU on 224×224 images — **not phone silicon**.

| Value | What it measures |
|---|---|
| 81.1 ms mean / 70.7 ms median | Whole alignment + change-detection pipeline, per pair. |
| 62.1 ms mean | Alignment alone — **76.6% of the total**. |
| 12.3 pairs/s | Throughput on one CPU core. |
| 63.4 MB | Peak process memory, including interpreter and OpenCV. |

TFLite conversion, from **`data/processed/deploy/quantization_report.txt`** —
**feasibility measurement on a parked model, never a performance claim**:

| Variant | Size | Mean ms | Accuracy |
|---|---|---|---|
| float32 | 8.88 MB | 20.92 | 0.6346 |
| float16 | 4.48 MB | 17.40 | 0.6346 |
| int8 | 2.72 MB | 15.03 | 0.6538 (+0.0192) |

All within the 15 MB budget. The accuracy column shows only what quantisation
costs relative to the same parked model.

### 1.11 Dataset and corpus figures

| Value | What it measures | Source |
|---|---|---|
| 6,246 → **199** | Raw images in the three Kaggle folders, reducing to distinct base photographs. | experiments.md 2026-09-10 |
| 345 (122/85/138) | Images surviving de-biasing → train 241 / val 52 / test 52. | experiments.md 2026-09-10 |
| 2,382 | Images in the `caries augmented data set` folder, dropped wholesale. | `data/processed/audit_report.txt` |
| 966 / 297 | Clipped-or-blown-out images dropped / images scoring ≥ 0.70 on the drawn-image detector. | experiments.md 2026-09-10 |
| 714 patients | Piyarathne, patient-disjoint → train 2,099 / val 451 / test 449. | experiments.md 2026-10-06; `data/processed/SPLIT_DESIGN.md` |
| 2,270 | SMART-OM images scored externally (of 2,469 unannotated). | experiments.md 2026-10-06 |
| 2,999 of 3,000 | Segmentation cache pairs built (`N-260-01` is truncated). | experiments.md 2026-10-06 |
| 729 | Images with an empty lesion mask — **exactly the Healthy count**, which is what validates the rasterisation. | experiments.md 2026-10-06 |
| 0.8200 (chance 0.333) | Best probe accuracy for identifying which dataset a 64×64 downsample came from; Piyarathne recall 0.970. | **`data/processed/provenance_probe.txt`** — `artefact only` |
| n=97, p95 = 27.47 → 28.0 | No-change lesions used to calibrate the colour threshold. | **`data/processed/delta/color_calibration.json`** — `artefact only` |

### 1.12 Data-quality findings

| Finding | Source |
|---|---|
| `S-184-01` — labelled OPMD, but a photograph of a **closed mouth with no intraoral content**. The model called it no_refer at p=0.92. | experiments.md 2026-10-06 |
| `N-226-01` and `C-24-7-5` — **two** degenerate zero-area oral-cavity polygons. The second was found only because the cache builder counts empty masks. | experiments.md 2026-10-06; `SPLIT_DESIGN.md` limitation 7 |
| Clinical Diagnosis is free text: **45 distinct strings for ~12 conditions**, with repeated misspellings. | experiments.md 2026-10-06 |
| 3 MIO `SANO` (healthy) photographs are the same scenes as 19 Kaggle images labelled `Gingivitis`. | `data/processed/exclusions.csv`; README §4 |

---

## UNSOURCED

Claims that appear in a document and trace to **no run and no retained
artefact**. Listed, not guessed at, and **not to be repeated** until sourced.

| Claim | Where it appears | Why it is unsourced |
|---|---|---|
| **"fired on 88/100 unchanged pairs"** — the original absolute-HSV colour metric | `README.md` line ~112 | Describes a **superseded implementation** whose output was never retained. No artefact in `data/processed` contains it, and it is absent from experiments.md. The current pipeline produces 3.2%. The defect was real; this specific count cannot be checked. |
| **"~85% augmented copies"** of the caries class | experiments.md 2026-09-10 line 43; `README.md` | `audit_report.txt` retains the numerator (2,382 in the augmented folder) but **not the denominator**, so the percentage cannot be recomputed from the retained artefacts. Treat as approximate, as its own "~" admits. |
| **"46-term condition vocabulary"** | My own commit message for the caption layer (`4e2eb81`) | **Wrong.** The guard's list holds **42** terms. Counted from source at build time; see §3. Recorded here because a claim in a commit message is still a claim. |

---

## 2. Capabilities

Read from `configs/config.yaml` and `models/logs/experiments.md`, not from any
plan document.

| Component | Status | Evidence |
|---|---|---|
| Image audit, de-duplication, patient-disjoint splitting | **BUILT** | `tests/test_splits.py`, `SPLIT_DESIGN.md` |
| Alignment (ORB + RANSAC homography) | **BUILT** | 200/200 aligned, median 1.18 px, experiments.md 2026-10-06 |
| Change detection on supplied masks | **BUILT** | recall 100%, colour false-flag 3.2%, MAE 1.51 pp |
| Refusal logic (confidence floor, out-of-frame, non-comparable) | **BUILT** | 17/17 out-of-frame caught |
| Caption layer (describes the photograph) | **BUILT** | 25 tests, experiments.md 2026-10-07 |
| Oral-cavity segmentation | **BUILT, in-domain only** | Dice 0.933 in-domain; **rejected** for alignment use on measurement, experiments.md 2026-10-07 |
| **Three-class classification** | **PARKED** | `classification.surface_predictions: false`. Grad-CAM 6/10 vs ≥ 8/10, warp 0.7885 vs ≥ 0.90. Decision 2026-09-11. |
| **Binary referral classification** | **FAILED GATE → PARKED** | Gate: Grad-CAM ≥ 8/10. Achieved **6/10**. Sensitivity 0.491 internal, **0.072 external**. 2026-10-06. |
| **Lesion segmentation** | **FAILED GATE** | Gate: lesion Dice ≥ 0.70 **and** ≥ 8/10 visual. Achieved **0.362** and **5/10**. 2026-10-06. |
| Risk band / screening band | **BUILT, narrow** | Rule-based only; **excludes** the edge metric (30.9% false-flag). Refused comparisons report `Not assessed`, not `Low`. |
| OPMD / oral-cancer detection | **NOT BUILT** | Blocked on data; confounded subset refused by `assert_scoreable()` |
| Validation on real repeat photographs | **NOT BUILT** | All change-detection validation is on **simulated** visit pairs |
| Android app (M0–M5) | **NOT BUILT** — written, never compiled | No Android toolchain has existed on the build machine; `android/README-android.md` Status |

> **`classification.surface_predictions: false` is the operative fact.** No
> classifier output reaches any user-facing surface, in the Streamlit demo or
> the Android app.

---

## 3. Forbidden vocabulary

**Pulled from source, not retyped.** The authoritative list is
`CONDITION_WORDS` in **`tests/test_caption.py`**. If that list changes, this
section is stale and must be regenerated.

**Count at build time: 42 terms.**

```
caries, calculus, gingivitis, leukoplakia, erythroplakia, lichen planus,
oral cancer, carcinoma, opmd, malignant, premalignant, dysplasia, tumour,
tumor, neoplasm, ulcer, abscess, cavity decay, decay, periodontitis,
gum disease, infection, inflammation, lesion, plaque, tartar, stomatitis,
candidiasis, thrush, cyst, polyp, melanoma, biopsy, benign, diagnos,
disease, healthy mouth, suspicious, severe, mild case, "stage ", "grade "
```

The list is **deliberately wider than the datasets' own labels**: it includes
conditions nobody trained on, because the realistic failure is someone adding a
helpful phrase later, not someone wiring up a classifier.

**Where it is enforced:**

| Surface | Guard |
|---|---|
| Caption layer, every generated sentence on every path | `tests/test_caption.py::TestNoDiseaseNames` |
| Streamlit demo, every UI string including caption text | `tests/test_app.py::ui_strings` + `DIAGNOSTIC_PATTERNS` |
| Android refusal screens | `ComparabilityTest.no_refusal_text_uses_disease_language` |
| Android guided capture | `GuidedCaptureStringsTest.no_capture_string_uses_disease_vocabulary` |
| Android export templates | `ExportStringsTest.no_export_template_uses_clinical_vocabulary` |

**Each guard carries a non-vacuity test** that feeds it a sentence which *must*
fail. Without that, a broken matcher makes every other assertion pass for the
wrong reason. The Android lists are **independently typed, not imported** from
the Python guard — they cannot be, across languages — so they can drift. That is
a known weakness, recorded here rather than hidden.

---

## 4. Parameter provenance

Separating reasoned defaults from calibrated values. A provisional number and a
calibrated number look identical in a config file, and only one of them is
evidence.

**Three states**, and the distinction between the first two matters:

| State | Meaning |
|---|---|
| `calibrated` | Evidence exists **and is written up as a logged run**. |
| `artefact only` | Evidence exists in a named artefact file, but was **never logged as a run** in `models/logs/experiments.md`. Real evidence, weaker provenance. |
| `provisional` | A reasoned default with **no data behind it**. An assumption. |

`src/explain/` parses this table at runtime. The `key` column is the dotted
config path and is what the parser matches on; do not reword it. A threshold
used by a risk rule and missing from this table makes
`tests/test_explain.py::test_every_rule_threshold_is_registered` fail — an
unregistered threshold deciding a band is the exact thing this register was
built to catch.

<!-- PARAMETER-TABLE-START -->

| key | value | state | evidence, or what it needs to be settled |
|---|---|---|---|
| `delta.relative_color_thresh` | 28.0 | `artefact only` | p95 of relative-colour magnitude over **97 no-change lesions** = 27.47, rounded up. Lives in `data/processed/delta/color_calibration.json` and **is not written up as a run** in experiments.md, so it does not qualify as calibrated. To promote it: write the calibration up as a logged run. |
| `dedup.phash_threshold` | 8 | `artefact only` | Swept; 10 chained unrelated images into a 364-image cluster. The sweep is not in experiments.md. |
| `dedup.family_phash_threshold` | 14 | `artefact only` | Set from the Phase 1 augmentation-family analysis; recorded in `audit_report.txt`, not as a run. |
| `risk.growth_moderate_pct` | 10.0 | `provisional` | What size of area change a clinician considers worth a professional look. **No clinical input has been obtained.** Needs a clinician to set it. |
| `risk.growth_elevated_pct` | 30.0 | `provisional` | As above, for a clear change. Needs a clinician to set it. |
| `risk.escalate_on_rule_count` | 2 | `provisional` | How many Moderate signals together should read as one Elevated. Never swept; needs outcome data that does not exist. |
| `risk.condition_prob_high` | 0.70 | `provisional` | Moot while `surface_predictions: false` - no rule using it can fire. |
| `delta.confidence_floor` | 0.3 | `provisional` | A human judgement of which alignments are too poor to compare. Note §1.6: the confidence score's **ranking** ability regressed to -0.387, which weakens any threshold placed on it. Needs a sweep against human judgements of alignment quality. |
| `delta.min_overlap_frac` | 0.60 | `provisional` | **Real repeat photographs of the same mouth.** Added 2026-10-22 for the Android comparability code, which has no lesion masks and asks a frame-overlap question instead of a per-lesion edge question. |
| `delta.change_area_thresh` | 0.10 | `provisional` | What counts as a notable area change. No clinical input. |
| `delta.compactness_change_thresh` | 0.15 | `provisional` | Moot - the edge metric is excluded from all output (30.9% false-flag). |
| `delta.edge_margin_px` | 3 | `provisional` | Chosen to be small; never swept. |
| `delta.match_iou_thresh` | 0.10 | `provisional` | Never swept. |
| `segmentation.bce_weight` | 0.5 | `provisional` | Never swept. **Loss reweighting toward the lesion channel is named future work and was deliberately left untested** - retrying on the same holdout immediately after a gate failure would compromise it. experiments.md 2026-10-07. |
| `alignment.cavity_threshold` | 0.5 | `provisional` | Moot - the cavity path lost its A/B and is retained only so the result can be rechecked. |
| `alignment.cavity_min_area_frac` | 0.05 | `provisional` | As above. |
| `alignment.cavity_dilate_px` | 6 | `provisional` | As above. |
| `screening.flag_threshold` | 0.5 | `provisional` | The argmax point of a two-class softmax, which is where the model's own decision already sits — **not a value anyone chose against an operating point**. Moving it trades the 0.072 external sensitivity against the 0.103 external PPV, and nobody has decided which way that trade should go. Needs someone to state what a missed case costs relative to a false flag. |
| `MatchIndicator.SCALE` | 40.0 | `provisional` | **Needs nothing, and that is the point.** It stretches a bar; it is not a threshold. The mapping contains no comparison, so it cannot create a state the bar did not have. Listed so nobody later mistakes it for a threshold and starts tuning it. |

<!-- PARAMETER-TABLE-END -->

**Nothing in this project is currently `calibrated`.** Every parameter with
evidence behind it has that evidence in an artefact file rather than in a
logged run. That is a consequence of finding 1 in
[What I expected to find and could not](#what-i-expected-to-find-and-could-not),
and it is why the explanation layer says `artefact only` instead of rounding up
to calibrated.

## 4b. Decisions — components surfaced despite a failed or unmet gate

**This is the record, not a judgement.** A component shipped with a failed gate
and no recorded decision looks concealed. A recorded decision is a position
that can be defended, argued with, or reversed.

On **2026-11-04** three components the register lists as PARKED or FAILED GATE
were surfaced as Tracks A, B and C. Every one was re-measured first
(`experiments.md`, 2026-11-04) and every score now carries its measured
performance as a single inseparable object.

### Track A — gum condition (`classifier_3class.h5`)

| | |
|---|---|
| Gate | Grad-CAM ≥ 8/10 on tissue; warp stability ≥ 0.90 |
| Achieved | **6/10** and **0.7885** |
| Date failed | 2026-09-10, confirmed 2026-09-11 |
| Date surfaced | **2026-11-04** |
| Current performance | accuracy 0.635 [0.499, 0.752] on 52 images. Per-class sensitivity: caries 0.833 [0.608, 0.942] (n=18), calculus 0.692 [0.424, 0.873] (n=13), **gingivitis 0.429 [0.245, 0.635] (n=21)** |
| External | **None exists and none can be run.** No second dataset in the project shares these three labels. |

The gingivitis interval spans "misses three quarters" to "finds two thirds".

### Track B — lesion outline (`segmenter.h5`, lesion channel)

| | |
|---|---|
| Gate | lesion Dice ≥ 0.70 **and** ≥ 8/10 overlays on annotated tissue |
| Achieved | **0.362** and **5/10** |
| Date failed | 2026-10-06 |
| Date surfaced | **2026-11-04** |
| Current performance | mean Dice **0.362** over 449 photographs; per-pixel sensitivity 0.689, **PPV 0.376** |
| External | **None exists and none can be run.** No second dataset carries lesion polygons; SMART-OM's were excluded in Phase 1 as burned into the images. |

The model was found to have learned to outline the mouth rather than the
lesion — Dice against the true cavity (0.494) exceeds Dice against the true
lesion (0.362). **That finding is unchanged by this decision.**

### Track C — screening score (`classifier_binary_referral.h5`)

| | |
|---|---|
| Gate | Grad-CAM ≥ 8/10 on oral tissue |
| Achieved | **6/10** |
| Date failed | 2026-10-06 |
| Date surfaced | **2026-11-04** |
| Current performance, internal | sensitivity 0.487 [0.423, 0.551], specificity 0.774, PPV 0.689, NPV 0.594 — on a set that is 50.8% positive |
| Current performance, external | **sensitivity 0.072 [0.038, 0.131] — 9 of 125** — on a set that is 5.5% positive |

**This is the only track with an external figure, and it is 0.072.** The
not-flagged branch states that count verbatim, generated from the measurement
file rather than typed, because a model that finds 9 of 125 must not produce a
silence that reads as an all-clear.

### What did not change

`surface_predictions` moved to `true`, but **no failed gate was reopened, no
threshold was moved to make anything pass, and no measurement was re-run until
it improved.** The gates remain failed and are recorded above as failed. What
changed is that the outputs are shown **with** their failure attached, rather
than withheld.

## 5. What this project does not do

Explicit, so the next document can be checked against it.

- **Does not name a condition.** No component outputs a condition name, and five
  independent guards enforce it (§3).
- **Does not assign a diagnostic category.** No severity, no grade, no stage.
- **Does not route to a clinical action.** It says "have this checked by a
  dentist"; it never says whether, when, or how urgently.
- **Does not detect cancer or OPMD.** Never built. The one subset that could
  have supported it is confounded with acquisition metadata and is refused in
  code by `assert_scoreable()`.
- **Does not measure whether anything got better or worse clinically.** It
  measures area, colour and edge change of regions **supplied to it**, and the
  edge metric is excluded from output.
- **Does not produce the regions it measures.** The lesion segmenter failed its
  gate. Regions come from simulated masks or a labelled placeholder.
- **Does not give a result when it cannot do the work.** A refused comparison
  reports `Not assessed` and carries no number and no colour band.
- **Does not claim accuracy for any classifier.** Both are parked;
  `surface_predictions: false`.
- **Has not been validated on real repeat photographs.** All change-detection
  validation is on simulated visit pairs with known ground truth.
- **Has not been reviewed or approved by any medical or dental body**, and has
  not been tested in a clinic.
- **Does not send photographs anywhere.** The Android manifest requests no
  `INTERNET` permission and removes it from merged manifests. If the user
  shares a photo themselves, it leaves the phone at that point.

---

## What I expected to find and could not

Reported as instructed, rather than papered over.

1. **Four significant numbers have no entry in `experiments.md`.** The confound
   baseline (0.9951), the provenance probe (0.8200), the pipeline benchmark
   (81.1 ms) and the colour calibration (n=97, p95 27.47) all trace to artefact
   files produced by named scripts, but were never written up as logged runs.
   They are real evidence with weaker provenance, marked `artefact only` above.
   **The experiment log is not the complete record it is assumed to be.**

2. **The `88/100` colour false-flag count cannot be reproduced.** The defect was
   real and the fix is documented, but the specific count describes an
   implementation that no longer exists and whose output was not retained.

3. **The `~85%` augmented-caries figure cannot be recomputed.** `audit_report.txt`
   retains the numerator but not the denominator.

4. **No clinician has scored anything.** Every Grad-CAM verdict and every
   segmentation overlay verdict in this register was made by the AI assistant
   that built the pipeline. Some were blind to the model's prediction; none were
   blind to the assistant's own earlier work. experiments.md says they must be
   re-scored before publication. They have not been.

5. **No Android figure exists at all.** Six milestones of Kotlin have never been
   compiled, so §2 lists the app as NOT BUILT. Any claim about the app's
   behaviour is currently a claim about source code, not about a program that
   has run.

6. **One claim in this project's own commit history is wrong** and is corrected
   here: the caption guard holds 42 terms, not the 46 I stated. Recorded in
   UNSOURCED because a commit message is a document too.

---

*OralTwin is a screening aid, not a diagnostic tool. It does not identify dental
conditions. Any concern should be checked by a dentist.*
