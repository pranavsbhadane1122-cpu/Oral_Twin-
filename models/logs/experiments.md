# OralTwin — experiment log

## 2026-09-09 — classifier_3class v1 (MobileNetV2 transfer learning)

- **Data:** data/processed/classification, 3-class mode (caries / calculus / gingivitis;
  "healthy" pending external dataset). Train 3255 / val 698 / test 697, leakage-free
  pHash splits, seed 42.
- **Config:** img 224, batch 32, epochs 30/stage, Adam lr 1e-3 (head) → 1e-4
  (fine-tune last 30 layers, BN frozen), dropout 0.3, class weights
  {caries 0.60, calculus 1.88, gingivitis 1.25}, EarlyStopping patience 5 on val_loss.
- **Hardware:** Google Colab T4 GPU (TF 2.20); local env is TF 2.15 CPU.
- **Test metrics (697 images):** accuracy **0.8996**
  | class | precision | recall (sens.) | specificity | F1 |
  |---|---|---|---|---|
  | caries | 0.9946 | 0.9483 | 0.9935 | 0.9709 |
  | calculus | 0.7734 | 0.8049 | 0.9495 | 0.7888 |
  | gingivitis | 0.8050 | 0.8610 | 0.9235 | 0.8320 |
- **Main error mode:** persistent two-way gingivitis <-> calculus confusion
  (25 + 23 errors) — both present as inflamed gum-line imagery; caries is
  nearly perfectly separated.
- **Notes:**
  - A first training run reached 90.39% but was LOST to a Colab runtime recycle
    before the model zip was downloaded; this 89.96% model is the retrain.
  - Colab's TF 2.20 (Keras 3) h5 does not load in local TF 2.15; converted with
    `scripts/convert_colab_model.py` (weight copy by name, 262/262 matched) and
    verified by locally reproducing the exact Colab test metrics.
  - Artifacts: models/classifier_3class.h5 (converted, local-loadable),
    models/classifier_3class.keras3_colab.h5 (original Colab file),
    logs/classifier_3class_history.csv, logs/confusion_matrix_3class.png,
    logs/eval_report_3class.txt.

### 2026-09-10 — v1 INVALIDATED by Grad-CAM inspection

Phase 7 explainability inspection of 10 representative cases found only ~3/10
heatmaps concentrating on teeth or gum tissue. Failures were systematic, not
random: pair_0025 attended to the four image corners and lips, pair_0150 to the
upper lip and skin, pair_0036 to the lower lip while teeth with visibly dark
caries stayed cold, pair_0097 to the blank background of a cartoon illustration.
pair_0036 also flipped from p(caries)=0.99 to 0.00 under a small warp of the
same photo — behaviour inconsistent with reading real pathology.

Diagnosis: shortcut learning driven by dataset provenance. The Phase 1 audit had
already recorded that the caries class is ~85% augmented copies of a few base
photos and that the source set mixes illustrations with clinical photos.

**The 0.8996 figure should be read as dataset separability, not clinical
accuracy.** v1 is retained only as the comparison baseline
(models/classifier_3class_v1_contaminated.h5) and must not ship.

## 2026-09-10 — classifier_3class v2 (de-biased data, identical hyperparameters)

**The headline finding is about the data, not the model: 6246 raw images in the
three source folders reduce to 199 distinct base photographs.** Everything else
is exact duplicates, flips, rotations, crops and exposure variants.

De-biasing applied (all three filters, user-approved 2026-09-10):
  - the entire `caries augmented data set` folder dropped (2382 images)
  - 966 clipped/blown-out images dropped
  - 297 images scoring >= 0.70 on the drawn/texture-free detector dropped
  - augmentation families collapsed at pHash distance 14, max 3 images/family
  - classes capped at 2x the smallest

Resulting dataset: **345 images** (caries 122, calculus 85, gingivitis 138)
from 199 families -> train 241 / val 52 / test 52. Hyperparameters unchanged.
Trained locally on CPU (TF 2.15) rather than Colab: at this size training takes
minutes, and it avoids the Keras 3 -> Keras 2 conversion entirely. Early
stopping fired at stage-b epoch 6, restoring epoch 1.

### Both models on the SAME clean test split (52 images)

| metric | v1 contaminated | v2 de-biased |
|---|---|---|
| accuracy | 0.8846 | **0.6346** |
| warp stability (+/-8 deg, 0.92-1.08 scale) | 1.0000 (0/52 flipped) | **0.7885** (11/52) |
| Grad-CAM on-target (10 cases) | ~3/10 | **6/10** |
| caries P / R / F1 | 1.000 / 0.722 / 0.839 | 0.682 / 0.833 / 0.750 |
| calculus P / R / F1 | 0.929 / 1.000 / 0.963 | 0.562 / 0.692 / 0.621 |
| gingivitis P / R / F1 | 0.800 / 0.952 / 0.870 | 0.643 / 0.429 / 0.514 |

**v1's numbers on this table are not trustworthy.** v1 trained on 3255 images
drawn from the same 199 families that the new test split is derived from, so
almost every "test" image is a near-duplicate of something v1 memorised. Its
0.8846 and its perfect warp stability are both what memorisation looks like.
v1's original 0.8996 was measured on the old contaminated split.

### Acceptance against the Phase 2b gate

  (a) accuracy reported honestly: **0.6346**, down from 0.8996. Expected.
  (b) Grad-CAM localisation: **6/10 on target, target was >= 8/10 - MISSED**,
      but up from ~3/10. The split by class is the informative part:
      calculus 3/3, gingivitis 3/4, **caries 0/3**.
  (c) warp stability: **0.7885, target was >= 0.90 - MISSED**. The test is kept
      in tests/test_training.py and currently fails by design; nothing was
      tuned to make it pass.

### Conclusion

De-biasing measurably improved *what the model looks at* (gum margins and
cervical tooth surfaces for gingivitis and calculus, where those conditions
actually present) while lowering accuracy, which is the expected direction. But
neither acceptance gate was met, and caries localisation did not improve at all.

With 199 base photographs, 241 of which are training images across three
classes, this corpus cannot support a trustworthy classifier. The honest
conclusion for the paper is that **the data, not the architecture or the
training recipe, is the binding constraint.**

### 2026-09-11 — CORRECTION to the 2026-09-10 v2 entry above

Two statements above are wrong and are superseded here (the log is kept as
written; this entry corrects it).

1. **Class breakdown was by PREDICTED class, not true class.** The "calculus 3/3,
   gingivitis 3/4, caries 0/3" split counted each case under the class the model
   predicted. By TRUE class the ten cases are 6 calculus and 4 caries, with **no
   true gingivitis images at all**. v2 by true class: **calculus 6/6, caries 0/4**
   (three of the six calculus passes were misclassified as gingivitis).
2. **"~3/10 -> 6/10" was not a like-for-like comparison.** v1's ~3/10 came from the
   Phase 7 sample on the ORIGINAL test split, which still held illustrations and
   blown-out images. Grad-CAM for both models on the SAME ten photographs
   (docs/paper/figures/fig_gradcam_v1_v2.jpg):

   | | pass | marginal | fail | calculus (true) | caries (true) |
   |---|---|---|---|---|---|
   | v1 | 6 | 2 | 2 | 5/6 | 1/4 (+2 marginal) |
   | v2 | 6 | 1 | 3 | 6/6 | 0/4 |

   On matched photographs de-biasing did **not** measurably change where the model
   looks. The conclusion that the data is the binding constraint stands, and is
   stronger for it.

Wilson 95% intervals (n = 52 test photographs): v2 accuracy 0.635 [0.499, 0.752],
v2 warp stability 0.788 [0.660, 0.878]; v1 accuracy on the clean split 0.885
[0.770, 0.946] (leakage-confounded). Grad-CAM 6/10 -> [0.313, 0.832].

Grad-CAM verdicts in this log were made by the AI assistant, unblinded, with the
prediction visible. They must be re-scored by a dental clinician before
publication.

**Decision (2026-09-11): the classifier is PARKED.** No further retraining on this
corpus; restart only when the Piyarathne et al. dataset clears access review. See
docs/paper/classifier_findings.md for the write-up and the restart runbook.

*OralTwin is a screening aid, not a diagnostic tool — findings must be checked
by a dentist.*

*OralTwin is a screening aid, not a diagnostic tool — findings must be checked
by a dentist.*

## 2026-10-06 - Phase 2 retrain: Piyarathne binary referral (GATE FAILED)

**Task.** `classification.task: binary_referral` - refer = OPMD + OCA,
no_refer = Healthy + Benign. Rationale: OralTwin is a screening aid that never
names a diagnosis, so "does this need a dentist to look at it" is the task it
actually performs; it is also far better powered than the 4-class OCA cell,
which holds 19 test images.

**Split provenance.** Piyarathne, patient-disjoint, 714 patients, built by
`scripts/build_splits.py` and documented in `data/processed/SPLIT_DESIGN.md`.
Train 2099 / val 451 / test 449 (one OPMD image is truncated in the source and
is absent). No patient appears in two splits (`tests/test_splits.py`).

**Config.** Unchanged from Phase 2 otherwise: MobileNetV2 ImageNet init, two
stages (frozen base at lr 1e-3, then last 30 layers at 1e-4), class weights,
dropout 0.3, EarlyStopping patience 5 on val_loss. Images from a 224px cache.

**Trained locally on CPU, NOT on Colab.** The brief asked for a Colab bundle,
but the Piyarathne dataset is CC BY-NC-ND: no derivatives may be redistributed,
and uploading the images (or the 224px cache derived from them) to Google would
be exactly that. Training stayed on the machine; no conversion step was needed,
so the "reproduce the Colab metrics after conversion" check does not apply.
Early stopping fired at stage-b epoch 7, restoring epoch 2.

### Gate results - two of three fail

| Gate | Result | Target | Verdict |
|---|---|---|---|
| (a) performance | accuracy 0.630 [0.585, 0.674] | report honestly | reported |
| | sensitivity 0.491 [0.427, 0.556] | - | **misses half of referrals** |
| | specificity 0.774 [0.714, 0.824] | - | - |
| (b) Grad-CAM on tissue | **6/10** | >= 8/10 | **FAIL** |
| (c) warp stability | 0.902 [0.871, 0.926] | >= 0.90 | PASS (point estimate only) |

PPV 0.691 [0.616, 0.757], NPV 0.596 [0.538, 0.651]. Confusion: TP 112, FP 50,
FN 116, TN 171. **116 of 228 cases needing referral were not referred.**

Breakdown by original category (95% Wilson intervals):

| category | n | handled correctly | note |
|---|---:|---|---|
| Healthy | 109 | 0.798 [0.713, 0.863] | not referred |
| Benign | 112 | 0.750 [0.662, 0.821] | not referred |
| OPMD | 209 | 0.464 [0.398, 0.532] | referred |
| OCA | 19 | 0.789 [0.567, 0.915] | referred - UNDERPOWERED, interval spans 0.35 |

### Gate (b) in detail - stratified and scored blind

10 cases, 5 per true class, categories spread within each class. The panel images
carry only a case number; the true class, prediction and probability were held in
key.json and read only after the verdicts were written. Scored by TRUE class.

| case | true class | category | verdict | where the attention fell |
|---|---|---|---|---|
| 01 | no_refer | Benign | FAIL | the finger retracting the lip, and the nostril |
| 02 | refer | OPMD | PASS | labial / alveolar mucosa |
| 03 | refer | OCA | PASS | intraoral lesion and teeth |
| 04 | refer | OCA | FAIL | lower lip vermilion and the black frame edge |
| 05 | refer | OCA | PASS | carious tooth and surrounding mucosa |
| 06 | no_refer | Healthy | PASS | dorsal tongue |
| 07 | no_refer | Benign | PASS | hard palate |
| 08 | refer | OPMD | FAIL | moustache and chin - the photograph has no intraoral view at all |
| 09 | no_refer | Benign | FAIL | dark background at the frame edge |
| 10 | no_refer | Healthy | PASS | floor of mouth, bleeding onto the lip |

By true class: refer 3/5, no_refer 3/5 - the failures are not concentrated in one
class. Verdicts were made by the AI assistant, unblinded as to its own earlier
work though blind to the model's predictions; a clinician should re-score them.

**Data-quality finding:** case 08 (`S-184-01.jpg`, labelled OPMD) is a
photograph of a closed mouth with no intraoral content. The model called it
no_refer with p=0.92. At least one labelled image in the dataset cannot support
its label.

### Outcome

**The Grad-CAM gate failed (6/10 against >= 8/10), so work stopped there.**
Nothing was tuned to make it pass. External validation on SMART-OM was NOT run -
the brief places it after the gate, never before - and `data/longitudinal` has
not yet been regenerated. `classification.surface_predictions` stays false.

Reading the three gates together: the model is stable under re-framing, but it
refers barely half the cases that need referring and only 6 of 10 heatmaps sit on
oral tissue. It is not a shortcut learner in the Phase 2 sense - no single
provenance cue is doing the work - it is simply weak, and attending to fingers,
frame edges and lips often enough to matter.

*OralTwin is a screening aid, not a diagnostic tool - findings must be checked by
a dentist.*

## 2026-10-06 - OPMD subtype analysis: resolution hypothesis NOT supported

Free re-score of the existing binary-referral model's test predictions against
the Clinical Diagnosis column. No retraining.

**Grouping** (by token, because the column is free text with many spellings -
Leukplakia, Leukopakia, Erythroplkia, Verucouss Ca):
  distinctive patch - leukoplakia, erythroplakia, erythroleukoplakia, PVL /
                      verrucous, oral cancer. A discrete colour change or
                      exophytic mass, which should survive downsampling.
  fine texture      - OSF, OLP / lichenoid / DLE. Striae, blanching and fibrous
                      bands, which downsampling is most likely to erase.
  neither (mixed)   - names both kinds, so it cannot test the hypothesis.

**Per-subtype sensitivity** (test split, n=228 referral images):

| n | sensitivity | group | diagnosis |
|---:|---|---|---|
| 92 | 0.533 [0.431, 0.631] | fine texture | OSF |
| 65 | 0.446 [0.332, 0.567] | fine texture | OLP |
| 19 | 0.789 [0.567, 0.915] | distinctive patch | Oral Cancer |
| 9 | 0.444 [0.189, 0.733] | distinctive patch | Leukoplakia |
| 8 | 0.250 [0.071, 0.591] | fine texture | OLP/LR |
| 4 | 0.500 [0.150, 0.850] | distinctive patch | Erythroplakia |

**Criterion:** distinctive patch 0.543 [0.402, 0.678] n=46 against fine texture
0.472 [0.399, 0.545] n=176. Gap +7.2 points, intervals overlap heavily.
**CRITERION NOT MET - no retrain at 384x384.**

**Sensitivity check that settles it.** Removing Oral Cancer from the patch group
drops it to **0.370 [0.215, 0.558] (n=27), BELOW the fine-texture group's 0.472**.
The apparent patch advantage is carried entirely by frank cancer - the most
grossly abnormal images in the set, which any resolution would show. White, red
and verrucous patches are if anything handled *worse* than striae and blanching.
The resolution hypothesis is not merely unsupported; the data point the other way.

The model is weak across every subtype, with only frank cancer handled at all
well. Nothing was retrained; `surface_predictions` stays false.

*OralTwin is a screening aid, not a diagnostic tool.*

## 2026-10-06 - External validation and housekeeping (Track C closed)

### External validation on SMART-OM

**External validation of a model that failed its acceptance gate; reported for
completeness, not as a performance claim.**

OPMD vs Normal, coarse labels, 2,270 images never seen in training. The
confounded Oral Cancer subset was refused by `assert_scoreable()` as designed.

| metric | value (95% CI) |
|---|---|
| accuracy | 0.915 [0.902, 0.925] (2076/2270) |
| sensitivity | **0.072 [0.038, 0.131]** (9/125 OPMD referred) |
| specificity | 0.964 [0.955, 0.971] (2067/2145) |
| PPV | 0.103 [0.055, 0.185] |

Confusion: TP 9, FP 78, FN 116, TN 2067.

**Internal 0.491 -> external 0.072, a fall of 41.9 points, and the external
interval excludes the internal figure.** This is not a consistently weak model
that transfers its mediocrity: the weak signal collapses almost entirely on
photographs from another clinic. It refers 9 of 125 OPMD cases.

The 0.915 accuracy is an artefact of base rate - SMART-OM is 94.5% Normal, so a
model that says "no referral" to almost everything scores well on accuracy while
being useless at the task. Accuracy must never be quoted for this dataset.

### Longitudinal pairs regenerated, Phase 4 and 5 re-verified

`data/longitudinal` was stale (built from an earlier Track B test split). It was
regenerated from the current split, with delta masks rebuilt, and both validation
suites re-run on the new pairs.

| metric | published | regenerated | drift |
|---|---|---|---|
| pairs aligned | 200/200 | 200/200 | none |
| median corner error | 1.17 px | 1.18 px | none |
| corner error under 10px | - | 98.5% | - |
| **r(confidence, corner error)** | **-0.516** | **-0.387** | **regression; misses the <= -0.50 target** |
| delta recall on injected change | 100% | 100% | none |
| delta area false-change | 1.0% | 0.0% | improved |
| delta colour false-flag | 4.1% | 3.2% | improved |
| delta MAE | 1.75 pp | 1.51 pp | improved |
| out-of-frame caught | 13/13 | 17/17 | none |
| edge false-flag (known weak) | 35.1% | 30.9% | improved |

**The one regression is the alignment confidence-error correlation, -0.387
against a Phase 4 target of <= -0.50.** Alignment *accuracy* is unchanged and
still excellent (median 1.18 px, 98.5% under 10 px); what degraded is the
confidence score's ability to RANK which alignments are worse. Phase 4 always
described that correlation as a ranking signal only, and the refusal logic keys
off the confidence floor rather than the ranking, so the practical effect is
limited - but the published -0.516 should not be quoted without this note. It
reflects sensitivity of a correlation to which 52 images happen to be in the
test split.

### Data-quality findings recorded

Three now stand in the limitations of `data/processed/SPLIT_DESIGN.md`:
S-184-01 (unassessable image), N-226-01 (degenerate zero-area polygon), and the
free-text Clinical Diagnosis column (45 strings for ~12 conditions, with
repeated misspellings).

Track C modelling is closed. The model stays parked and
`classification.surface_predictions` stays false.

*OralTwin is a screening aid, not a diagnostic tool.*

---

## Phase 3 - U-Net segmentation (2026-10-06): GATE FAILED

**Setup.** Two-channel U-Net (oral cavity, lesion), 1,947,010 parameters,
256px, base 16 filters, depth 4. Loss 0.5*BCE + 0.5*Dice. Local CPU training
only - Piyarathne never leaves the machine. Existing patient-disjoint splits,
unchanged: 2,099 train / 451 val / 449 test cached pairs.

**Timing.** Measured before committing, as required: 6.87 min/epoch steady
state, extrapolating to 5h45m worst case. Actual run stopped early at epoch 27
(best epoch 21), 2h27m.

**Results on the held-out test split.**

| channel | Dice [95% CI] | IoU [95% CI] | n |
|---|---|---|---|
| oral cavity | 0.933 [0.926, 0.940] | 0.883 [0.872, 0.893] | 449 |
| lesion | 0.362 [0.332, 0.392] | 0.278 [0.253, 0.304] | 446 |

CIs are bootstrap percentile intervals over images; Dice is a continuous
per-image quantity, not a success count, so Wilson does not apply here.

Lesion Dice is computed over lesion-bearing images only. Averaging in the
Healthy images, where the truth is empty and Dice undefined, would let a
correct all-zero prediction score 1.0 by convention and drag the mean past
the gate without the model ever locating a lesion.

**Gate: FAILED on both halves.**

- numeric: lesion Dice 0.362 vs 0.70 required
- visual: 5 of 10 overlays land on annotated tissue, vs 8 of 10 required

**Diagnosis - the lesion head learned to outline the mouth.**

- Dice(predicted lesion, ground-truth **cavity**) = 0.494. The predicted
  lesion mask resembles the cavity more than it resembles the lesion (0.362).
- r(lesion Dice, lesion/cavity area ratio) = 0.777 over 340 images.
- Stratified: lesions covering <10% of the cavity score 0.201 (n=144); those
  covering >75% score 0.778 (n=14).
- On 109 Healthy test images with no annotated lesion, the model paints a mean
  15.5% of the frame as lesion; only 2.8% predict essentially nothing.

Train lesion Dice 0.364 vs val 0.347 - no generalisation gap. This is not
overfitting; the model learned the wrong target consistently on both splits,
so more epochs or regularisation do not address it.

This is the third shortcut-learning finding in the project, after the Phase 2
Grad-CAM failure and the Ca N acquisition confound. Same signature each time:
a respectable aggregate number resting on an easier correlated target.

**Status: stopped, per the Phase 3 brief. No tuning.** The cavity channel is
the only reusable artefact.

Evidence: `models/logs/segmentation_eval.txt`,
`models/logs/segmentation_per_image.csv`,
`models/logs/segmenter_history.csv`,
overlays in `data/processed/segmentation_overlays/` (git-ignored, CC BY-NC-ND).

---

## Phase 3 follow-up (2026-10-07): the lesion failure is STRUCTURAL, not provenance

The lesion head's failure is recorded above. This note classifies it, because
the distinction changes what could ever fix it.

**The evidence, restated.**

- Dice(predicted lesion, true **cavity**) = 0.494 > Dice(predicted lesion, true
  lesion) = 0.362. The output resembles the mouth more than the target.
- r(per-image lesion Dice, lesion/cavity area ratio) = 0.777 over 340 images.
- 15.5% mean hallucinated lesion area on the 109 healthy test images, where the
  correct answer is nothing at all; only 2.8% predicted nothing.

**Why this is a different kind of defect from the three already recorded.**

The project has three provenance confounds on record: the Ca N acquisition
confound, the illustration/augmentation contamination, and the SMART-OM
burned-in annotations. All three are properties of *these datasets*. A
different dataset, or a cleaner one, removes them. They are fixable by
collecting or filtering data.

This one is not. The easy wrong answer here is intrinsic to the **task
geometry**: a lesion is a small region nested inside a large one, both are
being predicted from the same features, and per-pixel losses pay far more for
the large region than the small one. Any dataset of oral photographs has this
property. Swapping in cleaner data would not remove the gradient incentive to
answer "mouth" when asked "lesion" - the incentive is in the loss, not in the
provenance.

Recorded as a **structural shortcut** to keep it distinct from the provenance
confounds. Conflating the two would suggest the fix is better data, which it
is not.

**FUTURE WORK - deliberately not attempted.**

Reweighting the loss toward the lesion channel, to counteract the area
imbalance, is the obvious next move. It was NOT tried. The gate failed, and
every other result in this project rests on the discipline of stopping when a
gate fails rather than tuning until it passes. Trying the fix that is most
likely to work, immediately after a failure, on the same test split, is how
that discipline erodes - the second attempt would be reported against a split
that has now informed a modelling decision.

If this is picked up later it needs a fresh holdout, not this one.

**What was reused instead.** The cavity channel (Dice 0.933 in-domain) was
wired into the alignment pipeline as the ORB region of interest. See the
region-source A/B below.

---

## Region source A/B (2026-10-07): cavity model REJECTED, heuristic retained

The Phase 3 cavity channel (Dice 0.933 in-domain) was substituted for the HSV
colour heuristic as the ORB region of interest, and measured against it. Both
arms ran the same evaluator functions with only the config key changed.

**Domain note that frames the whole result.** The segmenter was trained on
Piyarathne. The longitudinal pairs are built from the MIO classification test
split - a different dataset, different cameras, different framing. 0.933 is an
IN-domain number and was never evidence about these images.

| metric | heuristic | cavity model | |
|---|---|---|---|
| pairs aligned | 200 | 199 | worse |
| median corner error (px) | **1.18** | 1.41 | worse |
| p90 corner error (px) | **2.78** | 4.64 | worse |
| mean corner error (px) | **1.75** | 2.43 | worse |
| under 10px | **98.5%** | 95.5% | worse |
| r(confidence, error), aligned | **-0.387** | -0.263 | worse |
| r(confidence, error), all pairs | -0.387 | **-0.522** | better |
| delta recall | 100.0% | 100.0% | same |
| colour false-flag rate | 3.2% | **3.1%** | same within noise |
| area false-change rate | **0.0%** | 1.0% | worse |
| MAE measured vs injected | **1.51 pp** | 2.29 pp | worse |
| out-of-frame caught / missed | **17 / 0** | **17 / 0** | same |
| runtime | **0.4 min** | 3.4 min | 8x slower |

**Verdict: the heuristic is kept.** The swap was not an improvement.

**Why, and it is not a fallback artefact.** All 400 images used the cavity
model - no fallback fired anywhere, so this is the model performing as itself,
not a degraded path. The mechanism is simply that the cavity mask is tighter:
it covers 70.4% of the frame against the heuristic's 91.2%, leaving ORB a
median 677 keypoints against 840. Homography estimation wants wide, well-spread
correspondences. A more semantically correct region that yields fewer and more
clustered keypoints is a worse region for this job. Being right about anatomy
and being useful for RANSAC are different things.

### Delta valid-region sweep (step 4)

| valid_region_source | caught | missed | partial | over-flagged | MAE | n(MAE) |
|---|---|---|---|---|---|---|
| frame_intersection | **17** | **0** | **29** | **12** | 1.51 | 171 |
| cavity | 11 | 6 | 112 | 101 | 1.96 | 88 |
| cavity_and_frame | **17** | **0** | 122 | 105 | 1.19 | 78 |

(of 17 pairs that ground truth says are genuinely partly out of frame)

**Verdict: frame_intersection is kept.**

- `cavity` alone **misses 6 of 17** genuinely out-of-frame lesions. This was
  predictable and is now measured: the cavity mask answers "is this mouth?",
  which carries no information about whether a region was photographed twice.
  Out-of-frame detection needs the second question, so the frame intersection
  cannot be replaced by the cavity - only intersected with it.
- `cavity_and_frame` catches all 17, and its MAE of 1.19 pp looks like the best
  in the table. It is not. MAE is computed over pairs that stayed comparable,
  and that variant labels 122 of 200 pairs partial against the frame
  intersection's 29, so its MAE is measured on n=78 instead of n=171 - a
  smaller, easier subset. The apparent gain is a selection effect. The real
  cost is 105 over-flagged pairs: comparisons thrown away on lesions that were
  fully visible.

The `cavity` and `cavity_and_frame` paths are kept in the code and under test
rather than deleted, so the result can be rechecked rather than taken on trust.

Evidence: `models/logs/region_source_comparison.txt`.

---

## Caption layer (2026-10-07)

Describes the photograph. Does not describe the mouth's condition.

**Cavity spot-check decided the design.** Full verdict in
`models/logs/cavity_spotcheck_verdict.txt`. 7 of 10 masks sensible; the three
failures are extreme close-ups already entirely intraoral, where the model
outlines soft tissue only and reports 22-40% against a true ~100%. That error
points the wrong way - it would tell someone already too close to move closer.
The HSV heuristic is worse for this figure: it returns exactly 100.0% on 4 of
10, which is its fallback path, not a measurement.

**Policy adopted instead of the brief's fallback:** the mouth fraction is
printed only when the two independent estimates agree within
`caption.mouth_fraction_tolerance`, and refused with a reason otherwise. The
easier question - is an intraoral view present at all - is answered
confidently, because both sources get it right on all 10.

**In practice the figure is usually withheld:** 4 of 10 example reports print a
fraction, 6 refuse it. That is the honest rate on MIO-type images and is worth
knowing before anyone designs a UI around the number.

**No clinical content, enforced.** `tests/test_caption.py` runs a regex over a
46-term condition vocabulary against every generated sentence on every path.
The vocabulary is deliberately wider than the datasets' labels: the realistic
failure is someone adding a helpful phrase later, not someone wiring up a
classifier. A non-vacuity test confirms the regex fires on "early caries" and
"possible gingivitis".

**The refusal path has no natural example in this corpus.** Alignment succeeds
on 200/200 pairs, so no pair is genuinely uncomparable. The path is covered by
unit tests and demonstrated on real data in
`data/processed/caption_examples/pair_0012_INDUCED_refusal.txt`, produced by
raising `delta.confidence_floor` to 0.70 against that pair's genuine 0.62 and
labelled as induced. It has NOT been exercised by a real alignment failure,
which is a gap in the evidence, not a clean result.

**A defect found and fixed while reading the output:** excluded regions were
being described with generic wording because the lookup keyed on the delta
engine's free-text `reason` rather than its stable `comparability` code. The
engine had supplied a perfectly good explanation and the caption was discarding
it. Found by reading a rendered report, not by a test.

Evidence: `data/processed/caption_examples/` (git-ignored).
