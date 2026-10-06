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
