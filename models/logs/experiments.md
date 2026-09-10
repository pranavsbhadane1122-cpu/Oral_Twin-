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

*OralTwin is a screening aid, not a diagnostic tool — findings must be checked
by a dentist.*

*OralTwin is a screening aid, not a diagnostic tool — findings must be checked
by a dentist.*
