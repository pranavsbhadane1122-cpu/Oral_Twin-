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

## Phase 2b — de-biased retrain (in progress)

Plan: detect and exclude illustrations, collapse augmentation families so
flips/rotations of one base photo count once and never straddle splits,
rebalance classes, retrain with identical hyperparameters so data is the only
variable. Acceptance is Grad-CAM localisation and warp stability, not accuracy;
a lower honest number on clean data is the better model.

*OralTwin is a screening aid, not a diagnostic tool — findings must be checked
by a dentist.*
