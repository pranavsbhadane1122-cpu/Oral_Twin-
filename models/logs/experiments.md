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

*OralTwin is a screening aid, not a diagnostic tool — findings must be checked
by a dentist.*
