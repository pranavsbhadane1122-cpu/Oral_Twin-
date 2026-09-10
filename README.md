# OralTwin

OralTwin is a smartphone-based AI screening aid that helps track oral health over time from patient-taken photos. Across visits, it classifies common oral conditions (caries, calculus, gingivitis), segments regions of interest, aligns images longitudinally to detect changes, and flags potential oral-cancer risk for early attention. OralTwin is a **screening aid, not a diagnostic tool** — any finding it surfaces should be checked by a dentist.

## Project structure

```
oraltwin/
├── data/
│   ├── raw/            # Original images (read-only, never modified)
│   ├── processed/      # Preprocessed / augmented images
│   └── longitudinal/   # Per-patient visit series
├── src/
│   ├── classification/     # Oral condition classification
│   ├── segmentation/       # Lesion / region segmentation
│   ├── cancer_detection/   # Oral-cancer risk flagging
│   ├── alignment/          # Cross-visit image alignment
│   ├── delta/              # Change detection between visits
│   ├── explainability/     # Model explanations (e.g. Grad-CAM)
│   ├── risk/               # Risk scoring
│   └── utils/              # Shared utilities (config loading, etc.)
├── models/         # Trained model weights
├── notebooks/      # Experiments and analysis
├── app/            # Streamlit app
├── tests/          # Tests
├── configs/        # config.yaml — all paths and hyperparameters
├── requirements.txt
└── README.md
```

All paths and hyperparameters live in `configs/config.yaml` — never hard-coded.

## Plan deviations

- **Phase 4 (alignment) is being built before Phase 3 (segmentation).** Phase 3's
  best training data — the COCO lesion annotations of the Piyarathne et al. Zenodo
  dataset — is pending restricted-access approval. Alignment only needs the
  simulated longitudinal pairs, which exist and passed their tests, so it was
  pulled forward (decided 2026-09-10).
- **Phase 5 (delta / change detection) is validated against SYNTHETIC lesion masks**
  derived from the longitudinal ground truth, for the same reason: Phase 3's
  segmentation model has no training data yet. `src/delta/compare.py` takes any
  binary/instance mask, so real U-Net masks plug in unchanged once Phase 3 lands —
  only the mask source changes, not the comparison code (decided 2026-09-10).
- **Phase 5 Definition of Done was amended after the first validation run**
  (2026-09-10). The original area targets passed, but validation exposed two
  design defects, so two acceptance criteria were added:
  (a) colour false-flag rate must be <= 10% on no-change pairs — the original
  colour metric compared absolute HSV and so fired on 88/100 unchanged pairs,
  reacting to camera lighting rather than tissue;
  (b) lesions lying partly outside the newer photo's frame must be labelled
  non-comparable and never reported as an area change — previously a lesion
  pushed off-frame read as up to -50% "shrinkage" on a no-change pair.
- **Phase 7 (explainability + rule-based risk) was built before Phases 3 and 6**,
  which remain blocked on the restricted-access Zenodo dataset. Grad-CAM runs on
  the existing 3-class classifier and the risk module is rule-based, so neither
  needs the missing data (decided 2026-09-10). The risk rules deliberately
  **exclude the edge/compactness metric**, which is known to be
  resampling-sensitive (35% false-flag rate) and must not reach user-facing
  output until reformulated.
