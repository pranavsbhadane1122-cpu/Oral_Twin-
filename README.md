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

## Known issues

- **The 3-class classifier (v1, 89.96% test accuracy) learned shortcuts, not
  pathology.** Grad-CAM inspection in Phase 7 (2026-09-10) found that only about
  3 of 10 sampled cases put their attention on teeth or gum tissue; most caries
  cases attended to lips, image corners, or plain background, and one case
  attended to the blank area of a cartoon illustration. A small warp of the same
  photo flipped one prediction from p=0.99 to p=0.00.
  Root-cause hypothesis under test in Phase 2b: the caries class is roughly 85%
  augmented copies of a small number of base photos and mixes cartoon
  illustrations with real clinical photos, so the three classes are separable by
  image *provenance* (crop style, borders, illustration-vs-photo, lighting
  signature) without learning any dental feature.
  **Consequence: 89.96% measures dataset separability, not clinical accuracy.**
  The v1 classifier must not ship and its Grad-CAMs must not be shown to users.
- **The corpus contains only ~199 distinct base photographs.** Phase 2b
  de-biasing collapsed the 6246 raw images of the three source folders into 199
  augmentation families; after capping and rebalancing, 345 usable images remain
  (241 for training). The de-biased v2 classifier scores 0.6346 (95% CI
  0.499-0.752) with 0.79 warp stability, missing both Phase 2b acceptance gates.
  On the same ten photographs, v1 and v2 both put Grad-CAM attention on tooth or
  gum tissue in 6/10 cases, so de-biasing did not measurably change where the
  model looks; v2 fails on every true-caries case (0/4).
  **The data, not the architecture, is the binding constraint**; a trustworthy
  classifier needs a larger, less duplicated corpus. See
  `models/logs/experiments.md` (including its 2026-09-11 correction) and
  `docs/paper/classifier_findings.md`.
- **Status: the classifier is parked (2026-09-11).** It will not be retrained on
  this corpus and its predictions should not be presented as findings. The
  alignment, change-detection, explainability and risk modules do not depend on
  it and continue. The restart runbook for the Piyarathne et al. dataset is in
  `docs/paper/classifier_findings.md`.

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
