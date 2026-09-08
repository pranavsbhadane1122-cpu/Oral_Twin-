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
