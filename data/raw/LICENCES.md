# Licences for the raw datasets

`data/raw` is READ-ONLY. Nothing in this directory is modified; every pipeline
writes to `data/processed`. Licence terms below govern what may be done with
each dataset and, importantly, what may be redistributed.

## mio/ — MIO: My Intraoral Images

- Source: Zenodo record **22047269**
- Licence: **CC-BY-4.0**
- Terms: use, share and adapt permitted, including commercially, with
  attribution to the original authors and an indication of any changes.
- Practical effect here: derived images (resized copies, annotated renders,
  figures in the paper) may be published provided the dataset is credited.

## piyarathne_oral/ — Piyarathne et al., oral cavity dataset

- Source: Zenodo record **10664056** (Oral Oncology 2024;156:106946)
- Licence: **CC BY-NC-ND 4.0** — academic, non-commercial use only,
  **no redistribution of the images**.
- Terms in practice:
  - Non-commercial research use only. If OralTwin is ever taken commercial,
    this dataset cannot come with it.
  - **No derivatives may be redistributed.** Resized or cropped copies,
    annotated renders and figures containing these images must not be
    published or shared outside the project without the authors' permission.
  - Attribution to Piyarathne et al. is required wherever the dataset is
    described.
- Consequences for this project:
  - Do not put images from this dataset into the repository, the demo
    screenshots, the manuscript figures, or any published artifact.
  - Sample renders produced during the audit stay under `data/processed`,
    which is git-ignored, and must not be committed or shared.
  - A model trained on these images is arguably a derivative; clear the
    intended use with the authors before any release.

## Previously present (for completeness)

- `oral_diseases/` — Kaggle "Oral Diseases" (salmansajid05).
- `oral_cancer/` — Kaggle "Oral Cancer (Lips and Tongue) images" (shivam17299).

Both were downloaded from Kaggle under their respective dataset terms; check
each dataset page before redistributing anything derived from them.
