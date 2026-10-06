# OralTwin split design

*Built 2026-10-06 by `scripts/build_splits.py` (manifests) and
`src/utils/prepare_classification.py` (Track B image copies). Leakage guarantees
are asserted in `tests/test_splits.py`. This is the document that answers "how do
you know there is no leakage".*

**The three sources are never pooled.** A probe that sees only a 64×64 downsample
identifies which dataset an image came from 82% of the time against a 33% chance
baseline, and Piyarathne alone at 97% recall (`data/processed/provenance_probe.txt`).
Pooling them would let any model score well by recognising the source instead of
the disease, which is exactly how the Phase 2 classifier failed.

Splits are **CSV manifests, not copied image trees** — `data/raw` is read-only and
holds 9 GB, so a manifest naming each image and its split is cheaper and easier to
audit than a second copy of the pixels. Track B is the exception: its existing
pipeline writes resized copies, and it was left unchanged.

---

## Summary

| Source | Role | Split unit | Train | Val | Test |
|---|---|---|---:|---:|---:|
| Piyarathne | primary | **patient** (714) | 2100 | 451 | 449 |
| SMART-OM | external validation only | *none — never trains* | — | — | 2449 usable |
| MIO flat folders | Track A | **pHash cluster** (698) | 534 | 114 | 114 |
| Kaggle | Track B | pHash **family** (199) | 241 | 52 | 52 |

---

## 1. Piyarathne — primary train / val / test

- **Split unit: the patient.** 714 patients, assigned whole to one split; the
  patient id is the image-name prefix (`R-01-03` → patient `R-01`), which matches
  `Patientwise_Data.csv` exactly in both directions.
- **Rule:** 70/15/15 by patient, seed 42. Patients are assigned greedily in
  shuffled order to whichever split has the largest relative deficit for that
  patient's majority category, so category balance is approached without ever
  splitting a patient.
- **Exclusions:** `data/processed/exclusions.csv` applied. None of the 22 fall in
  this dataset, so 3000 of 3000 images are used.

| Split | Images | Patients | Benign | Healthy | OCA | OPMD |
|---|---:|---:|---:|---:|---:|---:|
| train | 2100 (70.0%) | 392 | 524 | 510 | 90 | 976 |
| val | 451 (15.0%) | 161 | 112 | 110 | 20 | 209 |
| test | 449 (15.0%) | 161 | 112 | 109 | 19 | 209 |

**What could not be balanced.** 193 of the 714 patients have images in more than
one category, so no patient-disjoint split can also balance categories exactly.
Patient-disjointness wins, always. The residual imbalance is small — each split
is within about a percentage point of the overall category mix — but **OCA is the
weak point: 19 cancer images in test, from 161 patients.** Any sensitivity figure
computed there will have a confidence interval tens of points wide, and should be
reported with it rather than as a point estimate.

Note the patient counts are 392/161/161 rather than a clean 70/15/15 of 714,
because patients hold between 1 and 18 images each and the balancing works on
images.

---

## 2. SMART-OM — external validation only

- **Split unit: none.** SMART-OM never appears in any training split. A test
  asserts no path containing `SMART-OM` occurs in any train manifest, and that no
  `train.csv` / `val.csv` / `test.csv` exists for it at all.
- **Source level:** the unannotated level only, read through
  `src/utils/smartom.py`. The other levels have the annotations burned into the
  pixels, directly over the lesion the label names.
- **Labels: coarse only — lesion present / absent.** `01. Normal` → absent;
  `02. Variation from normal` and `03. OPMD` → present. Fine-grained types are
  *not* derived from the positional descriptor matching: that mapping agrees with
  the region counts only 89–96% of the time, so a meaningful share of such labels
  would be wrong.

| Category | Images | Patients (SMITA) | Files w/o id | Usable |
|---|---:|---:|---:|---|
| 01. Normal | 2145 | 301 | 31 | yes |
| 02. Variation from normal | 179 | 110 | 1 | yes |
| 03. OPMD | 125 | 63 | 1 | yes |
| **04. Oral Cancer** | **20** | **0** | **20** | **excluded** |

Subject counts are exact where a SMITA id exists; elsewhere the filename stem
stands in, which is a lower bound (two Oral Cancer files share the stem "Ca 2"
despite being different photographs).

---

## 3. The Ca N decision — SMART-OM Oral Cancer is excluded

**Decision: the 20 Oral Cancer images are excluded from every performance figure.**
Not reported separately in a results table — excluded. Piyarathne's 129 OCA images
are the cancer performance estimate.

**Evidence** (`data/processed/can_subset_check.txt`,
`data/processed/confound_baseline.txt`):

- The category contains **no SMITA-named images at all**, so naming scheme and
  cancer label are perfectly confounded. The comparison against SMITA-named cancer
  images could not be run, because there are none.
- **8 of 20 are PNG; not one of the other 2,449 SMART-OM images is.**
- Only 2 of 20 carry EXIF make/model, against 97% of the rest. The study used two
  handsets (iPhone 14, vivo X80); these came from a Nikon D300, an iPhone XR, or
  carry nothing.
- Median resolution is 0.27 MP against 1.36 MP — roughly five times lower.
- A classifier given **no image content at all** — only "is it a PNG" and "does it
  have EXIF" — recovers **40% of the class at 100% precision** under 5-fold
  cross-validation. The rule it learns is simply `is_png → cancer`.

Any metric computed on this subset therefore partly measures how the photograph
was acquired rather than what is in the mouth.

**Retention:** the 20 images stay in the manifest, flagged
`excluded_from_metrics=1` with their reason and **no label**, as a documented
probe for the appendix only. Wherever they are mentioned they carry the wording:
*"20 images from a distinct acquisition stream, one subject each — not a
sensitivity estimate."*

**Enforcement:** `src/utils/smartom.assert_scoreable()` raises
`ConfoundedSubsetError` for any evaluation touching this category unless the
caller passes `allow_confounded_subset=True`. Tested both ways.

---

## 4. MIO flat folders — Track A

- **Split unit: the pHash cluster**, never the image. These folders carry no
  patient ids, so repeat photographs of one mouth are grouped by perceptual hash
  and the whole cluster goes to one split.
- **Threshold: Hamming ≤ 6**, calibrated rather than assumed. At 10 the clusters
  chain — one cluster swallowed 364 images spanning three unrelated categories. At
  6 and below the largest cluster is 8, which matches one patient photographed at
  eight intraoral sites.
- **Exclusions:** 3 SANO images removed (see §6), leaving 762 images in 698
  clusters.

| Split | Images | Clusters | gingivitis | periodontitis | sano |
|---|---:|---:|---:|---:|---:|
| train | 534 (70.1%) | 470 | 219 | 159 | 156 |
| val | 114 (15.0%) | 114 | 47 | 34 | 33 |
| test | 114 (15.0%) | 114 | 47 | 34 | 33 |

**What could not be balanced.** Nothing material — clusters are small (largest 3),
so cluster-level assignment lands close to the targets. The limitation is that
cluster identity is a *proxy* for patient identity: two photographs of the same
mouth that look different enough to exceed Hamming 6 would be treated as separate
subjects and could land in different splits. With no patient ids in these folders
there is no way to rule that out.

---

## 5. Kaggle — Track B

Unchanged de-biasing pipeline (`src/utils/prepare_classification.py`), now with
the manual exclusions applied as well. Split unit is the **augmentation family**
(pHash distance 14), at most 3 images per family, classes capped at 2× the
smallest.

- Exclusions applied: 966 clipped/blown-out, 297 drawn/texture-free, **14 label
  conflict** (the remaining 5 of the 19 were already removed as exact duplicates
  or sit in the YOLO subset, which this pipeline does not read).
- 6,246 raw images → 199 families → **345 images**: train 241, val 52, test 52
  (caries 86/18/18, calculus 59/13/13, gingivitis 96/21/21).

**What could not be balanced.** This corpus is 199 distinct photographs in total;
no split design fixes that. Track B remains a pilot, not a basis for a
performance claim — see `docs/paper/classifier_findings.md`.

---

## 6. Exclusions

`data/processed/exclusions.csv`, 22 images, each row carrying its evidence
(counterpart path, normalised cross-correlation, RANSAC inlier count):

- 3 MIO `SANO` images and 19 Kaggle `oral_diseases` images are **the same
  photographs under contradicting labels** — clinician-labelled healthy in MIO,
  gingivitis in Kaggle (NCC 0.90–0.95, 113–297 inliers). One of the two labels is
  wrong and we cannot tell which, so both copies are dropped rather than trusting
  one source over the other.

A test asserts these appear in no manifest, and — because Track B stores resized
copies whose paths cannot be compared — that no image in the Kaggle splits
perceptually matches an excluded original.

---

## 7. Leakage guarantees (`tests/test_splits.py`, 18 tests)

| Guarantee | How it is checked |
|---|---|
| No patient in two Piyarathne splits | set intersection over all split pairs |
| No image in two Piyarathne splits | path counted across splits |
| No pHash cluster in two MIO splits | set intersection, plus every image of a cluster must share one split |
| SMART-OM never trains | no train manifest may contain `SMART-OM`; no split files may exist for it |
| SMART-OM is unannotated-level only | every manifest path contains `01. Unannotated` |
| Confounded subset is unlabelled and flagged | `lesion_present` empty, `excluded_from_metrics=1`, reason present |
| Confounded subset cannot be scored | `assert_scoreable` raises without the explicit flag |
| Exclusions applied everywhere | absent from all manifests, and perceptually absent from Track B copies |
| Split sizes near target | each within 3 points of 70/15/15 |

---

## Known limitations, stated plainly

1. **OCA in Piyarathne test is 19 images.** Report cancer sensitivity with its
   interval, never as a point estimate.
2. **MIO flat clusters approximate patients.** No patient ids exist there.
3. **SMART-OM subject counts outside SMITA naming are lower bounds**, because the
   filename stem stands in for the subject.
4. **Longitudinal pairs under `data/longitudinal` were generated from an earlier
   Track B test split** and have not been rebuilt. They drive alignment and
   change-detection validation, neither of which is a learned model, so no
   training leakage follows — but they should be regenerated before Track B is
   trained on and evaluated end to end.
5. **Nothing here has been trained.** Training is the next decision, not this one.

*OralTwin is a screening aid, not a diagnostic tool.*
