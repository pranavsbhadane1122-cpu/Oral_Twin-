# Alignment port: Python → Kotlin, every judgement call

A port, not a rewrite. Same algorithm, same parameters, same rejection rules.
This file lists every place where the translation was not mechanical, because
a silent reinterpretation is the thing most likely to break it.

Source: `src/alignment/{features,homography,confidence}.py` and the
`alignment:` block of `configs/config.yaml`, as of 2026-10-22.

---

## 1. RANSAC is randomised, so exact agreement is impossible

**The big one.** `findHomography(..., RANSAC, ...)` samples minimal sets at
random. Python and Kotlin will not agree exactly, and **Python does not agree
with itself** across runs either — it is only stable here because the reference
figures were recorded from one particular run.

I did **not** seed the RNG on either side to force agreement. `cv2.setRNGSeed`
and `Core.setRNGSeed` both exist, but the two implementations consume random
numbers in their own order, so equal seeds would not produce equal sampling
and the matching numbers would be a coincidence dressed up as a proof.

**Consequence for the gate:** the within-2× criterion is the right shape for
this. Treat a small divergence as expected and a large one as a real defect.
Re-running the self-check will shift the numbers slightly; that is the
algorithm, not a bug.

## 2. Decoding: `Imgcodecs.imdecode`, never `BitmapFactory`

`cv2.imread` returns **BGR**, and the HSV mouth thresholds were derived
against BGR input. `BitmapFactory` plus `Utils.bitmapToMat` yields **RGBA**,
which swaps the red and blue channels and therefore moves every hue. The
teeth band would still roughly work; the reddish-tissue bands would select
the wrong colours entirely.

This would not crash. It would quietly select a different mouth region and
show up as a modest accuracy loss. This project has already been bitten once
by exactly this class of bug, in `compare_models.py`, where it inverted a whole
comparison table.

`SelfCheck.readAssetBgr` therefore decodes the raw asset bytes with
`Imgcodecs.imdecode(..., IMREAD_COLOR)`, which reproduces `cv2.imread`.

## 3. `findHomography` overload and its defaults

Python calls `cv2.findHomography(src, dst, cv2.RANSAC, thresh)` and takes
OpenCV's defaults for `maxIters` (2000) and `confidence` (0.995).

Kotlin uses the matching 4-argument-plus-mask overload rather than passing
those two explicitly. **Deliberate:** writing them out would pin them to
today's values, so if OpenCV ever changed a default, Python would follow and
Kotlin would not, and the two would silently diverge. Taking the default on
both sides keeps them tied together.

## 4. Argument direction into `findHomography`

Python passes `src = points2`, `dst = points1`, so H maps visit2 into visit1.
Reversing them yields the inverse, which is still a valid-looking homography
that passes every plausibility check and warps the wrong way. Preserved
exactly, and commented at the call site.

## 5. Largest connected component

Python: `np.where(labels == largest, 255, 0).astype(np.uint8)`.
Kotlin: `Core.compare(labels, Scalar(best), out, CMP_EQ)`.

`compare` emits 255 where equal and 0 elsewhere, which is the same 0/255
convention. The `argmax` over `stats[1:, CC_STAT_AREA]` becomes an explicit
loop from label 1, skipping the background, with the same tie-breaking (first
maximum wins).

## 6. Variance of the Laplacian

Python uses numpy `.var()`, which is the **population** variance (ddof = 0).
Kotlin uses `Core.meanStdDev` and squares the standard deviation;
`meanStdDev` also returns the population figure. These agree exactly rather
than approximately. Had I used a sample variance (ddof = 1), every blur score
would be slightly off and the confidence would drift with it.

## 7. Scalar construction from config triples

OpenCV's `Scalar(double[])` only clones arrays of length 4; anything else
falls through to `set()`, which happens to handle short arrays correctly
today. Rather than depend on that, `scalar3()` builds the three-component
Scalar element by element.

## 8. The mouth region: heuristic only, and why that is not a compromise

The brief said to port the HSV heuristic and not the cavity model. Two
independent reasons, and it is worth being clear that the licence is only the
first:

1. The cavity model is trained on Piyarathne (CC BY-NC-ND) and cannot ship.
2. **It was measured as worse for this job.** It covers 70.4% of the frame
   against the heuristic at 91.2%, leaves ORB a median 677 keypoints against
   840, and raises median corner error from 1.18 to 1.41 px. The Python
   default is `hsv_heuristic` on that evidence.

So the Android port is not a licence-constrained downgrade. It matches the
Python's own default path.

`Features.mouthMask` carries the extension point, with the contract a
replacement must satisfy and a warning not to assume a model would be better.

**Not ported:** `features.region_mask`, the dispatcher that chooses between
the two sources and records which was used. With only one source it would be
a wrapper around a constant. The extension point says to reinstate it, with
its fallback reporting, if a second source ever exists.

## 9. Per-image region source is reported, but `full_frame` is inferred

Python knows it fell back to the whole frame because `mouth_mask` returns a
full-white mask. Kotlin reports `full_frame` when the mask covers ≥ 99.9% of
the frame rather than threading a flag out of `mouthMask`.

**Judgement call.** The two agree in every case except the vanishingly
unlikely one where the heuristic genuinely selects ≥ 99.9% of the frame
without having failed. That distinction does not affect alignment at all —
only the label — so the simpler signature won.

## 10. Corner error uses the inverse of the stored ground truth

`ground_truth.json` stores H mapping **visit1 → visit2**; alignment estimates
**visit2 → visit1**. `evaluate_alignment.py` compares against
`np.linalg.inv(gt["homography"])`, and `SelfCheck` does the same with
`Core.invert`. Comparing against the stored matrix directly would produce
large, meaningless errors that look like a broken port.

## 11. Pairs with fewer than two neighbours

Python's ratio test iterates `(p for p in knn if len(p) == 2)`, silently
dropping any match that did not return two neighbours. Kotlin does the same
with `if (candidates.size != 2) continue`. Easy to miss; it changes the
denominator of the inlier ratio.

## 12. Rounding in the breakdown

Python rounds the breakdown terms to 4 decimal places for its JSON output.
Kotlin keeps full precision in `Confidence.Breakdown` and rounds only when
formatting for the screen. The rounding was presentational in Python — it does
not feed back into the score — so carrying it into the data model would have
been copying an artefact of serialisation.

---

## Not ported in M2

`align.py`'s file I/O and JSON writing, and `warp_visit2` beyond a thin
helper. M2 is the algorithm and its self-check; capture and storage are M1/M3.

## Test pair provenance

The five bundled pairs are **not** Piyarathne-derived. Verified three ways:

1. All 200 longitudinal pairs have a `source_image` under
   `data/processed/classification/test`, whose classes (calculus, caries,
   gingivitis) come from `data/raw/oral_diseases` — the Kaggle set — not
   `data/raw/piyarathne_oral`.
2. `scripts/bundle_android_testpairs.py` re-checks each pair at bundling time
   and refuses rather than trusting the above.
3. `android/app/src/main/assets` was added to `licence.publishable_dirs`, so
   `tests/test_licence_guard.py` now perceptually hashes all 10 bundled images
   against the restricted dataset. They pass.

The image files are nonetheless **git-ignored**. `data/raw/LICENCES.md` says to
check each Kaggle dataset page before redistributing anything derived from
them, and that check has not been done — committing them to a public
repository would be redistribution. Run
`python scripts/bundle_android_testpairs.py` to produce them locally. The
manifest, which holds only numbers, is tracked.
