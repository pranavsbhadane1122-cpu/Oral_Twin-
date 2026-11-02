# Claims audit — every document against docs/CLAIMS.md

Audited 2026-11-02. **Report only. No document was edited.**

Ordered most severe first: FORBIDDEN, CONTRADICTED, UNSUPPORTED, UNSOURCED,
OVERSTATED, STALE, CLEAN.

---

## Scope: what was found, and three things that were not

**The documents are not in the repository.** They are in
`C:\Users\Utente\Downloads`. The repository contains only Markdown.

| Document | Modified | Audited |
|---|---|---|
| `OralTwin_Simplified_Overview.docx` | 21 Aug 2026 | yes |
| `OralTwin_Project_Plan.docx` | 21 Aug 2026 | yes |
| `OralTwin_Development_Status.docx` | 10 Sep 2026 | yes |
| `OralTwin_Development_Status (1).docx` | 11 Sep 2026 | yes (duplicate) |
| `OralTwin_Project_Report.docx` | 24 Sep 2026 | yes |
| `OralTwin_Project_Report (1).docx` | 24 Sep 2026 | yes (duplicate) |
| `OralTwin_Viva_Six_Questions_v1.docx` | 23 Oct 2026 | yes |
| 4 embedded figures (2 per status/report doc) | — | yes, rendered and read |
| `README.md`, `docs/paper/classifier_findings.md`, `android/*.md`, `experiments.md` | — | yes |

### Not found — reported rather than guessed at

**1. There is no OralTwin IEEE paper.** The file matching that description is
`AI_Powered_Screening_amp_Self-Diagnosis_in_Oral_Health.pdf` — a **third-party
paper by other authors**: Kandi Jyothsna, Konda Rakesh Reddy, Jadhav Pranavi
BDS MDS, Uddaraju Meghana, Battu Prudhvi, Akkirala Vishwanth (MLR Institute of
Technology, Hyderabad). It is the **base work** this project builds on, cited
as `[1]` in `OralTwin_Project_Report.docx`. It mentions OralTwin **zero** times
and contains **no Tracks A/B/C** and no extractable "Figure 1" caption.

Auditing it against OralTwin's register would be a category error — it makes
claims about *its own* system. Its abstract is quoted below anyway, because
the project cites it as a foundation and the abstract is squarely diagnostic.

**2. There is no Track A / Track B / Track C anywhere.** Zero occurrences
across all seven documents and the PDF. If a tracks framing exists, it is not
in any file on this machine.

**3. The diagnostic flowchart you described was not found.** Specifically:
MobileNetV2 → "Diagnostic category?" → Healthy / Benign / Oral diseases /
Suspicious — Possible Cancer → "High-Grade?" → Refer for Biopsy, dated
27-09-2026.

Searched: all four embedded figures in the Word files (rendered and read); all
`.png/.jpg/.svg/.drawio/.vsdx/.mmd` in Downloads; `docs/paper/figures/`;
`PROJECT_ARCHIVE.docx` and `PROJECT_ARCHIVE_v2.docx` (both checked — **neither
mentions OralTwin at all**, and neither contains any image). **No file on this
machine is dated 27-09-2026.** The nearest OralTwin artefact is
`OralTwin_Project_Report.docx` at 24-09-2026.

### Search scope, recorded so the three NOT FOUND claims are auditable

A recursive search of `Desktop`, `Downloads`, `Documents` and `OneDrive` for
every `.docx`, `.doc`, `.tex`, `.pdf` and `.pptx` returned **no OralTwin
document beyond the seven above**. Desktop: none. Documents: two unrelated R
package manuals. OneDrive: empty.

Five near-miss files were opened and ruled out rather than assumed irrelevant.
`Literature Review.docx`, `Literature_Review_Revised.docx` and
`Methodology_Log_Secondary.docx` are all about **ICT use in European secondary
schools** (Politecnico di Milano) and mention OralTwin zero times.
`PROJECT_ARCHIVE.docx` and `PROJECT_ARCHIVE_v2.docx` likewise mention it zero
times and contain no images.

Two further PDFs in Downloads are third-party literature, not project
documents: a scoping review of AI/ML in oral cancer detection, and
`paper 1.pdf`.

The words "biopsy", "Diagnostic category", "High-Grade" appear in **no**
OralTwin document. If that flowchart exists it is somewhere I cannot see — a
slide deck, an email, or another machine. **It should be found before anything
is published, because its content would be the most severe item in this
audit.**

---

## FORBIDDEN

Names a condition, assigns a diagnostic category, or routes to a clinical
action.

### F1 — `OralTwin_Simplified_Overview.docx` — the whole document

The most severe item found. It is written in the **present tense throughout**,
describing as working a set of capabilities that were never built or that
failed their gates. Ten instances:

| Line | Claim as written | Why forbidden |
|---|---|---|
| 2 (subtitle) | "A Smartphone-Based Tool to Track Oral Disease and **Flag Cancer Risk** Over Time" | Register: *does not detect cancer or OPMD* — NOT BUILT |
| 4 | "flags anything that looks concerning — **including early warning signs of oral cancer** — so a dentist can take a closer look" | Same; also present tense for a capability that does not exist |
| 6 | "AI scans the photo and **highlights parts of the gums and mouth that look unhealthy**" | Assigns a condition category ("unhealthy"); the segmenter failed its gate |
| 9 | "**Watch for signs of oral cancer** — ... looks for patches, sores, or lumps that resemble early warning signs of oral cancer (called OPMD...)" | Condition named, capability NOT BUILT |
| 13 (heading) | "**Cancer Risk Flagging (New Addition)**" | An entire section describing a non-existent capability |
| 15 | "**Detect suspicious spots** — the AI is trained to recognize patterns dentists look for, such as unusual white or red patches, sores..., or lumps" | "the AI is trained to" — **no such model was ever trained** |
| 17 | "Prioritize what needs attention — ... flags 'this spot changed the most since last visit' **so a dentist can focus their attention efficiently**" | Routes to a clinical action |
| 23 | "U-Net — a model that **segments (outlines) lesions** and gum tissue" | Present tense; lesion segmentation FAILED GATE at 0.362 vs 0.70 |
| 29 | "**Cancer-pattern detection (new)** — an additional recognition layer **trained on** known OPMD visual patterns" | Asserts a trained model that does not exist |
| 11 | "...also notes known links between gum disease and **other health risks such as heart disease**" | Names a body system |

Line 11 and line 19 do carry correct caveats ("never says 'you have cancer'",
"not a diagnostic tool"). **They do not rescue the document** — they sit beside
nine present-tense claims that contradict them, and a reader takes the specific
claim over the general disclaimer.

**Mitigating context, stated fairly:** this file is dated **21 Aug 2026**,
before Phase 2b (10 Sep) and Phase 3 (6 Oct) discovered the failures. It was
accurate-as-intent when written. It is a **plan that reads as a description**,
and the tense is the whole problem.

### F2 — `OralTwin_Development_Status.docx`, embedded figure 2 of 2 ("OralTwin — System Architecture")

Rendered and read. Shows as live, unqualified components:

- **"OUTPUT: change report + heatmaps + `cancer-risk flag` + 'see a dentist' message"** — a cancer-risk flag is a diagnostic category the project does not produce
- **"MobileNetV2 Classifier — healthy / caries / calculus / gingivitis"** — names four conditions; classifier is PARKED
- **"OPMD Cancer-Pattern Layer — white/red patch, sore, lump"** — NOT BUILT
- **"U-Net Segmentation — lesion & gum masks"** — FAILED GATE
- "Rule-Based Risk Stratification ... (oral-systemic links)" — body system

**No box is greyed, marked parked, or marked blocked.** Everything reads as
shipped.

> **This figure has a corrected successor.** The equivalent figure in
> `OralTwin_Project_Report.docx` marks `Classifier [PARKED]`,
> `U-Net segmentation [BLOCKED]`, `OPMD layer [BLOCKED]`, drops the condition
> names, replaces "cancer-risk flag" with "risk band **or a refusal with a
> reason**", and adds a legend: *"Grey = parked or blocked pending data."*
> **The fix already exists. It was not back-ported.** This is the clearest
> instance of the drift you described.

### F3 — `OralTwin_Project_Plan.docx` line 6

> "flags concerning changes (**including OPMD / early oral-cancer warning
> patterns**) for review by a dentist"

Forbidden content, in a document that is explicitly a plan. Lower severity than
F1 because the document's framing is prospective — but the phrase would be
forbidden in any published descendant.

### F4 — third-party base paper, abstract (quoted verbatim, as requested)

> "Abstract—This research work describes a deep learning-based solution that
> uses the MobileNetV2 architecture to **diagnose dental diseases such as
> dental caries, calculus and gingivitis** using intraoral colour pictures. The
> model uses multi-class classification for all the classes both diseases class
> and a healthy class **with high accuracy** while using minimal computer
> resources, giving it a **low-cost alternative to standard diagnostics**.
> Furthermore, a segmentation module which is made using traditional U-Net
> architecture, is included to offer exact localization of damaged regions using
> bounding boxes, **improving diagnostic precision and interpretability**. The
> dataset used to support the model has been carefully chosen to encompass a
> wide range of demographic characteristics, ensuring robustness and
> applicability across populations. This complete method has enormous promise to
> improve dental healthcare by providing efficient, **accurate, and easily
> accessible diagnostics** for common dental problems."

**Not OralTwin's claim to fix** — different authors. Recorded because the
project cites it as `[1]`, its title contains "**Self-Diagnosis**", and its
abstract quotes "high accuracy" with no figure, no sensitivity and no base
rate. Any OralTwin document that inherits its framing inherits this problem.

---

## CONTRADICTED

The register says the opposite.

### C1 — `OralTwin_Simplified_Overview.docx` line 23 and line 6

U-Net "segments (outlines) lesions"; AI "highlights parts ... that look
unhealthy". Register: lesion segmentation **FAILED GATE** — Dice 0.362 against
0.70, 5/10 visual against 8/10, and the model **learned to outline the mouth
rather than the lesion** (Dice against true cavity 0.494 > against true lesion
0.362). The document asserts precisely the capability that was measured and
rejected.

### C2 — `OralTwin_Development_Status.docx` figure 1, Phase 3 "BLOCKED"

Register: Phase 3 was **BUILT and FAILED ITS GATE** on 2026-10-06. "Blocked on
external dataset" says the work never happened. It happened, produced a
measured result, and the result was a failure — which is a materially different
and more useful thing to report.

The same box is stale in the Project Report figure, which also still says
`[BLOCKED]`.

---

## UNSUPPORTED

No evidence in the register at all.

### U1 — `OralTwin_Simplified_Overview.docx` line 15 and 29

"the AI **is trained to** recognize patterns dentists look for"; "an additional
recognition layer **trained on** known OPMD visual patterns". **No such model
was ever trained.** Not parked, not failed — never built. The register lists
OPMD/oral-cancer detection as **NOT BUILT**.

### U2 — `OralTwin_Project_Plan.docx` line 110

> "Definition of Done: **OPMD sensitivity ≥ 85%** on test set"

A target for a capability that was never built. Also appears in
`OralTwin_Development_Status.docx` line 44 and in figure 1 as Phase 6's gate.
As a *plan target* this is legitimate; it is listed here because the nearest
real measurement is **0.072 external sensitivity**, and the two figures should
never appear in the same document without that comparison.

### U3 — `OralTwin_Simplified_Overview.docx` line 17

"flags 'this spot changed the most since last visit'". No ranking or
prioritisation capability exists in the register.

---

## UNSOURCED

Quotes a number on the register's UNSOURCED list.

### S1 — `OralTwin_Project_Report.docx` line 191 — "~85% augmented"

> "The caries class consisted of **roughly 85% augmented copies** of a small
> number of base photographs"

Register, UNSOURCED: `audit_report.txt` retains the numerator (2,382 images in
the augmented folder) but **not the denominator**, so the percentage cannot be
recomputed from the retained artefacts. The underlying finding is sound and
well evidenced; **this specific figure is not checkable**.

Also in `README.md` ("~85% augmented copies") and in
`experiments.md` line 43 — which is where it originates, and which already
hedges it with "~".

### S2 — `README.md` line ~112 — "88/100"

> "the original colour metric compared absolute HSV and so fired on **88/100**
> unchanged pairs"

Register, UNSOURCED: describes a superseded implementation whose output was
never retained. The defect was real; the count cannot be reproduced. Current
figure: **3.2%**.

### S3 — the 46-term count

Not found in any document — it exists only in commit message `4e2eb81` and is
already corrected in the register (**42 terms**, derived from source). Listed
for completeness, as the register requires.

---

## OVERSTATED

The register says `artefact only` or `provisional`; the document presents it as
established.

### O1 — `OralTwin_Project_Report.docx` line 165 — benchmark figures

> "This corresponds to **12.3 pairs per second** with a peak resident memory of
> **63.4 MB** ... **Alignment dominates at 77% of the cost**"

All three are `artefact only` — they live in
`data/processed/deploy/pipeline_benchmark.txt` and were **never written up as a
logged run** in `experiments.md`. The document presents them as settled
results.

**In mitigation, the same line is the most careful in any of these documents:**

> "These are desktop CPU figures; **a phone requires its own measurement before
> any on-device claim is made.**"

That caveat is exactly right and should be the model for the rest.

### O2 — every document quoting `relative_color_thresh` as "calibrated"

`experiments.md` and the figures describe the colour threshold as "calibrated".
The register now classifies it **`artefact only`**: 97 no-change lesions,
p95 = 27.47 — real evidence, in `color_calibration.json`, never logged as a
run. **Nothing in this project is currently `calibrated`.**

### O3 — risk thresholds presented without their provisional status

The 10% / 30% growth thresholds appear in the figures and reports as settled
gates. Register: both **`provisional`** — reasoned defaults, **no clinical
input obtained**. See the new finding below, which makes this worse.

---

## STALE

Was true, superseded. Current value given.

| Document | Location | As written | Current |
|---|---|---|---|
| `Development_Status` fig. 1 | Phase 2 | "Gate: ≥85% accuracy — **met (89.96%)**", status **REOPENED** | 89.96% **DISCARDED**; Project Report's figure already says "met, later invalidated" and PARKED |
| `Development_Status` fig. 1 | Phase 2b | status **RUNNING** | Complete and **FAILED** (6/10, warp 0.7885) |
| `Development_Status` fig. 1 | Phase 8 | status **PENDING** | **CLOSED** |
| `Development_Status` line 24 | gates table | "≥85% test accuracy — **met at 89.96%**" | Reclassified as dataset separability |
| both figures | Phase 4 | "**1.17 px** median error" | **1.18 px** |
| both figures | Phase 5 | "colour false-flag **4.1%**" | **3.2%** |
| both figures | Phase 3 | **BLOCKED** | Built, **FAILED GATE** (0.362) |
| `Project_Report` fig. 2 | U-Net box | **[BLOCKED]** | Built, FAILED GATE |
| `Development_Status` line 104 | Grad-CAM | "roughly **seven of ten**" failing | Consistent with ~3/10 on-target; the later matched-photograph comparison found **6 pass / 2 marginal / 2 fail** for v1 and **6 / 1 / 3** for v2 |
| all docs | corner-error correlation | not mentioned | **−0.387**, a regression from the published −0.516 that misses the ≤ −0.50 target. **No document records this.** |

---

## CLEAN

Checked and found sound. As useful to know as the rest.

### `OralTwin_Viva_Six_Questions_v1.docx` — CLEAN, and the best document here

The only document with **zero** occurrences of "diagnos*" and one incidental
mention of "cancer". Verified correct against the register on every number:

| Claim | Verdict |
|---|---|
| "0.362 Dice. Dice(predicted lesion, true cavity) = 0.494 was higher than Dice(predicted lesion, true lesion) = 0.362 — the model had learned to find the mouth, not the lesion." | **SUPPORTED**, exactly |
| "it is why a 99.51% accuracy was discarded: the base rate was 99.19%" | **SUPPORTED** — the only document that quotes 99.51% *with* its base rate |
| "91.5% external accuracy on a set that was 94.5% negative — with 7.2% sensitivity. The accuracy figure is the publishable one. The sensitivity figure is the true one." | **SUPPORTED** — the only document that pairs 91.5% with both the base rate and the sensitivity |
| "Passed (0.933 Dice)" for the cavity channel | **SUPPORTED** |
| "The configuration flag classification.surface_predictions is set to false" | **SUPPORTED** |
| "a native Android application written in Kotlin ... **It is not yet compiled** — the development machine has no Android toolchain installed, so **no line of the application has yet been executed**" | **SUPPORTED** — the only document that states this, and it states it exactly right |

**No document other than the Viva makes any Android claim at all**, so there is
no "the app works" overstatement anywhere. The register's NOT BUILT status is
not contradicted by any document.

### `OralTwin_Project_Report.docx` — CLEAN on its central claims

Its handling of the headline failure is careful and correct:

| Claim | Verdict |
|---|---|
| "The 89.96% figure is therefore reported as a measure of dataset separability, not clinical accuracy, and the classifier is parked." | **SUPPORTED** |
| "Objective 4 was blocked by data access and **is not claimed**." | **SUPPORTED** |
| "the sample is small, unstratified, and **was scored by the developer with each model's prediction visible rather than blinded. A clinician should re-score a stratified, blinded sample before any attention figure is published.**" | **SUPPORTED** — correctly disclaims clinical validation, the only document to do so |
| "a phone requires its own measurement before any on-device claim is made" | **SUPPORTED** |

Its remaining problems are O1 (`artefact only` figures), S1 (~85%), and the two
stale figure boxes — not its prose.

### Repository Markdown — CLEAN

| File | Verdict |
|---|---|
| `docs/CLAIMS.md` | the authority |
| `models/logs/experiments.md` | CLEAN — carries its own corrections, including the 2026-09-11 entry retracting two of its own claims |
| `docs/paper/classifier_findings.md` | CLEAN — limitations section records the structural shortcut and the in-domain-only cavity figure |
| `android/README-android.md` | CLEAN — states NOT BUILT plainly |
| `android/PORT-NOTES.md` | CLEAN — twelve judgement calls, no claims |
| `README.md` | CLEAN except S2 (88/100) and S1 (~85%) |

---

## Two new findings from Part 1

Not part of the document audit, but they bear on what the documents may claim.

### N1 — the area-change figure is far less precise than any document implies

At the median alignment corner error of **1.18 px**, a coherent boundary
misplacement on a median **1,792 px** region (radius ≈ 24 px, circumference
≈ 150 px) moves the measured area by **≈ 177 px — about 10% of the region**. At
the p90 error of 2.78 px it is **≈ 23%**.

That bound is pessimistic — it assumes the whole boundary shifts coherently,
whereas homography error is smallest near the image centre where these regions
sit — so it is an **upper bound, not a point estimate**.

Even so: **`risk.growth_moderate_pct` is 10%**, which places the
"worth mentioning" threshold **at or near the measurement's own noise floor**.
A measured 10% change may be indistinguishable from no change. The threshold is
already `provisional` in the register; this suggests it needs a sweep against
alignment error, not only clinical input.

Display precision is now capped at one decimal (71.48% → **71.5%**). Two
decimals asserted a hundredth of a percent on a quantity whose uncertainty is
nearer ten percent.

### N2 — the 0.99 floor is not a configured threshold

You asked which check uses it. **None.** It is `INDUCED_FLOOR = 0.99` in
`scripts/render_report_baselines.py`, a baseline-harness constant chosen
specifically to force a refusal, because no pair in `data/longitudinal` fails
alignment naturally (200/200 succeed). The shipped
**`delta.confidence_floor` is 0.3**.

Your reasoning was right: against a score correlating with corner error at only
−0.387, a 0.99 floor would refuse nearly everything. That is why it was chosen
for the harness, and it should never appear in a real report. It no longer can
— refusals now render no number derived from the images.

---

## Summary

| Verdict | Count | Concentrated in |
|---|---|---|
| FORBIDDEN | 4 (one covering 10 lines) | `Simplified_Overview`, `Development_Status` fig. 2 |
| CONTRADICTED | 2 | `Simplified_Overview`, both status figures |
| UNSUPPORTED | 3 | `Simplified_Overview`, `Project_Plan` |
| UNSOURCED | 3 | `Project_Report`, `README.md` |
| OVERSTATED | 3 | `Project_Report`, all figures |
| STALE | 10 | both status figures, `Development_Status` |
| CLEAN | 8 documents/files | `Viva`, `Project_Report` prose, all repository Markdown |

**The pattern is chronological, not careless.** Severity tracks the file date
almost exactly: the 21 Aug planning documents are the worst, the 24 Sep report
is largely sound, and the 23 Oct Viva is clean. The documents did not drift
apart through sloppiness — **they were each accurate when written, and the
project then found things out.** What is missing is back-propagation: the
Project Report already contains corrected versions of both figures, and nobody
carried them back.

**The single highest-value action** is not editing prose. It is replacing the
two figures in `OralTwin_Development_Status.docx` with the ones that already
exist in `OralTwin_Project_Report.docx`, which removes F2, C2 and four STALE
rows at once.

**Before anything is published**, the flowchart described in the brief
(MobileNetV2 → "Diagnostic category?" → ... → Refer for Biopsy) must be found.
It is not on this machine. Its content would be the most severe item in this
audit, and it cannot be fixed while it cannot be located.

---

*OralTwin is a screening aid, not a diagnostic tool. It does not identify
dental conditions. Any concern should be checked by a dentist.*
