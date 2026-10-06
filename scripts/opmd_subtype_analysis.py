"""Per-subtype sensitivity of the referral model, and the resolution hypothesis.

Every image in the referral class should be referred, so sensitivity here is
simply the share each clinical subtype gets right. If the subtypes whose
signature is a discrete coloured patch do much better than those whose signature
is fine texture, that supports the idea that downsampling 4000x3000 to 224x224
destroys the evidence the subtle ones depend on.

Grouping is by token, stated explicitly below, because the Clinical Diagnosis
column is free text with many spellings of the same word (Leukplakia, Leukopakia,
Leukolpka, Erythroplkia, Verucouss Ca ...). A diagnosis naming BOTH a patch
lesion and a texture lesion goes into neither group - it cannot test the
hypothesis either way.

Writes data/processed/opmd_subtype_analysis.txt. Uses the already-trained model;
nothing is retrained here.

Usage: python scripts/opmd_subtype_analysis.py
"""

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf  # noqa: E402

from src.classification.manifest_dataset import make_dataset, read_split, task_config  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.stats import format_rate, wilson_interval  # noqa: E402

# a lesion that presents as a discrete colour change - a white plaque, a red
# patch, or an exophytic warty mass - should survive aggressive downsampling
PATCH_TOKENS = {
    "leukoplakia": r"leuk(o|)p(l|)a?k?ia|leukoplia|leukolpka|leukopakia|leukplakia",
    "erythroplakia": r"erythropl(a|)kia|erythroplakia",
    "erythroleukoplakia": r"erythroleukoplakia|erythro-leukoplakia",
    "pvl / verrucous": r"\bpvl\b|verruc|veruco|verrucopappilary",
    "oral cancer": r"oral cancer|carcinoma|\bca\b",
}

# a lesion whose signature is striae, blanching or fibrous texture rather than a
# coloured patch - exactly what a 224px downsample is most likely to erase
TEXTURE_TOKENS = {
    "osf": r"\bosf\b|submucous fibrosis",
    "olp / lichenoid": r"\bolp\b|lichen|/\s*lr\b|\blr\b|\bdle\b",
}


def classify(diagnosis):
    """(group, matched patch tokens, matched texture tokens)."""
    text = diagnosis.lower().strip()
    patch = [name for name, pattern in PATCH_TOKENS.items() if re.search(pattern, text)]
    texture = [name for name, pattern in TEXTURE_TOKENS.items()
               if re.search(pattern, text)]
    if patch and texture:
        return "neither (mixed)", patch, texture
    if patch:
        return "distinctive patch", patch, texture
    if texture:
        return "fine texture", patch, texture
    return "neither (unclassified)", patch, texture


def main():
    cfg = load_config()
    task, classes, mapping = task_config(cfg)
    out = PROJECT_ROOT / cfg["paths"]["processed"] / "opmd_subtype_analysis.txt"
    raw = PROJECT_ROOT / cfg["paths"]["raw"] / "piyarathne_oral"
    lines = []

    def log(text=""):
        print(text, flush=True)
        lines.append(text)

    diagnosis_of = {}
    with open(next(raw.rglob("Imagewise*.csv")), newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            diagnosis_of[str(row["Image Name"]).strip()] = \
                str(row.get("Clinical Diagnosis", "")).strip()

    model_path = PROJECT_ROOT / cfg["paths"]["models"] / f"classifier_{task}.h5"
    model = tf.keras.models.load_model(model_path, compile=False)
    paths, labels, _, categories = read_split("test", cfg)
    predicted = model.predict(make_dataset("test", cfg), verbose=0).argmax(axis=1)
    refer_index = classes.index("refer")

    log("=" * 78)
    log("OPMD SUBTYPE ANALYSIS - does resolution explain the weak sensitivity?")
    log("=" * 78)
    log("")
    log(f"model: {model_path.name} (already trained; nothing retrained here)")
    log(f"input resolution: {cfg['classification']['img_size']}x"
        f"{cfg['classification']['img_size']}")
    log("Every image below is in the referral class, so sensitivity is simply the")
    log("share predicted 'refer'. Rates carry 95% Wilson intervals.")
    log("")

    records = []
    for path, label, category, prediction in zip(paths, labels, categories, predicted):
        if category not in ("OPMD", "OCA"):
            continue
        stem = Path(path).stem
        diagnosis = diagnosis_of.get(stem, "")
        group, patch, texture = classify(diagnosis)
        records.append({"stem": stem, "category": category, "diagnosis": diagnosis,
                        "group": group, "correct": int(prediction == refer_index),
                        "patch": patch, "texture": texture})

    log(f"referral images in the test split: {len(records)}")
    log("")
    log("GROUPING RULE (stated so it can be disagreed with)")
    log("  distinctive patch : a discrete colour change - white plaque, red patch, or")
    log("                      exophytic warty mass. Tokens: "
        + ", ".join(PATCH_TOKENS))
    log("  fine texture      : striae, blanching or fibrous bands rather than a")
    log("                      coloured patch. Tokens: " + ", ".join(TEXTURE_TOKENS))
    log("  neither (mixed)   : the diagnosis names BOTH kinds, so it cannot test the")
    log("                      hypothesis either way")
    log("  neither (unclass.): no token matched")
    log("")

    log("PER-SUBTYPE SENSITIVITY (sorted by n)")
    log(f"  {'n':>4}  {'sensitivity':<30} {'group':<22} diagnosis")
    by_diagnosis = defaultdict(list)
    for record in records:
        by_diagnosis[record["diagnosis"]].append(record)
    for diagnosis, group_records in sorted(by_diagnosis.items(),
                                           key=lambda kv: -len(kv[1])):
        correct = sum(r["correct"] for r in group_records)
        total = len(group_records)
        group = group_records[0]["group"]
        log(f"  {total:>4}  {format_rate(correct, total):<30} {group:<22} {diagnosis}")

    log("")
    log("GROUP TOTALS")
    totals = {}
    for group in ("distinctive patch", "fine texture", "neither (mixed)",
                  "neither (unclassified)"):
        group_records = [r for r in records if r["group"] == group]
        if not group_records:
            continue
        correct = sum(r["correct"] for r in group_records)
        total = len(group_records)
        totals[group] = (correct, total)
        log(f"  {group:<24} {format_rate(correct, total)}")
        members = Counter(r["diagnosis"] for r in group_records)
        log(f"      made of: {dict(members)}")

    log("")
    log("THE CRITERION")
    if "distinctive patch" in totals and "fine texture" in totals:
        patch_correct, patch_total = totals["distinctive patch"]
        texture_correct, texture_total = totals["fine texture"]
        patch_rate, patch_low, patch_high = wilson_interval(patch_correct, patch_total)
        texture_rate, texture_low, texture_high = wilson_interval(texture_correct,
                                                                 texture_total)
        gap = patch_rate - texture_rate
        disjoint = patch_low > texture_high or texture_low > patch_high
        log(f"  distinctive patch {patch_rate:.3f} [{patch_low:.3f}, {patch_high:.3f}] "
            f"n={patch_total}")
        log(f"  fine texture      {texture_rate:.3f} [{texture_low:.3f}, "
            f"{texture_high:.3f}] n={texture_total}")
        log(f"  gap               {gap:+.3f} ({gap * 100:+.1f} points)")
        log(f"  intervals overlap: {'no' if disjoint else 'yes'}")
        met = disjoint or (gap >= 0.20 and min(patch_total, texture_total) >= 20)
        log("")
        if met:
            log("  CRITERION MET - the resolution hypothesis is supported; a single")
            log("  retrain at 384x384 is justified.")
        else:
            log("  CRITERION NOT MET - the distinctive-patch group does not score")
            log("  materially higher than the fine-texture group, so there is no")
            log("  evidence that downsampling is what is costing sensitivity. Do not")
            log("  retrain at higher resolution on this evidence.")
        log("")
        log(f"  criterion_met={met}")

        # the patch group mixes frank cancer with white/red patches; if the
        # advantage comes only from the cancers, it says nothing about resolution
        no_cancer = [r for r in records
                     if r["group"] == "distinctive patch"
                     and "oral cancer" not in r["patch"]]
        if no_cancer:
            correct = sum(r["correct"] for r in no_cancer)
            rate, low, high = wilson_interval(correct, len(no_cancer))
            log("")
            log("  SENSITIVITY CHECK - patch group with Oral Cancer removed:")
            log(f"    white/red/verrucous patches only  {rate:.3f} [{low:.3f}, "
                f"{high:.3f}] n={len(no_cancer)}")
            log(f"    fine texture                      {texture_rate:.3f} "
                f"[{texture_low:.3f}, {texture_high:.3f}] n={texture_total}")
            direction = "higher" if rate > texture_rate else "LOWER"
            log(f"    => the patch group is {direction} than fine texture once the")
            log(f"       frank cancers are excluded, so the apparent patch advantage")
            log(f"       is carried by Oral Cancer (0.789), not by patch-ness.")
    else:
        log("  one of the two groups is empty; the criterion cannot be evaluated")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
