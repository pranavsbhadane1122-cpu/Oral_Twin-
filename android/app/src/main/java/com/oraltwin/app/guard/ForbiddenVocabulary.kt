package com.oraltwin.app.guard

/**
 * GENERATED FILE - DO NOT EDIT BY HAND.
 *
 * Generated from tests/test_caption.py (CONDITION_WORDS) by
 * scripts/generate_kotlin_vocabulary.py. That Python list is the single
 * authority for this project; this file is a derived copy, because Kotlin
 * cannot import it.
 *
 * Editing this file by hand reintroduces exactly the drift it exists to
 * prevent. Change the Python list and re-run the generator instead.
 *
 * ForbiddenVocabularyTest asserts this matches the Python source term for
 * term, so the two cannot part company silently. The count is NOT written
 * here: it is derived on both sides, because a count typed by a person is
 * a claim that rots - this one was reported as 46 and is in fact the size
 * of the list below.
 */
object ForbiddenVocabulary {

    /** Condition terms that must not appear in any user-facing string. */
    val TERMS: List<String> = listOf(
        "caries",
        "calculus",
        "gingivitis",
        "leukoplakia",
        "erythroplakia",
        "lichen planus",
        "oral cancer",
        "carcinoma",
        "opmd",
        "malignant",
        "premalignant",
        "dysplasia",
        "tumour",
        "tumor",
        "neoplasm",
        "ulcer",
        "abscess",
        "cavity decay",
        "decay",
        "periodontitis",
        "gum disease",
        "infection",
        "inflammation",
        "lesion",
        "plaque",
        "tartar",
        "stomatitis",
        "candidiasis",
        "thrush",
        "cyst",
        "polyp",
        "melanoma",
        "biopsy",
        "benign",
        "diagnos",
        "disease",
        "healthy mouth",
        "suspicious",
        "severe",
        "mild case",
        "stage ",
        "grade ",
    )

    /**
     * True if [text] contains any forbidden term on a word boundary.
     *
     * Word boundaries matter: "ok" is a substring of ordinary words, and a
     * naive contains() would reject "looking" and "token".
     */
    fun offendingTerm(text: String): String? =
        TERMS.firstOrNull { term ->
            Regex("\\b" + Regex.escape(term.trim()) + "\\b",
                  RegexOption.IGNORE_CASE).containsMatchIn(text)
        }

    fun isClean(text: String): Boolean = offendingTerm(text) == null
}
