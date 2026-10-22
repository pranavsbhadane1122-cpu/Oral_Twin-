package com.oraltwin.app.export

/**
 * Every fixed word that can appear in an exported file.
 *
 * WHY THESE ARE TEMPLATES AND NOT SENTENCES BUILT AT THE CALL SITE.
 *
 * An exported file outlives the app that made it. It gets forwarded, printed
 * and read by someone - possibly a dentist - who has no way to ask what any of
 * it meant. So the export carries no measurement at all: no confidence value,
 * no overlap fraction, no area, no percentage. A number in a file like this
 * will be read as a finding, and there is no finding.
 *
 * Collected in one object so a unit test can enumerate them and assert that
 * none contains a digit, a percent sign, or clinical vocabulary. Strings built
 * inline inside a renderer cannot be enumerated, and a rule that cannot be
 * checked is a rule that will quietly stop holding.
 *
 * THE ONE EXCEPTION, STATED PLAINLY: the two photograph DATES are rendered
 * into the file, and a date contains digits. A date identifies which
 * photograph is which; it is not a measurement of anything and nothing can be
 * computed from it about the mouth. The templates below contain no digits, and
 * the date is passed in separately - so the test over these constants stays
 * honest rather than being weakened to accommodate it.
 */
object ExportStrings {

    /**
     * Rendered inside the image itself, never in the share text.
     *
     * Share text is stripped by some apps, rewritten by others, and simply not
     * read by most. A disclaimer that can be separated from the picture is a
     * disclaimer that will be.
     */
    const val DISCLAIMER =
        "Taken with a phone camera and compared automatically. This is not a " +
            "diagnosis and not a medical assessment. Have this checked by a dentist."

    const val TITLE = "OralTwin photo comparison"

    const val LABEL_OLDER = "Earlier photo"
    const val LABEL_NEWER = "Later photo"
    const val LABEL_OVERLAY = "The earlier photo lined up over the later one"

    /** What was compared, in plain language and without any result. */
    const val COMPARED_LINE =
        "These two photographs of the same mouth were lined up with each other " +
            "so they can be looked at side by side."

    /** The refusal variant. Same file, no overlay, no hedging. */
    const val REFUSED_LINE = "These two photos could not be compared reliably."

    const val REFUSED_EXPLANATION =
        "The app could not line these two photographs up well enough, so it has " +
            "not produced a combined view. The photographs themselves are included."

    const val PROJECT_NOTE =
        "Made by a student research project. It does not identify dental conditions."

    /** Share-sheet subject. Kept free of findings for the same reason. */
    const val SHARE_SUBJECT = "OralTwin photo comparison"

    /**
     * Share-sheet body. Deliberately thin: the disclaimer lives in the image,
     * and this text may be discarded by whatever app receives it.
     */
    const val SHARE_TEXT =
        "Two photos compared with OralTwin. The note inside the image explains " +
            "what it is and what it is not."

    /** Enumerated for the test that checks all of them at once. */
    val all: List<String> = listOf(
        DISCLAIMER, TITLE, LABEL_OLDER, LABEL_NEWER, LABEL_OVERLAY,
        COMPARED_LINE, REFUSED_LINE, REFUSED_EXPLANATION, PROJECT_NOTE,
        SHARE_SUBJECT, SHARE_TEXT,
    )

    /** The strings used by the comparable variant of the export. */
    val comparableVariant: List<String> = listOf(
        TITLE, LABEL_OLDER, LABEL_NEWER, LABEL_OVERLAY, COMPARED_LINE,
        PROJECT_NOTE, DISCLAIMER,
    )

    /** The strings used by the refusal variant. No overlay label, no result. */
    val refusedVariant: List<String> = listOf(
        TITLE, LABEL_OLDER, LABEL_NEWER, REFUSED_LINE, REFUSED_EXPLANATION,
        PROJECT_NOTE, DISCLAIMER,
    )

    /** Date format for the photo labels. Contains no digits itself. */
    const val DATE_PATTERN = "d MMMM yyyy"
}
