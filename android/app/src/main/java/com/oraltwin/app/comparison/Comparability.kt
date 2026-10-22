package com.oraltwin.app.comparison

/**
 * Whether two photographs can be compared, and what to do when they cannot.
 *
 * A CODE, NOT A STRING. The Python caption layer keyed its wording off a
 * free-text `reason` field, so rewording the engine's message silently changed
 * which explanation the user saw - and nothing failed, because every string is
 * a valid string. An enum cannot be reworded into a different branch, and a
 * `when` over it stops compiling if a case is added.
 */
enum class ComparabilityCode {
    COMPARABLE,
    PARTIAL_OUT_OF_FRAME,
    UNRELIABLE_LOW_CONFIDENCE,
    ALIGNMENT_FAILED,
}

/**
 * Every threshold the comparison uses, in one place, each carrying the
 * configs/config.yaml line it mirrors.
 *
 * A threshold in two places is a threshold that will eventually disagree with
 * itself, and the disagreement will be invisible: both values are plausible,
 * and only one of them is used.
 *
 * Mirrored as of 2026-10-22.
 */
object ComparisonThresholds {

    /** config.yaml `delta.confidence_floor: 0.3` */
    const val CONFIDENCE_FLOOR = 0.3

    /**
     * config.yaml `delta.min_overlap_frac: 0.60`
     *
     * PROVISIONAL. Unlike delta.relative_color_thresh, which was calibrated
     * against 97 no-change lesions, this is a reasoned default and has not
     * been swept against data. It is an assumption, not a finding.
     */
    const val MIN_OVERLAP_FRAC = 0.60

    /** config.yaml `delta.reliability_high: 0.7` */
    const val RELIABILITY_HIGH = 0.7

    /** config.yaml `delta.reliability_moderate: 0.5` */
    const val RELIABILITY_MODERATE = 0.5

    /**
     * Decide comparability. Pure: no images, no OpenCV, no Android.
     *
     * Separated out precisely so it can be unit-tested on a desktop JVM. The
     * surrounding pipeline cannot be - it needs OpenCV natives - and a rule
     * that only runs on a phone is a rule nobody checks.
     *
     * ORDER MATTERS, and it is not arbitrary:
     *   1. alignment failure first, because without a homography neither of
     *      the other two questions has an answer at all
     *   2. then confidence, because a transform we do not trust makes the
     *      overlap it implies equally untrustworthy
     *   3. then overlap, which is only meaningful once the transform is
     *      believable
     */
    fun classify(
        alignmentFailed: Boolean,
        confidence: Double,
        overlapFraction: Double,
    ): ComparabilityCode = when {
        alignmentFailed -> ComparabilityCode.ALIGNMENT_FAILED
        confidence < CONFIDENCE_FLOOR -> ComparabilityCode.UNRELIABLE_LOW_CONFIDENCE
        overlapFraction < MIN_OVERLAP_FRAC -> ComparabilityCode.PARTIAL_OUT_OF_FRAME
        else -> ComparabilityCode.COMPARABLE
    }
}

/**
 * Plain-language guidance per refusal. No numbers, by construction: these are
 * constants, so there is nothing to interpolate a measurement into.
 */
object RefusalGuidance {

    const val DISCLAIMER = "Have this checked by a dentist."

    fun forCode(code: ComparabilityCode): String = when (code) {
        ComparabilityCode.ALIGNMENT_FAILED ->
            "These two photographs could not be matched up at all. They may show " +
                "different parts of the mouth, or one may be too blurred to use. " +
                "Please retake the newer photo from the same angle as the older " +
                "one, with the same area in view."

        ComparabilityCode.UNRELIABLE_LOW_CONFIDENCE ->
            "These two photographs could not be lined up well enough to compare. " +
                "This is usually lighting or a change of angle. Please retake the " +
                "newer photo in better, even light, holding the phone still."

        ComparabilityCode.PARTIAL_OUT_OF_FRAME ->
            "Too little of the older photograph appears in the newer one to " +
                "compare them fairly. Please retake the newer photo with more of " +
                "the same area in frame."

        ComparabilityCode.COMPARABLE ->
            // unreachable by construction: ComparisonResult.Refused is only
            // built from a non-comparable code. Kept so the when is exhaustive
            // without an else, which is what makes a new code a compile error.
            error("COMPARABLE is not a refusal")
    }
}
