package com.oraltwin.app.capture

import com.oraltwin.app.data.PhotoEntity
import kotlin.math.exp

/**
 * Guided capture: framing help only.
 *
 * The reference photograph is drawn over the live preview so the user can line
 * the new one up with it. That is the whole value, and it costs no
 * computation. Nothing here judges image quality and nothing here is clinical.
 *
 * THE REFERENCE LINK IS SESSION STATE AND IS NEVER PERSISTED. A photograph
 * taken with guidance and one taken without are byte-identical in shape - same
 * file path, same Room row, no extra column, no "captured with guidance" flag.
 * If they differed, the comparison layer would acquire a second code path that
 * nothing has measured, and the first time it behaved differently nobody would
 * know which path produced it.
 */
data class GuidedSession(
    /** The earlier photograph being matched against. Not written anywhere. */
    val reference: PhotoEntity,
    /** Ghost opacity, 0f..MAX_GHOST_OPACITY. */
    val ghostOpacity: Float = DEFAULT_GHOST_OPACITY,
) {
    init {
        require(ghostOpacity in 0f..MAX_GHOST_OPACITY) {
            "ghost opacity must be within 0..$MAX_GHOST_OPACITY"
        }
    }

    fun withOpacity(value: Float): GuidedSession =
        copy(ghostOpacity = value.coerceIn(0f, MAX_GHOST_OPACITY))

    companion object {
        /**
         * Capped at half. Above this the ghost stops being a guide and starts
         * being the picture: the user would be framing against an image rather
         * than against what the camera actually sees.
         */
        const val MAX_GHOST_OPACITY = 0.50f
        const val DEFAULT_GHOST_OPACITY = 0.30f
    }
}

/**
 * Turns an ORB match count into a bar fill, continuously.
 *
 * NO THRESHOLD, NO STATE, NO BRANCH. This is the same failure as the green
 * band in a new place: an indicator that turns "ready" and is then followed by
 * UNRELIABLE_LOW_CONFIDENCE has told the user something it had no basis for.
 * A hint can say "more alike" without ever saying "enough".
 *
 * The curve is 1 - exp(-matches / SCALE): smooth, strictly increasing, never
 * reaching 1, and containing no comparison at all. There is no match count at
 * which anything changes character, because there is no `if`.
 *
 * SCALE is a presentation constant, not a threshold. Changing it stretches the
 * bar; it cannot create a state the bar did not have.
 */
object MatchIndicator {

    const val SCALE = 40.0

    /** Bar fill in [0, 1). Monotone in [matchCount], with no discontinuity. */
    fun fill(matchCount: Int): Float {
        val n = if (matchCount > 0) matchCount.toDouble() else 0.0
        return (1.0 - exp(-n / SCALE)).toFloat()
    }

    /**
     * The ends of the bar, as fixed labels.
     *
     * Deliberately NOT a function of the value. If the wording changed as the
     * bar filled, that would be a state transition wearing different clothes.
     */
    const val LOW_END = "less alike"
    const val HIGH_END = "more alike"
}

/**
 * Every user-facing string in guided capture, in one object.
 *
 * Collected here so a unit test can enumerate them and assert that none of
 * them promises an outcome. Strings scattered through composables cannot be
 * checked, and the rule only holds if it is checked.
 */
object GuidedCaptureStrings {

    const val TITLE = "Line this up with your earlier photo"

    /** Verbatim from the M4 brief. The caveat is the point of the sentence. */
    const val HINT_CAVEAT =
        "A closer match here makes a comparison more likely to work. " +
            "It doesn't guarantee one."

    const val GHOST_LABEL = "Faded earlier photo"

    const val OPACITY_LABEL = "How strongly to show the earlier photo"

    const val FRAMING_LABEL = "How alike the framing looks"

    const val REFERENCE_NOTE =
        "Your earlier photo is shown faded over the camera. Move the phone " +
            "until the two line up."

    const val CAPTURE = "Take photo"

    const val CANCEL = "Stop using a reference"

    const val DISCLAIMER = "Have this checked by a dentist."

    /** Enumerated for the test that asserts none of these promises anything. */
    val all: List<String> = listOf(
        TITLE, HINT_CAVEAT, GHOST_LABEL, OPACITY_LABEL, FRAMING_LABEL,
        REFERENCE_NOTE, CAPTURE, CANCEL, DISCLAIMER,
        MatchIndicator.LOW_END, MatchIndicator.HIGH_END,
    )
}
