package com.oraltwin.app.comparison

import android.graphics.Bitmap

/**
 * The outcome of comparing two photographs.
 *
 * WHY Refused CARRIES NO NUMBER.
 *
 * Phase 5 of the Python project printed raw pixel areas - 755 and 1004 - beside
 * a "partly out of frame" label. Nothing claimed a result, but a reader can
 * divide one by the other and conclude the spot grew by a third, from a
 * comparison the system had just declined to make. The label said one thing and
 * the numbers said another, and numbers win.
 *
 * The fix is structural rather than editorial. [Refused] has no numeric
 * property: not a visible one, not a hidden one, not a nullable one. A screen
 * rendering a Refused has nothing to print, so it cannot print a measurement by
 * accident, and a future edit that wants to show one has to change this type
 * first - where a test is waiting.
 *
 * See ComparisonResultShapeTest, which asserts this by reflection rather than
 * by trusting the comment above.
 */
sealed class ComparisonResult {

    /**
     * The pair could be compared. Carries the aligned overlay and the
     * confidence that produced it.
     */
    data class Comparable(
        /** The older photograph warped into the newer one's viewpoint. */
        val alignedOlder: Bitmap,
        /** The newer photograph, unmodified, as the overlay base. */
        val newer: Bitmap,
        val confidence: Double,
    ) : ComparisonResult() {
        val code: ComparabilityCode get() = ComparabilityCode.COMPARABLE
    }

    /**
     * The pair could not be compared. Carries only the code and what to do
     * about it.
     */
    data class Refused(
        val code: ComparabilityCode,
        val guidance: String,
    ) : ComparisonResult() {
        init {
            require(code != ComparabilityCode.COMPARABLE) {
                "Refused cannot carry COMPARABLE"
            }
        }
    }

    companion object {
        /** Build a refusal, with the guidance that belongs to the code. */
        fun refuse(code: ComparabilityCode): Refused =
            Refused(code, RefusalGuidance.forCode(code))
    }
}
