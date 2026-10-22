package com.oraltwin.app.capture

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

/**
 * The indicator must have no state, and the capture wording must promise
 * nothing. Both are checked here rather than trusted.
 */
class MatchIndicatorTest {

    private val range = 0..600

    @Test
    fun fill_is_monotone_non_decreasing() {
        var previous = -1.0f
        for (n in range) {
            val value = MatchIndicator.fill(n)
            assertTrue("fill($n) = $value went backwards from $previous",
                       value >= previous)
            previous = value
        }
    }

    @Test
    fun fill_has_no_discontinuity() {
        // The real assertion behind "no branch": a threshold shows up as a
        // jump. The largest single-count step is at n = 0, where the curve is
        // steepest, and 1/SCALE bounds it. Anything larger means something
        // switched rather than grew.
        val bound = (1.0 / MatchIndicator.SCALE).toFloat()
        for (n in range) {
            val step = abs(MatchIndicator.fill(n + 1) - MatchIndicator.fill(n))
            assertTrue("step from $n to ${n + 1} was $step, above $bound - " +
                           "that is a state change, not a continuous fill",
                       step <= bound + 1e-6f)
        }
    }

    @Test
    fun fill_has_no_plateau_that_could_hide_a_threshold() {
        // a strictly increasing curve cannot sit still and then jump
        for (n in 0..200) {
            assertTrue("fill should strictly increase at $n",
                       MatchIndicator.fill(n + 1) > MatchIndicator.fill(n))
        }
    }

    @Test
    fun fill_stays_within_the_bar_and_never_reaches_the_end() {
        // never 1.0: a full bar is a completion claim, and there is nothing to
        // complete. More matches is always possible.
        for (n in range) {
            val value = MatchIndicator.fill(n)
            assertTrue("fill($n) = $value below 0", value >= 0f)
            assertTrue("fill($n) = $value reached or passed 1", value < 1f)
        }
        assertEquals(0f, MatchIndicator.fill(0), 1e-6f)
    }

    @Test
    fun negative_or_absent_counts_are_treated_as_none() {
        assertEquals(0f, MatchIndicator.fill(-5), 1e-6f)
        assertEquals(0f, MatchIndicator.fill(0), 1e-6f)
    }

    @Test
    fun the_end_labels_do_not_depend_on_the_value() {
        // wording that changed as the bar filled would be a state transition
        // wearing different clothes
        assertEquals("less alike", MatchIndicator.LOW_END)
        assertEquals("more alike", MatchIndicator.HIGH_END)
    }
}

class GuidedCaptureStringsTest {

    /** Word-boundary matching: "ok" is a substring of ordinary words. */
    private fun containsWord(haystack: String, word: String): Boolean =
        Regex("\\b${Regex.escape(word)}\\b", RegexOption.IGNORE_CASE)
            .containsMatchIn(haystack)

    @Test
    fun no_capture_string_claims_a_quality_or_an_outcome() {
        val promises = listOf("ready", "good", "ok", "okay", "perfect", "great",
                              "correct", "sufficient", "enough", "done",
                              "success", "valid", "pass", "passed")
        for (text in GuidedCaptureStrings.all) {
            for (word in promises) {
                assertTrue(
                    "capture wording must not promise an outcome, but found " +
                        "the word $word in: $text",
                    !containsWord(text, word),
                )
            }
        }
    }

    @Test
    fun no_capture_string_uses_disease_vocabulary() {
        val banned = listOf("caries", "calculus", "gingivitis", "cancer",
                            "lesion", "disease", "infection", "decay",
                            "diagnosis", "diagnose", "ulcer", "plaque",
                            "tartar", "healthy", "abnormal", "severe", "mild")
        for (text in GuidedCaptureStrings.all) {
            for (word in banned) {
                assertTrue(
                    "capture wording must not use clinical language, but " +
                        "found $word in: $text",
                    !containsWord(text, word),
                )
            }
        }
    }

    @Test
    fun the_caveat_is_present_and_says_it_guarantees_nothing() {
        val caveat = GuidedCaptureStrings.HINT_CAVEAT
        assertTrue(caveat.contains("more likely to work"))
        assertTrue(caveat.lowercase().contains("guarantee"))
    }

    @Test
    fun the_word_check_is_not_vacuous() {
        // if containsWord were broken, every assertion above would pass for
        // the wrong reason
        assertTrue(containsWord("the photo looks good", "good"))
        assertTrue(containsWord("Ready to capture", "ready"))
        // and it must not fire on a substring inside another word
        assertTrue(!containsWord("looking at the token", "ok"))
    }
}

class GuidedSessionTest {

    private val photo = com.oraltwin.app.data.PhotoEntity(
        id = 1, filename = "a.jpg", capturedAtEpochMs = 1_000,
        width = 100, height = 100)

    @Test
    fun default_opacity_is_within_the_allowed_range() {
        val session = GuidedSession(photo)
        assertTrue(session.ghostOpacity <= GuidedSession.MAX_GHOST_OPACITY)
        assertTrue(session.ghostOpacity >= 0f)
        assertEquals(0.30f, GuidedSession.DEFAULT_GHOST_OPACITY, 1e-6f)
    }

    @Test
    fun opacity_is_capped_at_half() {
        assertEquals(0.50f, GuidedSession.MAX_GHOST_OPACITY, 1e-6f)
        val session = GuidedSession(photo).withOpacity(0.9f)
        assertEquals(GuidedSession.MAX_GHOST_OPACITY, session.ghostOpacity, 1e-6f)
    }

    @Test
    fun opacity_cannot_go_below_zero() {
        assertEquals(0f, GuidedSession(photo).withOpacity(-1f).ghostOpacity, 1e-6f)
    }

    @Test
    fun an_out_of_range_opacity_is_rejected_at_construction() {
        try {
            GuidedSession(photo, ghostOpacity = 0.8f)
            org.junit.Assert.fail("opacity above the cap should not construct")
        } catch (expected: IllegalArgumentException) {
            // as intended
        }
    }
}
