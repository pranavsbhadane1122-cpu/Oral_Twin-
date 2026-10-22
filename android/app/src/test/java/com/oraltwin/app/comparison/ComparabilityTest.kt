package com.oraltwin.app.comparison

import com.oraltwin.app.data.PhotoEntity
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

/**
 * Desktop JVM tests for the parts of M3 that need neither OpenCV nor Android.
 *
 * The decision logic was separated out precisely so it could be tested here.
 * A rule that only runs on a phone is a rule nobody checks.
 */
class ComparabilityTest {

    private fun photo(id: Long, at: Long) =
        PhotoEntity(id = id, filename = "p$id.jpg", capturedAtEpochMs = at,
                    width = 100, height = 100)

    // ----------------------------------------------------------------- codes

    @Test
    fun alignment_failure_wins_over_everything_else() {
        // even with perfect confidence and overlap, no homography means the
        // other two questions have no answer at all
        val code = ComparisonThresholds.classify(
            alignmentFailed = true, confidence = 0.99, overlapFraction = 1.0)
        assertEquals(ComparabilityCode.ALIGNMENT_FAILED, code)
    }

    @Test
    fun low_confidence_is_checked_before_overlap() {
        // an overlap implied by a transform we do not trust is equally untrusted
        val code = ComparisonThresholds.classify(
            alignmentFailed = false, confidence = 0.1, overlapFraction = 0.0)
        assertEquals(ComparabilityCode.UNRELIABLE_LOW_CONFIDENCE, code)
    }

    @Test
    fun small_overlap_is_partly_out_of_frame() {
        val code = ComparisonThresholds.classify(
            alignmentFailed = false, confidence = 0.9, overlapFraction = 0.4)
        assertEquals(ComparabilityCode.PARTIAL_OUT_OF_FRAME, code)
    }

    @Test
    fun good_alignment_is_comparable() {
        val code = ComparisonThresholds.classify(
            alignmentFailed = false, confidence = 0.9, overlapFraction = 0.9)
        assertEquals(ComparabilityCode.COMPARABLE, code)
    }

    @Test
    fun the_boundaries_read_the_way_the_config_comment_reads() {
        // "below this ... refuse": at the floor exactly, do not refuse
        assertEquals(
            ComparabilityCode.COMPARABLE,
            ComparisonThresholds.classify(
                false, ComparisonThresholds.CONFIDENCE_FLOOR, 1.0))
        assertEquals(
            ComparabilityCode.UNRELIABLE_LOW_CONFIDENCE,
            ComparisonThresholds.classify(
                false, ComparisonThresholds.CONFIDENCE_FLOOR - 1e-9, 1.0))
        assertEquals(
            ComparabilityCode.COMPARABLE,
            ComparisonThresholds.classify(
                false, 1.0, ComparisonThresholds.MIN_OVERLAP_FRAC))
        assertEquals(
            ComparabilityCode.PARTIAL_OUT_OF_FRAME,
            ComparisonThresholds.classify(
                false, 1.0, ComparisonThresholds.MIN_OVERLAP_FRAC - 1e-9))
    }

    @Test
    fun thresholds_resolve_from_the_shared_object() {
        // the values mirrored from configs/config.yaml, asserted so that an
        // edit on one side shows up here rather than disagreeing in silence
        assertEquals(0.3, ComparisonThresholds.CONFIDENCE_FLOOR, 1e-9)
        assertEquals(0.60, ComparisonThresholds.MIN_OVERLAP_FRAC, 1e-9)
        assertEquals(0.7, ComparisonThresholds.RELIABILITY_HIGH, 1e-9)
        assertEquals(0.5, ComparisonThresholds.RELIABILITY_MODERATE, 1e-9)
    }

    // --------------------------------------------------------- result mapping

    @Test
    fun every_non_comparable_code_becomes_a_refusal_with_guidance() {
        val refusable = ComparabilityCode.entries
            .filter { it != ComparabilityCode.COMPARABLE }
        assertEquals(3, refusable.size)
        for (code in refusable) {
            val result = ComparisonResult.refuse(code)
            assertEquals(code, result.code)
            assertTrue("guidance must not be blank for $code",
                       result.guidance.isNotBlank())
        }
    }

    @Test
    fun refusing_with_COMPARABLE_is_rejected() {
        try {
            ComparisonResult.refuse(ComparabilityCode.COMPARABLE)
            fail("building a Refused from COMPARABLE should not be possible")
        } catch (expected: IllegalStateException) {
            // RefusalGuidance.forCode calls error(); this is the expected path
        } catch (expected: IllegalArgumentException) {
            // or the require() inside Refused, whichever fires first
        }
    }

    @Test
    fun every_refusal_says_what_to_do_about_it() {
        for (code in ComparabilityCode.entries.filter {
            it != ComparabilityCode.COMPARABLE
        }) {
            val guidance = ComparisonResult.refuse(code).guidance.lowercase()
            assertTrue("$code should tell the user to retake the photo",
                       guidance.contains("retake"))
        }
    }

    @Test
    fun no_refusal_text_contains_a_digit() {
        // the Phase 5 failure: a number beside a non-comparable label is a
        // number a reader will go on to use
        for (code in ComparabilityCode.entries.filter {
            it != ComparabilityCode.COMPARABLE
        }) {
            val guidance = ComparisonResult.refuse(code).guidance
            assertTrue("$code guidance must contain no digits: $guidance",
                       guidance.none { it.isDigit() })
        }
    }

    @Test
    fun no_refusal_text_uses_disease_language() {
        val banned = listOf("caries", "calculus", "gingivitis", "cancer", "lesion",
                            "disease", "infection", "decay", "diagnos", "ulcer",
                            "plaque", "tartar", "severe", "mild")
        for (code in ComparabilityCode.entries.filter {
            it != ComparabilityCode.COMPARABLE
        }) {
            val guidance = ComparisonResult.refuse(code).guidance.lowercase()
            for (word in banned) {
                assertTrue("$code guidance must not contain the word $word",
                           !guidance.contains(word))
            }
        }
    }

    // -------------------------------------------------------- pair selection

    @Test
    fun pair_selection_orders_by_timestamp_not_tap_order() {
        val older = photo(1, 1_000)
        val newer = photo(2, 2_000)
        // tapped newest first, which is the natural thing to do on a
        // reverse-chronological timeline
        val selection = PairSelection.from(listOf(newer, older))
        assertNotNull(selection)
        assertEquals(older.id, selection!!.older.id)
        assertEquals(newer.id, selection.newer.id)
    }

    @Test
    fun pair_selection_gives_the_same_answer_whichever_order_is_tapped() {
        val a = photo(1, 1_000)
        val b = photo(2, 2_000)
        val forward = PairSelection.from(listOf(a, b))!!
        val reverse = PairSelection.from(listOf(b, a))!!
        assertEquals(forward.older.id, reverse.older.id)
        assertEquals(forward.newer.id, reverse.newer.id)
    }

    @Test
    fun pair_selection_requires_exactly_two() {
        assertNull(PairSelection.from(emptyList()))
        assertNull(PairSelection.from(listOf(photo(1, 1))))
        assertNull(PairSelection.from(listOf(photo(1, 1), photo(2, 2), photo(3, 3))))
    }

    @Test
    fun a_photo_cannot_be_paired_with_itself() {
        val same = photo(1, 1_000)
        assertNull(PairSelection.from(listOf(same, same)))
    }

    @Test
    fun two_photos_taken_in_the_same_millisecond_are_still_accepted() {
        // possible if two frames land in the same millisecond; the ordering is
        // then arbitrary, but it must not crash
        val selection = PairSelection.from(listOf(photo(1, 5_000), photo(2, 5_000)))
        assertNotNull(selection)
    }
}
