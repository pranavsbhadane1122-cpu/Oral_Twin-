package com.oraltwin.app

import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.oraltwin.app.comparison.ComparabilityCode
import com.oraltwin.app.comparison.ComparisonResult
import com.oraltwin.app.comparison.ComparisonScreen
import com.oraltwin.app.comparison.PairSelection
import com.oraltwin.app.data.PhotoEntity
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/**
 * A refusal must not be able to reach the overlay, and a pair must be ordered
 * by timestamp however it was tapped.
 *
 * The first is already enforced by the type system - the overlay composable
 * takes a ComparisonResult.Comparable, so a Refused cannot be handed to it at
 * all - but a type-level guarantee is only as good as the dispatch in front of
 * it. These tests render the real screen and check what actually appears.
 */
@RunWith(AndroidJUnit4::class)
class ComparisonNavigationTest {

    @get:Rule
    val compose = createComposeRule()

    private val refusalCodes = ComparabilityCode.entries
        .filter { it != ComparabilityCode.COMPARABLE }

    // text that appears only on the overlay screen
    private val overlayHeading = "Your two photos, lined up"
    private val overlayControl = "Side by side"

    private fun photo(id: Long, at: Long) =
        PhotoEntity(id = id, filename = "p$id.jpg", capturedAtEpochMs = at,
                    width = 64, height = 48)

    /**
     * setContent may be called only once per rule, so the screen is mounted
     * once over a state holder and the result is swapped underneath it.
     */
    private fun mountSwappable(): MutableState<ComparisonResult> {
        val state: MutableState<ComparisonResult> =
            mutableStateOf(ComparisonResult.refuse(ComparabilityCode.ALIGNMENT_FAILED))
        compose.setContent { ComparisonScreen(result = state.value, onBack = {}) }
        return state
    }

    private fun MutableState<ComparisonResult>.show(result: ComparisonResult) {
        compose.runOnUiThread { value = result }
        compose.waitForIdle()
    }

    @Test
    fun no_refusal_ever_shows_the_overlay() {
        val state = mountSwappable()
        for (code in refusalCodes) {
            state.show(ComparisonResult.refuse(code))

            compose.onNodeWithText("These photos could not be compared")
                .assertIsDisplayed()
            compose.onNodeWithText(overlayHeading).assertDoesNotExist()
            compose.onNodeWithText(overlayControl).assertDoesNotExist()
        }
    }

    @Test
    fun every_refusal_carries_the_dentist_line() {
        val state = mountSwappable()
        for (code in refusalCodes) {
            state.show(ComparisonResult.refuse(code))
            compose.onNodeWithText("Have this checked by a dentist.")
                .assertIsDisplayed()
        }
    }

    @Test
    fun every_refusal_names_its_code_on_screen() {
        val state = mountSwappable()
        for (code in refusalCodes) {
            state.show(ComparisonResult.refuse(code))
            compose.onNodeWithText("comparability: $code").assertIsDisplayed()
        }
    }

    @Test
    fun no_refusal_puts_a_digit_on_the_screen() {
        // Refused carries no number by construction, so the only text that
        // could contain one is the guidance or the code name. Both are checked
        // rather than assumed.
        for (code in refusalCodes) {
            val refusal = ComparisonResult.refuse(code)
            assertTrue("guidance for $code must contain no digits: ${refusal.guidance}",
                       refusal.guidance.none { it.isDigit() })
            assertTrue("the code name $code must contain no digits",
                       code.name.none { it.isDigit() })
        }
    }

    @Test
    fun a_refusal_offers_only_a_way_back() {
        var backPressed = false
        compose.setContent {
            ComparisonScreen(
                result = ComparisonResult.refuse(ComparabilityCode.ALIGNMENT_FAILED),
                onBack = { backPressed = true },
            )
        }
        compose.onNodeWithText("Back to my photos").performClick()
        assertTrue("the only action offered on a refusal is going back", backPressed)
    }

    // -------------------------------------------------------- pair ordering

    @Test
    fun selecting_newest_first_still_compares_oldest_to_newest() {
        val older = photo(1, 1_000_000)
        val newer = photo(2, 2_000_000)

        // taps in reverse order, which is what happens on a reverse-
        // chronological timeline, where the newest row sits at the top
        val selection = PairSelection.from(listOf(newer, older))

        assertNotNull(selection)
        assertEquals("older must be the earlier timestamp",
                     older.id, selection!!.older.id)
        assertEquals("newer must be the later timestamp",
                     newer.id, selection.newer.id)
        assertTrue(selection.older.capturedAtEpochMs <= selection.newer.capturedAtEpochMs)
    }

    @Test
    fun the_pair_is_rejected_unless_exactly_two_are_chosen() {
        assertNull(PairSelection.from(listOf(photo(1, 1))))
        assertNull(PairSelection.from(listOf(photo(1, 1), photo(2, 2), photo(3, 3))))
    }
}
