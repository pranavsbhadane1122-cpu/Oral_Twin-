package com.oraltwin.app.export

import com.oraltwin.app.about.AboutText
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * An exported file outlives the app that made it, and is read by someone who
 * cannot ask what anything meant. These tests are the whole defence.
 */
class ExportStringsTest {

    private fun containsWord(haystack: String, word: String): Boolean =
        Regex("\\b${Regex.escape(word)}\\b", RegexOption.IGNORE_CASE)
            .containsMatchIn(haystack)

    private val clinical = listOf(
        "caries", "calculus", "gingivitis", "periodontitis", "cancer",
        "carcinoma", "malignant", "lesion", "ulcer", "abscess", "decay",
        "plaque", "tartar", "infection", "inflammation", "disease",
        "abnormal", "severe", "moderate", "mild", "healthy", "normal",
        "worse", "better", "improved", "deteriorated", "progression",
    )

    // ------------------------------------------------------------- numbers

    @Test
    fun no_export_template_contains_a_digit() {
        // A number here would be read as a measurement by someone with no way
        // to know what it measures. There is no measurement to give.
        for (text in ExportStrings.all) {
            val digits = text.filter { it.isDigit() }
            assertTrue("export text must contain no digits, found [$digits] in: $text",
                       digits.isEmpty())
        }
    }

    @Test
    fun no_export_template_contains_a_percent_sign() {
        for (text in ExportStrings.all) {
            assertTrue("export text must contain no percent sign: $text",
                       !text.contains("%"))
        }
    }

    @Test
    fun no_export_template_mentions_confidence_or_a_score() {
        val measurementWords = listOf("confidence", "score", "percent", "percentage",
                                      "overlap", "area", "pixels", "ratio",
                                      "accuracy", "probability", "likelihood")
        for (text in ExportStrings.all) {
            for (word in measurementWords) {
                assertTrue("export text must not mention $word: $text",
                           !containsWord(text, word))
            }
        }
    }

    @Test
    fun the_date_pattern_itself_carries_no_digits() {
        // dates are rendered into the file and do contain digits; the PATTERN
        // must not, so the digit rule above is about templates and stays true
        assertTrue(ExportStrings.DATE_PATTERN.none { it.isDigit() })
    }

    // -------------------------------------------------------------- bands

    @Test
    fun no_export_template_names_a_risk_band_or_a_colour() {
        val bands = listOf("low", "high", "elevated", "green", "amber", "red",
                           "yellow", "orange", "band", "level", "grade", "stage",
                           "risk", "alert", "warning")
        for (text in ExportStrings.all) {
            for (word in bands) {
                assertTrue("export text must not name $word: $text",
                           !containsWord(text, word))
            }
        }
    }

    // ---------------------------------------------------------- vocabulary

    @Test
    fun no_export_template_uses_clinical_vocabulary() {
        for (text in ExportStrings.all) {
            for (word in clinical) {
                assertTrue("export text must not use the word $word: $text",
                           !containsWord(text, word))
            }
        }
    }

    @Test
    fun the_word_check_is_not_vacuous() {
        // if containsWord were broken every assertion above would pass for the
        // wrong reason
        assertTrue(containsWord("this shows caries", "caries"))
        assertTrue(containsWord("Risk: elevated", "elevated"))
        assertTrue(!containsWord("unmistakable", "mistake"))
    }

    // ---------------------------------------------------------- disclaimer

    @Test
    fun the_disclaimer_appears_in_the_comparable_variant() {
        assertTrue("the comparable export must carry the disclaimer",
                   ExportStrings.comparableVariant.contains(ExportStrings.DISCLAIMER))
    }

    @Test
    fun the_disclaimer_appears_in_the_refused_variant() {
        // asserted per variant, not once: a refusal is the variant most likely
        // to be forwarded without explanation
        assertTrue("the refusal export must carry the disclaimer too",
                   ExportStrings.refusedVariant.contains(ExportStrings.DISCLAIMER))
    }

    @Test
    fun the_disclaimer_says_all_three_things_it_has_to_say() {
        val text = ExportStrings.DISCLAIMER.lowercase()
        assertTrue("it must say how the photo was taken",
                   text.contains("phone camera"))
        assertTrue("it must deny being a diagnosis", text.contains("not a diagnosis"))
        assertTrue("it must deny being a medical assessment",
                   text.contains("not a medical assessment"))
        assertTrue("it must point to a dentist", text.contains("dentist"))
    }

    @Test
    fun the_refusal_variant_carries_no_overlay_label() {
        // there is no trustworthy overlay for a refusal, so naming one would
        // describe something the file does not contain
        assertTrue(!ExportStrings.refusedVariant.contains(ExportStrings.LABEL_OVERLAY))
        assertTrue(ExportStrings.comparableVariant.contains(ExportStrings.LABEL_OVERLAY))
    }

    @Test
    fun the_refusal_variant_says_plainly_that_nothing_was_compared() {
        assertTrue(ExportStrings.refusedVariant.contains(ExportStrings.REFUSED_LINE))
        assertEquals("These two photos could not be compared reliably.",
                     ExportStrings.REFUSED_LINE)
    }

    @Test
    fun both_variants_are_covered_by_the_enumerated_list() {
        // if a variant used a string missing from `all`, the digit and
        // vocabulary tests above would silently skip it
        for (text in ExportStrings.comparableVariant + ExportStrings.refusedVariant) {
            assertTrue("every variant string must be in ExportStrings.all: $text",
                       ExportStrings.all.contains(text))
        }
    }
}

/** The About screen is user-facing prose and gets the same treatment. */
class AboutTextTest {

    private fun containsWord(haystack: String, word: String): Boolean =
        Regex("\\b${Regex.escape(word)}\\b", RegexOption.IGNORE_CASE)
            .containsMatchIn(haystack)

    @Test
    fun about_text_names_no_condition() {
        val banned = listOf("caries", "calculus", "gingivitis", "cancer",
                            "carcinoma", "lesion", "ulcer", "decay", "plaque",
                            "tartar", "infection")
        for (text in AboutText.all) {
            for (word in banned) {
                assertTrue("about text must not name $word: $text",
                           !containsWord(text, word))
            }
        }
    }

    @Test
    fun about_text_says_what_the_app_does_not_do() {
        val joined = AboutText.all.joinToString(" ").lowercase()
        assertTrue(joined.contains("does not identify dental conditions"))
        assertTrue(joined.contains("student research project"))
        assertTrue(joined.contains("dentist"))
    }

    @Test
    fun about_text_separates_the_app_behaviour_from_the_user_choice() {
        val privacy = AboutText.PRIVACY.lowercase()
        assertTrue("it must state the app cannot send anything",
                   privacy.contains("cannot send"))
        assertTrue("it must state what happens if the user shares",
                   privacy.contains("if you share"))
    }
}
