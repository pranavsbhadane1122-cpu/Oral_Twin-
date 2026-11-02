package com.oraltwin.app.explain

import com.oraltwin.app.comparison.ComparabilityCode

/**
 * SHAPE ONLY. No screen is built against this yet, deliberately.
 *
 * Six milestones of Kotlin in this module have never been compiled. A seventh
 * screen would add more unverified code to a pile that needs a compiler more
 * than it needs features. The data shape is fixed here so the Python side and
 * the eventual Android screen agree on it; the screen comes after the first
 * successful build.
 *
 * Mirrors the dictionary produced by src/explain/explain.py. The two must stay
 * in step, and the field names here are the Python keys verbatim so the mapping
 * is checkable by eye rather than by convention.
 */
sealed class Explanation {

    /**
     * Why a comparison landed in the band it did.
     *
     * Everything here is READ FROM the result that was already computed. The
     * screen will render these fields and must not derive any of them: a
     * screen that recomputed could describe a different result than the one it
     * is sitting next to.
     */
    data class Comparable(
        val band: String,
        val bandText: String,
        /** Every value that fed the decision, each with a plain-language name. */
        val measurements: List<Measurement>,
        /**
         * Which measurement set the band. When nothing crossed a threshold,
         * this says so explicitly - a band assigned because nothing stood out
         * is still a decision and still owes the user a reason.
         */
        val decidingText: String,
        val decidingRules: List<String>,
        val rules: List<RuleExplanation>,
        val counterfactual: Counterfactual?,
        val region: Region?,
        val advisories: List<String>,
        val disclaimer: String,
    ) : Explanation()

    /**
     * Why a comparison was refused. QUALITATIVE ONLY.
     *
     * No numeric field, matching ComparisonResult.Refused and for the same
     * reason: if there is no number on the type, no screen can print one. The
     * alignment confidence is deliberately absent - it is the number a reader
     * would most readily mistake for a result.
     */
    data class Refused(
        val code: ComparabilityCode,
        /** Which check failed, in words. */
        val checkFailed: String,
        val whatToDo: String,
        val disclaimer: String,
    ) : Explanation()
}

/** One measured value, named as a reader would say it, never as a variable. */
data class Measurement(
    val key: String,
    val name: String,
    val display: String,
)

/**
 * A rule, its threshold, and where that threshold came from.
 *
 * [provenanceState] is one of calibrated / artefact only / provisional /
 * unregistered, read from docs/CLAIMS.md. The user is entitled to know whether
 * a number that decided their band is evidence, weakly-sourced evidence, or an
 * assumption, so the state travels with the rule rather than being dropped on
 * the way to the screen.
 */
data class RuleExplanation(
    val rule: String,
    val level: String,
    /** Generated from the rule definition, never written beside it. */
    val sentence: String,
    val thresholdKey: String?,
    val threshold: Double?,
    val provenanceState: String?,
    val provenancePhrasing: String?,
    val deciding: Boolean,
)

/** The smallest change that would have produced a different band. */
data class Counterfactual(
    val threshold: Double,
    val measured: Double,
    val margin: Double,
    /** Carries the provenance caveat when the threshold is not calibrated. */
    val text: String,
)

/**
 * Where the deciding measurement was computed.
 *
 * The honest counterpart to a Grad-CAM heatmap: NOT where a model attended,
 * but which pixels the number came from. No Grad-CAM output in this project
 * has been scored by a clinician, so an attention claim is not available to
 * us; a computation record is.
 */
data class Region(
    val source: String,
    /** left, top, width, height in the earlier photograph, or null. */
    val bbox: List<Int>?,
    val describes: String,
)
