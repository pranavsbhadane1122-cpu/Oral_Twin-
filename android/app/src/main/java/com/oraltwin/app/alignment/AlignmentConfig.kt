package com.oraltwin.app.alignment

/**
 * Mirror of the `alignment:` section of ../../configs/config.yaml.
 *
 * Deliberately a single object of named constants rather than values inlined at
 * call sites, so the Python and Kotlin implementations can be compared
 * parameter by parameter. If a value here drifts from the YAML, the two
 * implementations silently stop being the same algorithm, and the only defence
 * is that both are readable in one place.
 *
 * Mirrored from configs/config.yaml as of 2026-10-22.
 */
object AlignmentConfig {

    // --- detector ----------------------------------------------------------
    const val N_FEATURES = 4000
    const val RATIO_TEST = 0.75           // Lowe ratio
    const val CLAHE_CLIP = 2.0
    const val CLAHE_GRID = 8

    // --- region of interest ------------------------------------------------
    /** Fall back to the whole frame if the mouth mask is smaller than this. */
    const val MASK_MIN_AREA_FRAC = 0.10

    // --- RANSAC ------------------------------------------------------------
    const val RANSAC_REPROJ_THRESH = 4.0
    const val MIN_INLIERS = 15

    // --- homography plausibility (H normalised to h33 = 1, visit2 -> visit1)
    const val MIN_SCALE = 0.5             // sqrt(|det| of the 2x2 affine part)
    const val MAX_SCALE = 2.0
    const val MAX_PERSPECTIVE = 0.002     // |h31|, |h32| above this = implausible
    const val MIN_ABS_DET = 1.0e-6

    // --- confidence --------------------------------------------------------
    object Confidence {
        const val W_INLIER_RATIO = 0.35   // the four weights sum to 1
        const val W_INLIERS = 0.20
        const val W_REPROJ = 0.30
        const val W_BLUR = 0.15
        const val INLIERS_SATURATION = 60.0
        const val REPROJ_HALFLIFE = 2.0   // px
        const val BLUR_SATURATION = 150.0 // variance of Laplacian
    }

    // --- HSV bands for the mouth heuristic ---------------------------------
    // OpenCV hue is 0-179, and oral tissue straddles the 0/179 wrap, which is
    // why tissue needs two bands rather than one.
    val TEETH_LOW = doubleArrayOf(0.0, 0.0, 120.0)      // bright, weakly saturated
    val TEETH_HIGH = doubleArrayOf(179.0, 90.0, 255.0)
    val TISSUE_LOW_A = doubleArrayOf(0.0, 40.0, 40.0)   // reddish-pink, low hue
    val TISSUE_HIGH_A = doubleArrayOf(20.0, 255.0, 255.0)
    val TISSUE_LOW_B = doubleArrayOf(160.0, 40.0, 40.0) // and the wrapped end
    val TISSUE_HIGH_B = doubleArrayOf(179.0, 255.0, 255.0)

    const val MORPH_KERNEL = 9
    const val DILATE_ITERATIONS = 2
}
