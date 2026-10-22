package com.oraltwin.app.alignment

import org.opencv.core.Mat

/**
 * Port of align_pair() in src/alignment/align.py: detect, match, fit, score.
 *
 * Refusal is a result here, not an error. When the homography is rejected the
 * call still returns normally with h = null, a reason, and a confidence of
 * zero, exactly as the Python does.
 */
object Alignment {

    data class Result(
        val h: Mat?,
        val confidence: Double,
        val breakdown: Confidence.Breakdown,
        val reason: String?,
        val homographyInfo: Homography.Info,
        val matchInfo: Features.MatchInfo,
        val blur: Double,
    ) {
        val rejected: Boolean get() = h == null
    }

    fun alignPair(img1Bgr: Mat, img2Bgr: Mat): Result {
        val matches = Features.detectAndMatch(img1Bgr, img2Bgr)
        val estimate = Homography.estimate(matches.points1, matches.points2)
        val blur = Confidence.pairBlurMetric(img1Bgr, img2Bgr)

        val rejected = estimate.h == null
        val scored = Confidence.score(
            estimate.info.inlierRatio,
            estimate.info.nInliers,
            estimate.info.reprojError,
            blur,
            rejected = rejected,
        )

        matches.points1.release()
        matches.points2.release()

        return Result(
            h = estimate.h,
            confidence = scored.value,
            breakdown = scored.breakdown,
            reason = estimate.info.reason,
            homographyInfo = estimate.info,
            matchInfo = matches.info,
            blur = blur,
        )
    }
}
