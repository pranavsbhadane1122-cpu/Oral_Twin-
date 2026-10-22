package com.oraltwin.app.alignment

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.imgproc.Imgproc
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Port of src/alignment/confidence.py.
 *
 * Four signals into one number in [0, 1], plus the breakdown that produced it:
 *
 *   inlierRatio   fraction of ratio-test matches RANSAC kept, used directly
 *   nInliers      absolute support, saturating at INLIERS_SATURATION
 *   reprojError   exp(-err / REPROJ_HALFLIFE), so sub-pixel fits score ~1
 *   blur          variance of Laplacian of the BLURRIER image of the pair
 *
 * A rejected homography scores 0, with no partial credit: there is no such
 * thing as a somewhat-usable alignment that was rejected.
 */
object Confidence {

    data class Breakdown(
        val inlierRatio: Double,
        val nInliers: Double,
        val reprojection: Double,
        val blur: Double,
        val rejected: Boolean,
        val rawInlierRatio: Double,
        val rawNInliers: Int,
        val rawReprojErrorPx: Double?,
        val rawBlurVarLaplacian: Double,
    )

    data class Score(val value: Double, val breakdown: Breakdown)

    /**
     * Variance of the Laplacian - higher means sharper.
     *
     * numpy .var() is the population variance (ddof = 0) and meanStdDev
     * returns the population standard deviation, so squaring it matches
     * Python exactly rather than approximately.
     */
    fun blurMetric(img: Mat): Double {
        val gray = Mat()
        if (img.channels() == 3) {
            Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY)
        } else {
            img.copyTo(gray)
        }
        val lap = Mat()
        Imgproc.Laplacian(gray, lap, CvType.CV_64F)
        val mean = MatOfDouble()
        val stddev = MatOfDouble()
        Core.meanStdDev(lap, mean, stddev)
        val sd = stddev.toArray()[0]
        gray.release(); lap.release(); mean.release(); stddev.release()
        return sd * sd
    }

    /** The pair is only as sharp as its blurrier image. */
    fun pairBlurMetric(img1: Mat, img2: Mat): Double =
        min(blurMetric(img1), blurMetric(img2))

    private fun clamp(v: Double, lo: Double, hi: Double) = max(lo, min(hi, v))

    fun score(
        inlierRatio: Double,
        nInliers: Int,
        reprojectionError: Double,
        blur: Double,
        rejected: Boolean = false,
    ): Score {
        val c = AlignmentConfig.Confidence

        if (rejected) {
            return Score(
                0.0,
                Breakdown(0.0, 0.0, 0.0, 0.0, true,
                          inlierRatio, nInliers,
                          if (reprojectionError.isFinite()) reprojectionError else null,
                          blur)
            )
        }

        val tRatio = clamp(inlierRatio, 0.0, 1.0)
        val tInliers = clamp(nInliers / c.INLIERS_SATURATION, 0.0, 1.0)
        val tReproj = if (!reprojectionError.isFinite()) 0.0
                      else exp(-max(reprojectionError, 0.0) / c.REPROJ_HALFLIFE)
        val tBlur = clamp(blur / c.BLUR_SATURATION, 0.0, 1.0)

        val weighted = c.W_INLIER_RATIO * tRatio +
                       c.W_INLIERS * tInliers +
                       c.W_REPROJ * tReproj +
                       c.W_BLUR * tBlur
        val weightSum = c.W_INLIER_RATIO + c.W_INLIERS + c.W_REPROJ + c.W_BLUR
        val total = clamp(weighted / weightSum, 0.0, 1.0)

        return Score(
            total,
            Breakdown(tRatio, tInliers, tReproj, tBlur, false,
                      inlierRatio, nInliers,
                      if (reprojectionError.isFinite()) reprojectionError else null,
                      blur)
        )
    }
}
