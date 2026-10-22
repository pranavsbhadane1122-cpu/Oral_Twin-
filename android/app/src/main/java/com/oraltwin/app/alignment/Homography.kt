package com.oraltwin.app.alignment

import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Port of src/alignment/homography.py.
 *
 * estimate() fits H mapping points in image2 onto image1, so the returned H
 * warps visit2 into visit1's viewpoint. Degenerate or implausible results are
 * REJECTED with a human-readable reason rather than returned as a transform
 * that happens to be garbage. That rule is the point of the module: a wrong
 * alignment produces a confident, wrong change measurement downstream, which
 * is worse than no measurement.
 */
object Homography {

    data class Info(
        val reason: String?,          // null on success
        val nInliers: Int,
        val inlierRatio: Double,
        val reprojError: Double,      // Double.POSITIVE_INFINITY when unknown
        val nMatches: Int,
    )

    data class Result(val h: Mat?, val info: Info)

    /** Scale H so h33 = 1, making entries comparable across estimates. */
    fun normalize(h: Mat): Mat {
        val out = Mat()
        h.convertTo(out, org.opencv.core.CvType.CV_64F)
        val h33 = out.get(2, 2)[0]
        if (abs(h33) < 1e-12) return out
        val scaled = Mat()
        Core.divide(out, org.opencv.core.Scalar(h33), scaled)
        out.release()
        return scaled
    }

    private fun det3(m: Mat): Double {
        val a = DoubleArray(9)
        var i = 0
        for (r in 0 until 3) for (c in 0 until 3) a[i++] = m.get(r, c)[0]
        return a[0] * (a[4] * a[8] - a[5] * a[7]) -
               a[1] * (a[3] * a[8] - a[5] * a[6]) +
               a[2] * (a[3] * a[7] - a[4] * a[6])
    }

    /** null if H looks physically plausible, else a reason string. */
    fun checkPlausible(h: Mat): String? {
        val hn = normalize(h)
        val det = det3(hn)
        if (!det.isFinite() || abs(det) < AlignmentConfig.MIN_ABS_DET) {
            hn.release()
            return "near-singular homography (|det|=%.2e)".format(abs(det))
        }

        // determinant of the 2x2 affine part
        val a = hn.get(0, 0)[0]
        val b = hn.get(0, 1)[0]
        val c = hn.get(1, 0)[0]
        val d = hn.get(1, 1)[0]
        val affineDet = a * d - b * c
        if (!affineDet.isFinite() || affineDet <= 0.0) {
            hn.release()
            return "non-positive affine determinant (%.3e) - mirrored/degenerate".format(affineDet)
        }
        val scale = sqrt(abs(affineDet))
        if (scale < AlignmentConfig.MIN_SCALE || scale > AlignmentConfig.MAX_SCALE) {
            hn.release()
            return "implausible scale %.3f (allowed %s-%s)".format(
                scale, AlignmentConfig.MIN_SCALE, AlignmentConfig.MAX_SCALE)
        }

        val persp = maxOf(abs(hn.get(2, 0)[0]), abs(hn.get(2, 1)[0]))
        hn.release()
        if (persp > AlignmentConfig.MAX_PERSPECTIVE) {
            return "heavy perspective term %.5f (max %s)".format(
                persp, AlignmentConfig.MAX_PERSPECTIVE)
        }
        return null
    }

    /**
     * Fit H mapping points2 -> points1.
     *
     * Note the argument order into findHomography: src = points2, dst =
     * points1. Reversing it silently produces the inverse transform, which
     * still looks plausible and warps the wrong way.
     */
    fun estimate(points1: MatOfPoint2f, points2: MatOfPoint2f): Result {
        val n = points1.rows()
        var info = Info(null, 0, 0.0, Double.POSITIVE_INFINITY, n)

        if (n < 4 || points2.rows() < 4) {
            return Result(null, info.copy(
                reason = "too few matches ($n) to fit a homography"))
        }

        val mask = Mat()
        // The 4-argument overload, matching the Python call: RANSAC with the
        // configured reprojection threshold and OpenCV defaults for maxIters
        // (2000) and confidence (0.995). Passing them explicitly here would
        // diverge from Python if OpenCV ever changes its defaults.
        val h = Calib3d.findHomography(points2, points1, Calib3d.RANSAC,
                                       AlignmentConfig.RANSAC_REPROJ_THRESH, mask)

        if (h.empty()) {
            mask.release()
            return Result(null, info.copy(reason = "findHomography failed to converge"))
        }

        var nIn = 0
        val maskBytes = ByteArray(mask.rows() * mask.cols())
        mask.get(0, 0, maskBytes)
        for (v in maskBytes) if (v.toInt() != 0) nIn++

        info = info.copy(nInliers = nIn,
                         inlierRatio = nIn.toDouble() / maxOf(n, 1))

        if (nIn > 0) {
            val src = ArrayList<Point>(nIn)
            val dst = ArrayList<Point>(nIn)
            val p1 = points1.toArray()
            val p2 = points2.toArray()
            for (i in maskBytes.indices) {
                if (maskBytes[i].toInt() != 0) {
                    src.add(p2[i])
                    dst.add(p1[i])
                }
            }
            val srcMat = MatOfPoint2f(); srcMat.fromList(src)
            val projected = MatOfPoint2f()
            Core.perspectiveTransform(srcMat, projected, h)
            val proj = projected.toArray()
            var sum = 0.0
            for (i in proj.indices) {
                val dx = proj[i].x - dst[i].x
                val dy = proj[i].y - dst[i].y
                sum += sqrt(dx * dx + dy * dy)
            }
            info = info.copy(reprojError = sum / proj.size)
            srcMat.release(); projected.release()
        }
        mask.release()

        if (nIn < AlignmentConfig.MIN_INLIERS) {
            h.release()
            return Result(null, info.copy(
                reason = "only $nIn inliers (min ${AlignmentConfig.MIN_INLIERS})"))
        }

        val reason = checkPlausible(h)
        if (reason != null) {
            h.release()
            return Result(null, info.copy(reason = reason))
        }

        val normalized = normalize(h)
        h.release()
        return Result(normalized, info)
    }

    /**
     * Mean L2 distance in pixels between the image corners projected through
     * two homographies. Scale-invariant: both are normalised first, and the
     * projection divides by the homogeneous coordinate anyway.
     *
     * Same definition as corner_error() in src/alignment/homography.py, which
     * is what evaluate_alignment.py reports.
     */
    fun cornerError(hEst: Mat, hTrue: Mat, width: Int, height: Int): Double {
        val corners = MatOfPoint2f(
            Point(0.0, 0.0),
            Point((width - 1).toDouble(), 0.0),
            Point((width - 1).toDouble(), (height - 1).toDouble()),
            Point(0.0, (height - 1).toDouble()),
        )
        val a = MatOfPoint2f()
        val b = MatOfPoint2f()
        val nEst = normalize(hEst)
        val nTrue = normalize(hTrue)
        Core.perspectiveTransform(corners, a, nEst)
        Core.perspectiveTransform(corners, b, nTrue)
        val pa = a.toArray()
        val pb = b.toArray()
        var sum = 0.0
        for (i in pa.indices) {
            val dx = pa[i].x - pb[i].x
            val dy = pa[i].y - pb[i].y
            sum += sqrt(dx * dx + dy * dy)
        }
        corners.release(); a.release(); b.release(); nEst.release(); nTrue.release()
        return sum / pa.size
    }

    /** Warp visit2 into visit1's frame. */
    fun warpVisit2(img2: Mat, h: Mat, targetHeight: Int, targetWidth: Int): Mat {
        val out = Mat()
        org.opencv.imgproc.Imgproc.warpPerspective(
            img2, out, h, org.opencv.core.Size(targetWidth.toDouble(), targetHeight.toDouble()))
        return out
    }
}
