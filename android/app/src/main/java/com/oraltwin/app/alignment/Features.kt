package com.oraltwin.app.alignment

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.features2d.BFMatcher
import org.opencv.features2d.ORB
import org.opencv.imgproc.Imgproc

/**
 * Port of src/alignment/features.py.
 *
 * Pipeline, unchanged: greyscale -> CLAHE -> ORB restricted to a mouth-region
 * mask -> BFMatcher with Hamming distance -> Lowe ratio test.
 *
 * ON THE MOUTH REGION: this ports the HSV colour heuristic only. The Python
 * project also has a trained oral-cavity segmenter and it is NOT here, for two
 * separate reasons. It is trained on the Piyarathne dataset, which is
 * CC BY-NC-ND and cannot be redistributed in an APK. And it measured WORSE for
 * this job anyway: it covered 70.4% of the frame against the heuristic at
 * 91.2%, left ORB a median 677 keypoints against 840, and median corner error
 * rose from 1.18 to 1.41 px. The Python default is the heuristic on that
 * evidence, so this port gives nothing up. See EXTENSION POINT below.
 *
 * All images here are BGR, matching cv2.imread, because the HSV thresholds
 * were derived against BGR input. Decode assets with Imgcodecs.imdecode rather
 * than BitmapFactory, which hands back RGBA and would silently invert the hue
 * bands - a bug this project has already been bitten by once.
 */
object Features {

    /**
     * A three-component Scalar from a config triple.
     *
     * Built element by element rather than handing the DoubleArray to
     * Scalar(double[]): that constructor only clones arrays of length 4 and
     * otherwise falls through to set(), which is correct today but is not a
     * contract worth relying on for threshold values.
     */
    private fun scalar3(v: DoubleArray) = Scalar(v[0], v[1], v[2])

    /** Greyscale + CLAHE, for lighting-robust descriptors. */
    fun preprocess(imgBgr: Mat): Mat {
        val gray = Mat()
        Imgproc.cvtColor(imgBgr, gray, Imgproc.COLOR_BGR2GRAY)
        val clahe = Imgproc.createCLAHE(
            AlignmentConfig.CLAHE_CLIP,
            Size(AlignmentConfig.CLAHE_GRID.toDouble(), AlignmentConfig.CLAHE_GRID.toDouble())
        )
        val out = Mat()
        clahe.apply(gray, out)
        gray.release()
        return out
    }

    /**
     * Binary mask (0/255) of the dental/oral region.
     *
     * Union of two colour families - teeth (bright, weakly saturated) and
     * gum/lip/tongue tissue (reddish-pink, wrapping around hue 0) - cleaned
     * with close+open, reduced to the largest connected component, then
     * dilated so keypoints on the tissue border survive.
     *
     * If that component covers less than MASK_MIN_AREA_FRAC of the frame, the
     * heuristic is assumed to have failed and the whole frame is used instead:
     * a bad mask is worse than no mask.
     *
     * EXTENSION POINT. A licence-clean segmentation model could replace this
     * by returning a mask of the same contract: CV_8UC1, values 0 or 255, the
     * same size as the input, and never near-empty. Route it through a
     * regionMask(img) -> (Mat, String) wrapper that keeps mouthMask as the
     * fallback and records which path was taken, mirroring
     * features.region_mask in the Python. Do not simply swap this body out:
     * the Python measured the model-backed region as worse, so any replacement
     * has to be measured against this heuristic rather than assumed better.
     */
    fun mouthMask(imgBgr: Mat): Mat {
        val h = imgBgr.rows()
        val w = imgBgr.cols()
        val hsv = Mat()
        Imgproc.cvtColor(imgBgr, hsv, Imgproc.COLOR_BGR2HSV)

        val teeth = Mat()
        Core.inRange(hsv, scalar3(AlignmentConfig.TEETH_LOW),
                     scalar3(AlignmentConfig.TEETH_HIGH), teeth)
        val tissueA = Mat()
        Core.inRange(hsv, scalar3(AlignmentConfig.TISSUE_LOW_A),
                     scalar3(AlignmentConfig.TISSUE_HIGH_A), tissueA)
        val tissueB = Mat()
        Core.inRange(hsv, scalar3(AlignmentConfig.TISSUE_LOW_B),
                     scalar3(AlignmentConfig.TISSUE_HIGH_B), tissueB)

        val mask = Mat()
        Core.bitwise_or(tissueA, tissueB, mask)
        Core.bitwise_or(teeth, mask, mask)
        hsv.release(); teeth.release(); tissueA.release(); tissueB.release()

        val k = Imgproc.getStructuringElement(
            Imgproc.MORPH_ELLIPSE,
            Size(AlignmentConfig.MORPH_KERNEL.toDouble(),
                 AlignmentConfig.MORPH_KERNEL.toDouble())
        )
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, k)
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, k)

        // keep only the largest connected component (the mouth)
        val labels = Mat()
        val stats = Mat()
        val centroids = Mat()
        val n = Imgproc.connectedComponentsWithStats(mask, labels, stats, centroids, 8, CvType.CV_32S)
        if (n > 1) {
            var best = 1
            var bestArea = -1.0
            for (label in 1 until n) {
                val area = stats.get(label, Imgproc.CC_STAT_AREA)[0]
                if (area > bestArea) {
                    bestArea = area
                    best = label
                }
            }
            // compare() yields 255 where equal, 0 elsewhere - the same 0/255
            // convention as the numpy where() it replaces
            val largest = Mat()
            Core.compare(labels, Scalar(best.toDouble()), largest, Core.CMP_EQ)
            largest.copyTo(mask)
            largest.release()
        }
        labels.release(); stats.release(); centroids.release()

        val area = Core.countNonZero(mask).toDouble()
        if (area < AlignmentConfig.MASK_MIN_AREA_FRAC * h * w) {
            mask.release()
            k.release()
            return Mat(h, w, CvType.CV_8UC1, Scalar(255.0))   // whole frame
        }

        Imgproc.dilate(mask, mask, k, Point(-1.0, -1.0), AlignmentConfig.DILATE_ITERATIONS)
        k.release()
        return mask
    }

    data class MatchInfo(
        val nKp1: Int,
        val nKp2: Int,
        val nRawMatches: Int,
        val nGoodMatches: Int,
        val regionFrac1: Double,
        val regionFrac2: Double,
        val regionSource1: String,
        val regionSource2: String,
    )

    data class Matches(
        val points1: MatOfPoint2f,
        val points2: MatOfPoint2f,
        val info: MatchInfo,
    )

    /** Detect, describe and match. Returns corresponding points in both images. */
    fun detectAndMatch(img1Bgr: Mat, img2Bgr: Mat): Matches {
        val g1 = preprocess(img1Bgr)
        val g2 = preprocess(img2Bgr)
        val m1 = mouthMask(img1Bgr)
        val m2 = mouthMask(img2Bgr)

        val frac1 = Core.countNonZero(m1).toDouble() / (m1.rows() * m1.cols())
        val frac2 = Core.countNonZero(m2).toDouble() / (m2.rows() * m2.cols())
        val src1 = if (frac1 >= 0.999) "full_frame" else "hsv_heuristic"
        val src2 = if (frac2 >= 0.999) "full_frame" else "hsv_heuristic"

        val orb = ORB.create(AlignmentConfig.N_FEATURES)
        val kp1 = MatOfKeyPoint()
        val kp2 = MatOfKeyPoint()
        val des1 = Mat()
        val des2 = Mat()
        orb.detectAndCompute(g1, m1, kp1, des1)
        orb.detectAndCompute(g2, m2, kp2, des2)

        val keypoints1 = kp1.toArray()
        val keypoints2 = kp2.toArray()
        var info = MatchInfo(keypoints1.size, keypoints2.size, 0, 0,
                             frac1, frac2, src1, src2)

        if (des1.empty() || des2.empty() || keypoints1.size < 2 || keypoints2.size < 2) {
            g1.release(); g2.release(); m1.release(); m2.release()
            des1.release(); des2.release(); kp1.release(); kp2.release()
            return Matches(MatOfPoint2f(), MatOfPoint2f(), info)
        }

        val matcher = BFMatcher.create(Core.NORM_HAMMING, false)   // crossCheck = false
        val knn = ArrayList<MatOfDMatch>()
        matcher.knnMatch(des1, des2, knn, 2)

        val p1 = ArrayList<Point>()
        val p2 = ArrayList<Point>()
        for (pair in knn) {
            val candidates = pair.toArray()
            // Python skips any pair that did not come back with two neighbours
            if (candidates.size != 2) continue
            val best = candidates[0]
            val second = candidates[1]
            if (best.distance < AlignmentConfig.RATIO_TEST * second.distance) {
                p1.add(keypoints1[best.queryIdx].pt)
                p2.add(keypoints2[best.trainIdx].pt)
            }
        }

        info = info.copy(nRawMatches = knn.size, nGoodMatches = p1.size)

        g1.release(); g2.release(); m1.release(); m2.release()
        des1.release(); des2.release(); kp1.release(); kp2.release()

        val out1 = MatOfPoint2f()
        val out2 = MatOfPoint2f()
        if (p1.isNotEmpty()) {
            out1.fromList(p1)
            out2.fromList(p2)
        }
        return Matches(out1, out2, info)
    }
}
