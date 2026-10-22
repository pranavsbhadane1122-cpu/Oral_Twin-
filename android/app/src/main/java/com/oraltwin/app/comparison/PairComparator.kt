package com.oraltwin.app.comparison

import android.graphics.Bitmap
import com.oraltwin.app.alignment.Alignment
import com.oraltwin.app.data.PhotoEntity
import com.oraltwin.app.data.PhotoRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 * Pair in, sealed result out. The whole comparison is this one function.
 *
 * The UI never decides comparability. It renders whichever branch it is
 * handed. A screen that can work out for itself whether something counts as
 * comparable is a screen that can be argued into showing a reassuring one.
 */
object PairComparator {

    /**
     * Compare two photographs.
     *
     * [older] and [newer] must already be ordered by timestamp; ordering is
     * the caller's job and PairSelection does it. This function does not
     * re-derive it, but it does assert it, because silently comparing the
     * wrong way round would invert every future change measurement.
     */
    suspend fun compare(
        repository: PhotoRepository,
        older: PhotoEntity,
        newer: PhotoEntity,
    ): ComparisonResult = withContext(Dispatchers.Default) {
        require(older.capturedAtEpochMs <= newer.capturedAtEpochMs) {
            "older must not be newer than newer"
        }

        // Imgcodecs.imdecode, via the repository: anything measured is decoded
        // as BGR. BitmapFactory would give RGBA and move every hue, which the
        // HSV mouth mask depends on. PORT-NOTES.md section 2.
        val olderMat = repository.readBgr(older)
        val newerMat = repository.readBgr(newer)

        try {
            if (olderMat.empty() || newerMat.empty()) {
                return@withContext ComparisonResult.refuse(
                    ComparabilityCode.ALIGNMENT_FAILED)
            }

            // DIRECTION. Alignment.alignPair(a, b) returns H mapping b into
            // a's viewpoint. We want the OLDER warped into the NEWER frame, so
            // the newer is the target and goes first. This inverts the Python
            // convention, where visit1 (the older) is the target - deliberate,
            // because the overlay is "aligned older over newer". Getting this
            // backwards still produces a plausible overlay, of the wrong image.
            val result = Alignment.alignPair(newerMat, olderMat)

            val h = result.h
            if (h == null) {
                return@withContext ComparisonResult.refuse(
                    ComparabilityCode.ALIGNMENT_FAILED)
            }

            val overlap = overlapFraction(h, olderMat, newerMat)
            val code = ComparisonThresholds.classify(
                alignmentFailed = false,
                confidence = result.confidence,
                overlapFraction = overlap,
            )

            if (code != ComparabilityCode.COMPARABLE) {
                h.release()
                return@withContext ComparisonResult.refuse(code)
            }

            val warped = Mat()
            Imgproc.warpPerspective(
                olderMat, warped, h,
                Size(newerMat.cols().toDouble(), newerMat.rows().toDouble()))

            val alignedBitmap = toBitmap(warped)
            val newerBitmap = toBitmap(newerMat)
            warped.release()
            h.release()

            ComparisonResult.Comparable(
                alignedOlder = alignedBitmap,
                newer = newerBitmap,
                confidence = result.confidence,
            )
        } finally {
            olderMat.release()
            newerMat.release()
        }
    }

    /**
     * Share of the newer frame that has a counterpart in the aligned older one.
     *
     * A white image the size of the older photograph, warped through H, marks
     * every pixel of the newer frame that the older one can actually speak to.
     * The rest was never photographed twice.
     */
    private fun overlapFraction(h: Mat, olderMat: Mat, newerMat: Mat): Double {
        val white = Mat(olderMat.rows(), olderMat.cols(), CvType.CV_8UC1, Scalar(255.0))
        val projected = Mat()
        Imgproc.warpPerspective(
            white, projected, h,
            Size(newerMat.cols().toDouble(), newerMat.rows().toDouble()),
            Imgproc.INTER_NEAREST)
        val covered = Core.countNonZero(projected).toDouble()
        val total = (newerMat.rows() * newerMat.cols()).toDouble()
        white.release()
        projected.release()
        return if (total > 0) covered / total else 0.0
    }

    /** BGR Mat to an ARGB Bitmap for display. Display only - never measured. */
    private fun toBitmap(bgr: Mat): Bitmap {
        val rgba = Mat()
        Imgproc.cvtColor(bgr, rgba, Imgproc.COLOR_BGR2RGBA)
        val bitmap = Bitmap.createBitmap(rgba.cols(), rgba.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(rgba, bitmap)
        rgba.release()
        return bitmap
    }
}
