package com.oraltwin.app.capture

import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.oraltwin.app.alignment.AlignmentConfig
import com.oraltwin.app.data.PhotoEntity
import com.oraltwin.app.data.PhotoRepository
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.Size
import org.opencv.features2d.BFMatcher
import org.opencv.features2d.ORB
import org.opencv.imgproc.Imgproc
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

/**
 * Counts ORB matches between the live preview and the reference photograph.
 *
 * A HINT, NOT A MEASUREMENT. Deliberately cheap and deliberately incomplete:
 *
 *   NO HOMOGRAPHY. Alignment is roughly 77% of the desktop pipeline cost, and
 *   none of it is needed to say "these look more alike than they did". The
 *   expensive part is skipped, not optimised.
 *
 *   DOWNSCALED HARD. Frames go to [ANALYSIS_LONG_EDGE] px on the long edge
 *   before anything touches them.
 *
 *   AT MOST [MAX_FPS] frames a second, and frames arriving while one is still
 *   in flight are DROPPED, never queued. A queue would turn a slow phone into
 *   a growing backlog and a preview that falls further behind the longer it
 *   runs. CameraX STRATEGY_KEEP_ONLY_LATEST does the same at its own level;
 *   the in-flight flag here covers our own processing.
 *
 * If this still costs too much on a real phone, delete it. The ghost overlay
 * is the feature; this is a garnish, and a stuttering preview is worse than no
 * hint at all. [intervalStats] reports what it actually cost so that decision
 * can be made from a number rather than an impression.
 */
class FramingHintAnalyzer(
    private val referenceDescriptors: Mat,
    private val onMatchCount: (Int) -> Unit,
) : ImageAnalysis.Analyzer {

    private val inFlight = AtomicBoolean(false)
    private val lastRunAtMs = AtomicLong(0)
    private val frameCount = AtomicInteger(0)
    private val droppedCount = AtomicInteger(0)
    private val totalProcessingMs = AtomicLong(0)

    private val orb: ORB by lazy { ORB.create(ANALYSIS_FEATURES) }
    private val matcher: BFMatcher by lazy {
        BFMatcher.create(Core.NORM_HAMMING, false)
    }

    /** frames seen, frames actually processed, mean ms per processed frame */
    data class Stats(val seen: Int, val processed: Int, val meanMs: Double)

    fun intervalStats(): Stats {
        val processed = frameCount.get()
        val mean = if (processed > 0) totalProcessingMs.get().toDouble() / processed
                   else 0.0
        return Stats(seen = processed + droppedCount.get(),
                     processed = processed, meanMs = mean)
    }

    override fun analyze(image: ImageProxy) {
        val now = System.currentTimeMillis()
        val due = now - lastRunAtMs.get() >= MIN_INTERVAL_MS

        // skip rather than queue: compareAndSet fails if one is already running
        if (!due || !inFlight.compareAndSet(false, true)) {
            droppedCount.incrementAndGet()
            image.close()
            return
        }

        val started = System.currentTimeMillis()
        try {
            val gray = image.toDownscaledGray()
            if (gray != null) {
                onMatchCount(countMatches(gray))
                gray.release()
            }
            lastRunAtMs.set(System.currentTimeMillis())
            frameCount.incrementAndGet()
            totalProcessingMs.addAndGet(System.currentTimeMillis() - started)
        } catch (t: Throwable) {
            // a hint that crashes the preview is worse than no hint
            Log.w(TAG, "framing hint failed: ${t.javaClass.simpleName}", t)
        } finally {
            inFlight.set(false)
            image.close()
        }
    }

    /** ORB + ratio test against the reference. Count only; no geometry. */
    private fun countMatches(gray: Mat): Int {
        if (referenceDescriptors.empty()) return 0
        val keypoints = MatOfKeyPoint()
        val descriptors = Mat()
        orb.detectAndCompute(gray, Mat(), keypoints, descriptors)
        try {
            if (descriptors.empty() || descriptors.rows() < 2) return 0
            val knn = ArrayList<MatOfDMatch>()
            matcher.knnMatch(descriptors, referenceDescriptors, knn, 2)
            var good = 0
            for (pair in knn) {
                val candidates = pair.toArray()
                if (candidates.size != 2) continue
                if (candidates[0].distance <
                    AlignmentConfig.RATIO_TEST * candidates[1].distance) good++
            }
            return good
        } finally {
            keypoints.release()
            descriptors.release()
        }
    }

    companion object {
        private const val TAG = "FramingHint"

        /** Long edge the analysis runs at. Preview frames, not stored photos. */
        const val ANALYSIS_LONG_EDGE = 320

        /** Fewer features than alignment uses: this is a count, not a fit. */
        const val ANALYSIS_FEATURES = 500

        const val MAX_FPS = 2
        const val MIN_INTERVAL_MS = 1000L / MAX_FPS

        /**
         * Y plane of a YUV_420_888 frame as a downscaled greyscale Mat.
         *
         * The luma plane IS the greyscale image, so there is no colour
         * conversion to pay for. This never touches a stored photograph -
         * only live preview frames - so the BGR rule in PORT-NOTES section 2
         * does not apply: there is no colour here to get wrong.
         */
        fun ImageProxy.toDownscaledGray(): Mat? {
            if (format != android.graphics.ImageFormat.YUV_420_888) return null
            val plane = planes[0]
            val buffer = plane.buffer
            val rowStride = plane.rowStride
            val full = Mat(height, width, CvType.CV_8UC1)
            val row = ByteArray(width)
            for (y in 0 until height) {
                buffer.position(y * rowStride)
                buffer.get(row, 0, width)
                full.put(y, 0, row)
            }
            val scaled = downscale(full)
            if (scaled !== full) full.release()
            return scaled
        }

        /** Shrink so the long edge is at most ANALYSIS_LONG_EDGE. */
        fun downscale(src: Mat): Mat {
            val longEdge = maxOf(src.rows(), src.cols())
            if (longEdge <= ANALYSIS_LONG_EDGE) return src
            val factor = ANALYSIS_LONG_EDGE.toDouble() / longEdge
            val out = Mat()
            Imgproc.resize(
                src, out,
                Size((src.cols() * factor), (src.rows() * factor)),
                0.0, 0.0, Imgproc.INTER_AREA)
            return out
        }

        /**
         * Descriptors for the reference photograph, at the same scale the
         * preview will be analysed at.
         *
         * Decoded with Imgcodecs.imdecode via the repository, because this one
         * IS measured. The ghost overlay loads the same file again with
         * BitmapFactory, because that copy is only displayed. Two loads of one
         * file, on purpose: PORT-NOTES section 2.
         */
        fun referenceDescriptors(
            repository: PhotoRepository,
            reference: PhotoEntity,
        ): Mat {
            val bgr = repository.readBgr(reference)
            if (bgr.empty()) return Mat()
            val gray = Mat()
            Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY)
            bgr.release()
            val scaled = downscale(gray)
            if (scaled !== gray) gray.release()

            val orb = ORB.create(ANALYSIS_FEATURES)
            val keypoints = MatOfKeyPoint()
            val descriptors = Mat()
            orb.detectAndCompute(scaled, Mat(), keypoints, descriptors)
            scaled.release()
            keypoints.release()
            return descriptors
        }
    }
}
