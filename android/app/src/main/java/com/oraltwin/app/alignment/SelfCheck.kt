package com.oraltwin.app.alignment

import android.content.Context
import org.json.JSONObject
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.imgcodecs.Imgcodecs

/**
 * Runs the ported alignment over bundled test pairs and compares each result
 * with the Python reference figure for the same pair.
 *
 * This exists because JVM unit tests cannot exercise OpenCV's Android native
 * libraries: a test that passes on a desktop JVM proves nothing about what
 * runs on the phone. So the comparison is done on-device and shown on screen.
 *
 * THE TEST PAIRS ARE NOT PIYARATHNE. They come from data/longitudinal, which
 * is derived from the Kaggle "Oral Diseases" set via
 * data/processed/classification. Verified by tracing all 200 pairs:
 * every source_image resolves under data/processed/classification/test, whose
 * classes (calculus, caries, gingivitis) come from data/raw/oral_diseases.
 * No Piyarathne-derived bytes are in this APK.
 */
object SelfCheck {

    private const val ASSET_DIR = "testpairs"
    private const val MANIFEST = "$ASSET_DIR/manifest.json"

    data class Reference(
        val cornerErrorPx: Double,
        val confidence: Double,
        val nInliers: Int,
    )

    data class Case(
        val pair: String,
        val sourceClass: String,
        val groundTruth: Mat,      // maps visit1 -> visit2, as stored
        val python: Reference,
    )

    data class Outcome(
        val pair: String,
        val sourceClass: String,
        val ok: Boolean,
        val reason: String?,
        val nInliers: Int,
        val confidence: Double,
        val cornerErrorPx: Double?,
        val python: Reference,
        val width: Int,
        val height: Int,
    ) {
        /** Kotlin / Python, or null when the Kotlin run refused. */
        val ratio: Double?
            get() = cornerErrorPx?.let {
                if (python.cornerErrorPx > 0) it / python.cornerErrorPx else null
            }
    }

    fun manifestPresent(context: Context): Boolean = try {
        context.assets.open(MANIFEST).close()
        true
    } catch (t: Throwable) {
        false
    }

    private fun matFromJsonRows(rows: org.json.JSONArray): Mat {
        val m = Mat(3, 3, CvType.CV_64F)
        for (r in 0 until 3) {
            val row = rows.getJSONArray(r)
            for (c in 0 until 3) m.put(r, c, row.getDouble(c))
        }
        return m
    }

    fun loadCases(context: Context): List<Case> {
        val text = context.assets.open(MANIFEST).bufferedReader().use { it.readText() }
        val root = JSONObject(text)
        val pairs = root.getJSONArray("pairs")
        val out = ArrayList<Case>(pairs.length())
        for (i in 0 until pairs.length()) {
            val entry = pairs.getJSONObject(i)
            val ref = entry.getJSONObject("python")
            out.add(
                Case(
                    pair = entry.getString("pair"),
                    sourceClass = entry.optString("source_class", "?"),
                    groundTruth = matFromJsonRows(entry.getJSONArray("homography")),
                    python = Reference(
                        cornerErrorPx = ref.getDouble("corner_error_px"),
                        confidence = ref.getDouble("confidence"),
                        nInliers = ref.getInt("n_inliers"),
                    ),
                )
            )
        }
        return out
    }

    /**
     * Decode a bundled JPEG as BGR.
     *
     * Imgcodecs.imdecode, not BitmapFactory: imdecode reproduces cv2.imread
     * exactly, including the BGR channel order the HSV mouth thresholds were
     * written against. Going via a Bitmap yields RGBA and would invert the hue
     * bands, which looks like a subtle accuracy loss rather than an error.
     */
    private fun readAssetBgr(context: Context, name: String): Mat {
        val bytes = context.assets.open("$ASSET_DIR/$name").use { it.readBytes() }
        val buf = MatOfByte(*bytes)
        val img = Imgcodecs.imdecode(buf, Imgcodecs.IMREAD_COLOR)
        buf.release()
        return img
    }

    fun run(context: Context, case: Case): Outcome {
        val img1 = readAssetBgr(context, "${case.pair}_visit1.jpg")
        val img2 = readAssetBgr(context, "${case.pair}_visit2.jpg")
        if (img1.empty() || img2.empty()) {
            img1.release(); img2.release()
            return Outcome(case.pair, case.sourceClass, false,
                           "could not decode the bundled images", 0, 0.0, null,
                           case.python, 0, 0)
        }

        val result = Alignment.alignPair(img1, img2)

        var cornerError: Double? = null
        if (result.h != null) {
            // ground truth maps visit1 -> visit2; we estimate visit2 -> visit1,
            // so the comparison is against its inverse. This mirrors
            // evaluate_alignment.py, which does np.linalg.inv(gt["homography"]).
            val hTrue = Mat()
            org.opencv.core.Core.invert(case.groundTruth, hTrue)
            cornerError = Homography.cornerError(result.h, hTrue, img1.cols(), img1.rows())
            hTrue.release()
        }

        val outcome = Outcome(
            pair = case.pair,
            sourceClass = case.sourceClass,
            ok = result.h != null,
            reason = result.reason,
            nInliers = result.homographyInfo.nInliers,
            confidence = result.confidence,
            cornerErrorPx = cornerError,
            python = case.python,
            width = img1.cols(),
            height = img1.rows(),
        )

        result.h?.release()
        img1.release(); img2.release()
        return outcome
    }

    fun runAll(context: Context): List<Outcome> =
        loadCases(context).map { case ->
            val outcome = run(context, case)
            case.groundTruth.release()
            outcome
        }

    /** Median of the Kotlin corner errors, over the pairs that aligned. */
    fun medianCornerError(outcomes: List<Outcome>): Double? {
        val values = outcomes.mapNotNull { it.cornerErrorPx }.sorted()
        if (values.isEmpty()) return null
        val mid = values.size / 2
        return if (values.size % 2 == 1) values[mid]
               else (values[mid - 1] + values[mid]) / 2.0
    }

    fun medianPythonCornerError(outcomes: List<Outcome>): Double? {
        val values = outcomes.map { it.python.cornerErrorPx }.sorted()
        if (values.isEmpty()) return null
        val mid = values.size / 2
        return if (values.size % 2 == 1) values[mid]
               else (values[mid - 1] + values[mid]) / 2.0
    }
}
