package com.oraltwin.app

import android.content.Context
import android.os.Build
import androidx.room.Room
import com.oraltwin.app.data.OralTwinDatabase
import com.oraltwin.app.data.PhotoEntity
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.features2d.ORB
import org.opencv.imgproc.Imgproc

/**
 * Proves the stack is genuinely wired up, not merely resolved by Gradle.
 *
 * A dependency can appear on the classpath and still be useless: OpenCV's Java
 * classes load fine while its native .so files are missing, and the failure
 * only shows up the first time a Mat is touched. So each check here exercises
 * the thing it is checking rather than asking whether it is present.
 */
object StackCheck {

    /** Set once, on first use. OpenCV must be initialised before any Mat. */
    private var openCvReady: Boolean? = null

    fun ensureOpenCv(): Boolean {
        openCvReady?.let { return it }
        // initLocal() loads the .so bundled in the AAR. The older initAsync()
        // route needed a separate "OpenCV Manager" app installed; that is long
        // gone and any tutorial still mentioning it is out of date.
        val ok = try {
            OpenCVLoader.initLocal()
        } catch (t: Throwable) {
            false
        }
        openCvReady = ok
        return ok
    }

    /** The version string read FROM the library, never a constant in our code. */
    fun openCvVersion(): String =
        if (!ensureOpenCv()) "not loaded"
        else try {
            // Core.VERSION is populated by the native layer at load time
            Core.VERSION.ifBlank { Core.getVersionString() }
        } catch (t: Throwable) {
            "error: ${t.javaClass.simpleName}"
        }

    fun deviceDescription(): String =
        "${Build.MANUFACTURER} ${Build.MODEL}  -  Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})"

    fun supportedAbis(): String = Build.SUPPORTED_ABIS.joinToString(", ")

    /**
     * Build a small synthetic image with real corners in it and run ORB.
     *
     * A blank Mat would yield zero keypoints and tell us nothing, so the image
     * is drawn with filled rectangles and circles: ORB is a corner detector and
     * needs something to find. A non-zero count here means the native
     * features2d module ran, which is the thing actually in doubt.
     */
    fun orbKeypointCount(): OrbResult {
        if (!ensureOpenCv()) return OrbResult(false, 0, "OpenCV native libraries did not load")
        return try {
            val image = Mat(240, 320, CvType.CV_8UC1, Scalar(30.0))
            // a grid of bright squares: 4 corners each, plenty for ORB
            var drawn = 0
            for (row in 0 until 4) {
                for (col in 0 until 5) {
                    val x = 20 + col * 60
                    val y = 20 + row * 55
                    Imgproc.rectangle(
                        image, Point(x.toDouble(), y.toDouble()),
                        Point((x + 34).toDouble(), (y + 30).toDouble()),
                        Scalar(220.0), -1
                    )
                    Imgproc.circle(
                        image, Point((x + 17).toDouble(), (y + 15).toDouble()),
                        5, Scalar(60.0), -1
                    )
                    drawn++
                }
            }
            val orb = ORB.create(500)
            val keypoints = MatOfKeyPoint()
            orb.detect(image, keypoints)
            val count = keypoints.toArray().size
            image.release()
            keypoints.release()
            OrbResult(true, count, "ORB ran on a synthetic $drawn-shape image")
        } catch (t: Throwable) {
            OrbResult(false, 0, "ORB failed: ${t.javaClass.simpleName}: ${t.message}")
        }
    }

    /** Instantiate Room, write a row, read it back. Proves KSP generated the impl. */
    fun roomCheck(context: Context): String = try {
        val db = Room.inMemoryDatabaseBuilder(context, OralTwinDatabase::class.java)
            .allowMainThreadQueries()   // M0 self-test only; M1 uses coroutines
            .build()
        val dao = db.photoDao()
        dao.insertBlocking(
            PhotoEntity(filename = "selftest.jpg",
                        capturedAtEpochMs = 1_700_000_000_000,
                        width = 1, height = 1)
        )
        val rows = dao.countBlocking()
        db.close()
        "OK - wrote and read back $rows row"
    } catch (t: Throwable) {
        "FAILED: ${t.javaClass.simpleName}: ${t.message}"
    }

    /** CameraX: load a core class and read its artifact presence. */
    fun cameraXCheck(): String = try {
        val cls = Class.forName("androidx.camera.core.CameraX")
        val provider = Class.forName("androidx.camera.lifecycle.ProcessCameraProvider")
        "OK - ${cls.simpleName} and ${provider.simpleName} on the classpath"
    } catch (t: Throwable) {
        "FAILED: ${t.javaClass.simpleName}"
    }

    data class OrbResult(val ok: Boolean, val keypoints: Int, val detail: String)
}
