package com.oraltwin.app.data

import android.content.Context
import android.graphics.BitmapFactory
import androidx.room.Room
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.withContext
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.imgcodecs.Imgcodecs
import java.io.File
import java.text.SimpleDateFormat
import java.util.Locale

/**
 * The only route by which a photograph enters or leaves storage.
 *
 * WHERE THE FILES GO, AND WHY. Everything is written under
 * Context.filesDir/photos - app-internal storage. Not MediaStore, not
 * getExternalFilesDir, not the shared Pictures collection. These are
 * intraoral photographs: they must not appear in the camera roll, must not be
 * picked up by a gallery scanner, and must not be swept into a cloud photo
 * backup. App-internal storage is private to the app, excluded from the media
 * scanner, and removed when the app is uninstalled.
 *
 * android:allowBackup="false" in the manifest closes the other door: without
 * it, Android's auto-backup would copy filesDir to the user's Google Drive.
 */
class PhotoRepository(private val context: Context) {

    private val db: OralTwinDatabase by lazy {
        Room.databaseBuilder(context, OralTwinDatabase::class.java, DB_NAME).build()
    }

    private val dao: PhotoDao get() = db.photoDao()

    val photosDir: File
        get() = File(context.filesDir, PHOTO_DIR).apply { if (!exists()) mkdirs() }

    fun fileFor(photo: PhotoEntity): File = File(photosDir, photo.filename)

    fun observeAll(): Flow<List<PhotoEntity>> = dao.observeAll()

    suspend fun all(): List<PhotoEntity> = dao.all()

    suspend fun count(): Int = dao.count()

    /**
     * Write JPEG bytes to internal storage and record the row.
     *
     * The file is written first and the row second, because the reverse order
     * can leave a row pointing at a file that was never created. The opposite
     * failure - a file with no row - is recoverable and is cleaned up by
     * [removeOrphanFiles].
     */
    suspend fun save(jpegBytes: ByteArray, capturedAtEpochMs: Long = System.currentTimeMillis()):
        PhotoEntity = withContext(Dispatchers.IO) {
        val name = filenameFor(capturedAtEpochMs)
        val file = File(photosDir, name)
        file.writeBytes(jpegBytes)

        // dimensions without decoding the pixels: these are full-resolution
        // camera frames and inJustDecodeBounds keeps them out of memory
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.absolutePath, bounds)

        val entity = PhotoEntity(
            filename = name,
            capturedAtEpochMs = capturedAtEpochMs,
            width = bounds.outWidth.coerceAtLeast(0),
            height = bounds.outHeight.coerceAtLeast(0),
        )
        val id = dao.insert(entity)
        entity.copy(id = id)
    }

    /**
     * Remove both the row and the file.
     *
     * Returns true only if the row went and the file is genuinely gone from
     * disk. A photograph the user deletes must actually be deleted, so this
     * checks rather than assuming File.delete() succeeded.
     */
    suspend fun delete(photo: PhotoEntity): Boolean = withContext(Dispatchers.IO) {
        val file = fileFor(photo)
        val fileGone = !file.exists() || file.delete()
        dao.deleteById(photo.id)
        fileGone && !file.exists() && dao.byId(photo.id) == null
    }

    /** Files with no row - possible if the process died between write and insert. */
    suspend fun removeOrphanFiles(): Int = withContext(Dispatchers.IO) {
        val known = dao.all().map { it.filename }.toSet()
        var removed = 0
        photosDir.listFiles()?.forEach { file ->
            if (file.isFile && file.name !in known && file.delete()) removed++
        }
        removed
    }

    /**
     * Read a stored photograph as an OpenCV BGR Mat.
     *
     * Imgcodecs.imdecode, never BitmapFactory, wherever OpenCV touches these
     * images. imdecode reproduces cv2.imread including BGR channel order;
     * going via a Bitmap yields RGBA, which swaps red and blue and moves every
     * hue, silently breaking the HSV mouth thresholds. Documented at length in
     * android/PORT-NOTES.md, section 2.
     *
     * BitmapFactory is still correct for DISPLAY - a thumbnail is not analysed
     * - and is used for that in the timeline.
     */
    fun readBgr(photo: PhotoEntity): Mat {
        val bytes = fileFor(photo).readBytes()
        val buf = MatOfByte(*bytes)
        val mat = Imgcodecs.imdecode(buf, Imgcodecs.IMREAD_COLOR)
        buf.release()
        return mat
    }

    fun close() = db.close()

    companion object {
        const val DB_NAME = "oraltwin.db"
        const val PHOTO_DIR = "photos"

        fun filenameFor(epochMs: Long): String {
            val stamp = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US)
                .format(java.util.Date(epochMs))
            return "visit_$stamp.jpg"
        }
    }
}
