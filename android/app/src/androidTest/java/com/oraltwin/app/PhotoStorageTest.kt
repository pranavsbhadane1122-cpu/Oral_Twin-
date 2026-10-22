package com.oraltwin.app

import android.content.pm.PackageManager
import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.oraltwin.app.data.OralTwinDatabase
import com.oraltwin.app.data.PhotoRepository
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Instrumented tests: capture writes file and row, a restart preserves both,
 * delete removes both.
 *
 * These run on a device or emulator rather than a desktop JVM, which is the
 * whole point: Room's generated code, the real filesystem and the real
 * PackageManager are all involved, and none of them behave the same under a
 * unit test.
 *
 * "Restart" is simulated by closing the database and opening a new
 * PhotoRepository over the same filesDir. That exercises the thing that
 * actually matters - whether state was persisted rather than held in memory -
 * without needing to kill the process.
 */
@RunWith(AndroidJUnit4::class)
class PhotoStorageTest {

    private val context = ApplicationProvider.getApplicationContext<android.content.Context>()
    private lateinit var repository: PhotoRepository

    /** A tiny but genuine JPEG, so BitmapFactory reports real dimensions. */
    private fun sampleJpeg(width: Int = 32, height: Int = 24): ByteArray {
        val bitmap = android.graphics.Bitmap.createBitmap(
            width, height, android.graphics.Bitmap.Config.ARGB_8888)
        bitmap.eraseColor(android.graphics.Color.rgb(180, 120, 110))
        val out = java.io.ByteArrayOutputStream()
        bitmap.compress(android.graphics.Bitmap.CompressFormat.JPEG, 90, out)
        bitmap.recycle()
        return out.toByteArray()
    }

    private fun freshRepository() = PhotoRepository(context)

    @Before
    fun setUp() {
        // start from a clean slate: delete the database and the photo directory
        context.deleteDatabase(PhotoRepository.DB_NAME)
        File(context.filesDir, PhotoRepository.PHOTO_DIR).deleteRecursively()
        repository = freshRepository()
    }

    @After
    fun tearDown() {
        repository.close()
        context.deleteDatabase(PhotoRepository.DB_NAME)
        File(context.filesDir, PhotoRepository.PHOTO_DIR).deleteRecursively()
    }

    // ---------------------------------------------------------------- capture

    @Test
    fun capture_writes_both_a_file_and_a_row() = runBlocking {
        val saved = repository.save(sampleJpeg())

        val file = repository.fileFor(saved)
        assertTrue("the JPEG should exist on disk", file.exists())
        assertTrue("the JPEG should not be empty", file.length() > 0)
        assertEquals(1, repository.count())

        val row = repository.all().single()
        assertEquals(saved.filename, row.filename)
        assertTrue("row id should have been assigned", row.id > 0)
    }

    @Test
    fun capture_records_the_real_dimensions() = runBlocking {
        val saved = repository.save(sampleJpeg(width = 48, height = 36))
        assertEquals(48, saved.width)
        assertEquals(36, saved.height)
    }

    @Test
    fun photos_are_stored_inside_app_internal_storage_only() = runBlocking {
        val saved = repository.save(sampleJpeg())
        val path = repository.fileFor(saved).canonicalPath
        val internal = context.filesDir.canonicalPath

        assertTrue("photo must live under filesDir, got $path",
                   path.startsWith(internal))
        // the paths an intraoral photograph must never be written to
        assertFalse(path.contains("/DCIM"))
        assertFalse(path.contains("/Pictures"))
        assertFalse(path.contains("/storage/emulated/0/Android/media"))
    }

    // ---------------------------------------------------------------- restart

    @Test
    fun a_restart_preserves_the_file_and_the_row() = runBlocking {
        val saved = repository.save(sampleJpeg())
        val filename = saved.filename

        // close everything, as a process death would
        repository.close()

        // a brand-new repository over the same storage: nothing in memory survives
        val reopened = freshRepository()
        try {
            assertEquals("the row should still be there", 1, reopened.count())
            val row = reopened.all().single()
            assertEquals(filename, row.filename)
            assertTrue("the file should still be there",
                       reopened.fileFor(row).exists())
        } finally {
            reopened.close()
        }
        repository = freshRepository()   // so tearDown has something valid to close
    }

    @Test
    fun the_timeline_order_is_newest_first_and_survives_a_restart() = runBlocking {
        val base = System.currentTimeMillis()
        repository.save(sampleJpeg(), capturedAtEpochMs = base - 2_000)
        repository.save(sampleJpeg(), capturedAtEpochMs = base - 1_000)
        repository.save(sampleJpeg(), capturedAtEpochMs = base)
        repository.close()

        val reopened = freshRepository()
        try {
            val stamps = reopened.all().map { it.capturedAtEpochMs }
            assertEquals(listOf(base, base - 1_000, base - 2_000), stamps)
        } finally {
            reopened.close()
        }
        repository = freshRepository()
    }

    // ----------------------------------------------------------------- delete

    @Test
    fun delete_removes_the_row_and_the_file_from_disk() = runBlocking {
        val saved = repository.save(sampleJpeg())
        val file = repository.fileFor(saved)
        assertTrue(file.exists())

        val gone = repository.delete(saved)

        assertTrue("delete() should report success", gone)
        assertEquals("the row should be gone", 0, repository.count())
        assertFalse("the file must actually be gone from disk", file.exists())
    }

    @Test
    fun a_deleted_photo_does_not_come_back_after_a_restart() = runBlocking {
        val saved = repository.save(sampleJpeg())
        val file = repository.fileFor(saved)
        repository.delete(saved)
        repository.close()

        val reopened = freshRepository()
        try {
            assertEquals(0, reopened.count())
            assertFalse(file.exists())
        } finally {
            reopened.close()
        }
        repository = freshRepository()
    }

    @Test
    fun deleting_one_photo_leaves_the_others_alone() = runBlocking {
        val first = repository.save(sampleJpeg(), capturedAtEpochMs = 1_700_000_000_000)
        val second = repository.save(sampleJpeg(), capturedAtEpochMs = 1_700_000_001_000)

        repository.delete(first)

        assertEquals(1, repository.count())
        assertFalse(repository.fileFor(first).exists())
        assertTrue(repository.fileFor(second).exists())
        assertEquals(second.filename, repository.all().single().filename)
    }

    @Test
    fun orphan_files_are_cleaned_up() = runBlocking {
        repository.save(sampleJpeg())
        // a file written without a row, as a crash between the two would leave
        val orphan = File(repository.photosDir, "visit_orphan.jpg")
        orphan.writeBytes(sampleJpeg())
        assertTrue(orphan.exists())

        val removed = repository.removeOrphanFiles()

        assertEquals(1, removed)
        assertFalse(orphan.exists())
        assertEquals(1, repository.count())
    }

    // ------------------------------------------------------------------- Room

    @Test
    fun room_generated_its_implementation() = runBlocking {
        // in-memory, so this says something about Room rather than about files
        val db = Room.inMemoryDatabaseBuilder(context, OralTwinDatabase::class.java).build()
        try {
            val dao = db.photoDao()
            val id = dao.insert(
                com.oraltwin.app.data.PhotoEntity(
                    filename = "x.jpg", capturedAtEpochMs = 1L, width = 2, height = 3)
            )
            assertTrue(id > 0)
            assertEquals(1, dao.count())
            assertNotNull(dao.byId(id))
            dao.deleteById(id)
            assertNull(dao.byId(id))
        } finally {
            db.close()
        }
    }
}

/**
 * The privacy claim, asserted against the installed package.
 *
 * A comment in the manifest is a promise; this is a check. If any dependency
 * ever merges an INTERNET permission in, or someone adds one by hand, this
 * fails rather than the app quietly gaining the ability to upload
 * photographs.
 */
@RunWith(AndroidJUnit4::class)
class PermissionsTest {

    private val context = ApplicationProvider.getApplicationContext<android.content.Context>()

    private fun requestedPermissions(): Array<String> {
        val info = context.packageManager.getPackageInfo(
            context.packageName, PackageManager.GET_PERMISSIONS)
        return info.requestedPermissions ?: emptyArray()
    }

    @Test
    fun the_app_does_not_request_internet_access() {
        val permissions = requestedPermissions().toList()
        assertFalse(
            "INTERNET must not be in the merged manifest; found $permissions",
            permissions.contains(android.Manifest.permission.INTERNET),
        )
        assertFalse(
            "ACCESS_NETWORK_STATE must not be there either; found $permissions",
            permissions.contains(android.Manifest.permission.ACCESS_NETWORK_STATE),
        )
    }

    @Test
    fun the_app_does_not_request_shared_storage_access() {
        val permissions = requestedPermissions().toList()
        for (denied in listOf(
            android.Manifest.permission.WRITE_EXTERNAL_STORAGE,
            android.Manifest.permission.READ_EXTERNAL_STORAGE,
        )) {
            assertFalse("$denied must not be requested; found $permissions",
                        permissions.contains(denied))
        }
    }

    @Test
    fun camera_is_requested_because_the_app_genuinely_needs_it() {
        // the complement of the tests above: this one SHOULD be present, so a
        // vacuous "no permissions at all" result cannot pass the suite
        assertTrue(requestedPermissions().contains(android.Manifest.permission.CAMERA))
    }
}
