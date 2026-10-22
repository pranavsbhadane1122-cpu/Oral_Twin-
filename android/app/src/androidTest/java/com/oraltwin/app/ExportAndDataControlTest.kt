package com.oraltwin.app

import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import androidx.core.content.FileProvider
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.oraltwin.app.comparison.ComparabilityCode
import com.oraltwin.app.comparison.ComparisonResult
import com.oraltwin.app.data.PhotoRepository
import com.oraltwin.app.export.ComparisonExport
import com.oraltwin.app.export.ExportStrings
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Export goes out through the share sheet, and nothing about it needs the
 * network. Delete-all actually empties storage.
 *
 * Both assertions are about END STATE. "The delete calls were made" is not the
 * claim the user cares about.
 */
@RunWith(AndroidJUnit4::class)
class ExportAndDataControlTest {

    private val context =
        ApplicationProvider.getApplicationContext<android.content.Context>()
    private lateinit var repository: PhotoRepository

    private fun sampleJpeg(): ByteArray {
        val bitmap = Bitmap.createBitmap(48, 36, Bitmap.Config.ARGB_8888)
        bitmap.eraseColor(Color.rgb(180, 120, 110))
        val out = java.io.ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out)
        bitmap.recycle()
        return out.toByteArray()
    }

    @Before
    fun setUp() {
        context.deleteDatabase(PhotoRepository.DB_NAME)
        File(context.filesDir, PhotoRepository.PHOTO_DIR).deleteRecursively()
        File(context.filesDir, PhotoRepository.EXPORT_DIR).deleteRecursively()
        repository = PhotoRepository(context)
    }

    @After
    fun tearDown() {
        repository.close()
        context.deleteDatabase(PhotoRepository.DB_NAME)
        File(context.filesDir, PhotoRepository.PHOTO_DIR).deleteRecursively()
        File(context.filesDir, PhotoRepository.EXPORT_DIR).deleteRecursively()
    }

    // -------------------------------------------------------------- export

    @Test
    fun exporting_produces_a_content_uri_through_FileProvider() = runBlocking {
        val older = repository.save(sampleJpeg(), capturedAtEpochMs = 1_000)
        val newer = repository.save(sampleJpeg(), capturedAtEpochMs = 2_000)
        val refused = ComparisonResult.refuse(ComparabilityCode.ALIGNMENT_FAILED)

        val bitmap = ComparisonExport.render(repository, older, newer, refused)
        val uri = ComparisonExport.writeForSharing(context, bitmap)

        assertEquals("content", uri.scheme)
        assertEquals(context.packageName + ComparisonExport.AUTHORITY_SUFFIX,
                     uri.authority)
        // the grant must actually resolve to readable bytes
        context.contentResolver.openInputStream(uri).use { stream ->
            assertNotNull("the shared URI must be readable", stream)
            assertTrue("the exported file must not be empty",
                       (stream?.available() ?: 0) > 0)
        }
    }

    @Test
    fun the_share_intent_carries_the_uri_and_grants_read() = runBlocking {
        val older = repository.save(sampleJpeg(), capturedAtEpochMs = 1_000)
        val newer = repository.save(sampleJpeg(), capturedAtEpochMs = 2_000)
        val result = ComparisonResult.refuse(ComparabilityCode.UNRELIABLE_LOW_CONFIDENCE)
        val uri = ComparisonExport.writeForSharing(
            context, ComparisonExport.render(repository, older, newer, result))

        val intent = ComparisonExport.shareIntent(uri)

        assertEquals(android.content.Intent.ACTION_SEND, intent.action)
        assertEquals("image/png", intent.type)
        assertTrue("the read grant must be set",
                   intent.flags and android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION != 0)
        assertEquals(ExportStrings.SHARE_SUBJECT,
                     intent.getStringExtra(android.content.Intent.EXTRA_SUBJECT))
    }

    @Test
    fun a_refusal_is_exportable_and_uses_the_refusal_templates() = runBlocking {
        val refused = ComparisonResult.refuse(ComparabilityCode.PARTIAL_OUT_OF_FRAME)
        val templates = ComparisonExport.templatesFor(refused)

        assertEquals(ExportStrings.refusedVariant, templates)
        assertTrue("a refusal export must still carry the disclaimer",
                   templates.contains(ExportStrings.DISCLAIMER))
        assertFalse("a refusal export must not claim an overlay",
                    templates.contains(ExportStrings.LABEL_OVERLAY))
    }

    @Test
    fun only_the_exports_directory_is_shareable() = runBlocking {
        // a URI must not be obtainable for a stored photograph: file_paths.xml
        // exposes exports/ alone, so the photo store stays unreachable
        val photo = repository.save(sampleJpeg())
        var threw = false
        try {
            FileProvider.getUriForFile(
                context, context.packageName + ComparisonExport.AUTHORITY_SUFFIX,
                repository.fileFor(photo))
        } catch (expected: IllegalArgumentException) {
            threw = true
        }
        assertTrue("the photo store must not be reachable through FileProvider",
                   threw)
    }

    @Test
    fun exporting_does_not_require_internet_access() {
        val info = context.packageManager.getPackageInfo(
            context.packageName, PackageManager.GET_PERMISSIONS)
        val permissions = (info.requestedPermissions ?: emptyArray()).toList()
        assertFalse(
            "the export path must not have introduced INTERNET; found $permissions",
            permissions.contains(android.Manifest.permission.INTERNET),
        )
    }

    // ---------------------------------------------------------- delete all

    @Test
    fun delete_all_leaves_filesDir_and_room_empty() = runBlocking {
        repository.save(sampleJpeg(), capturedAtEpochMs = 1_000)
        repository.save(sampleJpeg(), capturedAtEpochMs = 2_000)
        repository.save(sampleJpeg(), capturedAtEpochMs = 3_000)
        // and an export sitting in the share directory
        val bitmap = Bitmap.createBitmap(8, 8, Bitmap.Config.ARGB_8888)
        ComparisonExport.writeForSharing(context, bitmap)
        assertEquals(3, repository.count())

        val ok = repository.deleteAll()

        assertTrue("deleteAll must confirm the end state, not just run", ok)
        assertEquals("no rows may remain", 0, repository.count())
        assertEquals("no photo files may remain", 0,
                     repository.photosDir.listFiles()?.size ?: 0)
        assertEquals("no exported files may remain", 0,
                     repository.exportsDir.listFiles()?.size ?: 0)
    }

    @Test
    fun delete_all_survives_a_restart() = runBlocking {
        repository.save(sampleJpeg())
        repository.deleteAll()
        repository.close()

        val reopened = PhotoRepository(context)
        try {
            assertEquals(0, reopened.count())
            assertEquals(0, reopened.photosDir.listFiles()?.size ?: 0)
        } finally {
            reopened.close()
        }
        repository = PhotoRepository(context)
    }

    @Test
    fun delete_all_on_empty_storage_is_harmless() = runBlocking {
        assertTrue(repository.deleteAll())
        assertEquals(0, repository.count())
    }
}

/**
 * The self-check must not be reachable in a release build.
 *
 * screensFor() takes the debug flag rather than reading BuildConfig inline,
 * which is what makes the release shape testable without building a release.
 */
@RunWith(AndroidJUnit4::class)
class SelfCheckVisibilityTest {

    @Test
    fun a_release_build_has_no_checks_tab() {
        val release = screensFor(isDebug = false)
        assertFalse("the self-check must be absent from a release build: $release",
                    release.contains("Checks"))
    }

    @Test
    fun a_debug_build_keeps_the_checks_tab_for_the_device_gate() {
        val debug = screensFor(isDebug = true)
        assertTrue("the self-check must stay reachable in debug: $debug",
                   debug.contains("Checks"))
    }

    @Test
    fun the_other_tabs_are_present_in_both() {
        for (screens in listOf(screensFor(false), screensFor(true))) {
            assertTrue(screens.contains("Capture"))
            assertTrue(screens.contains("Timeline"))
            assertTrue(screens.contains("About"))
        }
    }

    @Test
    fun release_has_exactly_one_tab_fewer() {
        assertEquals(screensFor(true).size - 1, screensFor(false).size)
    }
}
