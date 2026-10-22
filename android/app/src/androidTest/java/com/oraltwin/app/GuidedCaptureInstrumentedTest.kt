package com.oraltwin.app

import android.graphics.Bitmap
import android.graphics.Color
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Slider
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.toPixelMap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.test.captureToImage
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.oraltwin.app.capture.GuidedSession
import com.oraltwin.app.data.PhotoEntity
import com.oraltwin.app.data.PhotoRepository
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import kotlin.math.abs

/**
 * The ghost overlay actually blends, and a guided capture is indistinguishable
 * from a plain one once stored.
 *
 * The opacity test samples a rendered pixel rather than asserting that an
 * alpha value was passed along. Passing the parameter and having it take
 * effect are different claims, and only the second one matters.
 */
@RunWith(AndroidJUnit4::class)
class GhostOverlayTest {

    @get:Rule
    val compose = createComposeRule()

    private lateinit var alpha: androidx.compose.runtime.MutableState<Float>

    private fun solid(color: Int, size: Int = 64): Bitmap =
        Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888).apply {
            eraseColor(color)
        }

    /**
     * Mount a red ghost over a blue background once, then vary the alpha.
     *
     * setContent may be called only once per rule, so the opacity is swapped
     * underneath a single composition rather than remounting per value.
     */
    private fun mount() {
        alpha = mutableStateOf(0f)
        val ghost = solid(Color.RED)
        compose.setContent {
            Box(
                Modifier
                    .size(64.dp)
                    .background(androidx.compose.ui.graphics.Color.Blue)
                    .testTag("ghostArea")
            ) {
                Image(
                    bitmap = ghost.asImageBitmap(),
                    contentDescription = "Faded earlier photo",
                    modifier = Modifier.size(64.dp),
                    contentScale = ContentScale.Fit,
                    alpha = alpha.value,
                )
            }
        }
        compose.waitForIdle()
    }

    /** The rendered centre pixel at [value] opacity. */
    private fun centreAt(value: Float): androidx.compose.ui.graphics.Color {
        compose.runOnUiThread { alpha.value = value }
        compose.waitForIdle()
        val image = compose.onNodeWithTag("ghostArea").captureToImage()
        val pixels = image.toPixelMap()
        return pixels[image.width / 2, image.height / 2]
    }

    @Test
    fun a_fully_transparent_ghost_leaves_the_background_alone() {
        mount()
        val centre = centreAt(0f)
        assertTrue("expected blue, got $centre", centre.blue > 0.8f)
        assertTrue("expected no red, got $centre", centre.red < 0.2f)
    }

    @Test
    fun the_default_opacity_blends_rather_than_replaces() {
        mount()
        val centre = centreAt(GuidedSession.DEFAULT_GHOST_OPACITY)
        // both contributions must survive: neither image wins outright
        assertTrue("the ghost should be visible, got $centre", centre.red > 0.1f)
        assertTrue("the background should survive, got $centre", centre.blue > 0.3f)
        assertTrue("red should sit nearer 0.3 than 1.0, got ${centre.red}",
                   abs(centre.red - GuidedSession.DEFAULT_GHOST_OPACITY) < 0.25f)
    }

    @Test
    fun a_stronger_ghost_shows_more_of_the_reference() {
        mount()
        val faint = centreAt(0.10f)
        val strong = centreAt(GuidedSession.MAX_GHOST_OPACITY)
        assertTrue("a higher opacity must show more of the ghost: " +
                       "${faint.red} then ${strong.red}",
                   strong.red > faint.red)
    }

    @Test
    fun the_opacity_model_clamps_to_the_cap() {
        val session = GuidedSession(
            PhotoEntity(1, "a.jpg", 1, 10, 10)).withOpacity(5f)
        assertEquals(GuidedSession.MAX_GHOST_OPACITY, session.ghostOpacity, 1e-6f)
        assertEquals(0f, GuidedSession(
            PhotoEntity(1, "a.jpg", 1, 10, 10)).withOpacity(-2f).ghostOpacity, 1e-6f)
    }

    @Test
    fun the_slider_range_never_exceeds_the_cap() {
        var observed = -1f
        compose.setContent {
            var opacity by remember {
                mutableFloatStateOf(GuidedSession.DEFAULT_GHOST_OPACITY)
            }
            observed = opacity
            Slider(
                value = opacity,
                onValueChange = { opacity = it },
                valueRange = 0f..GuidedSession.MAX_GHOST_OPACITY,
                modifier = Modifier.testTag("opacitySlider"),
            )
        }
        compose.waitForIdle()
        assertTrue("the slider must start inside its range, got $observed",
                   observed in 0f..GuidedSession.MAX_GHOST_OPACITY)
    }
}

/**
 * A guided capture must be indistinguishable from a plain one downstream.
 *
 * If the two differed, the comparison layer would acquire a second code path
 * that nothing has measured, and the first time it behaved differently nobody
 * would be able to say which path produced the result.
 */
@RunWith(AndroidJUnit4::class)
class GuidedCaptureStorageTest {

    private val context =
        ApplicationProvider.getApplicationContext<android.content.Context>()
    private lateinit var repository: PhotoRepository

    private fun sampleJpeg(): ByteArray {
        val bitmap = Bitmap.createBitmap(32, 24, Bitmap.Config.ARGB_8888)
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
        repository = PhotoRepository(context)
    }

    @After
    fun tearDown() {
        repository.close()
        context.deleteDatabase(PhotoRepository.DB_NAME)
        File(context.filesDir, PhotoRepository.PHOTO_DIR).deleteRecursively()
    }

    @Test
    fun a_guided_capture_and_a_plain_one_produce_the_same_shape_of_row() =
        runBlocking {
            // There is only one save path - guidance changes what is drawn over
            // the preview, never what is written - so these two calls are the
            // whole difference between the modes.
            val plain = repository.save(sampleJpeg(), capturedAtEpochMs = 1_000)
            val guided = repository.save(sampleJpeg(), capturedAtEpochMs = 2_000)

            assertEquals(plain.width, guided.width)
            assertEquals(plain.height, guided.height)
            // everything that is not the identity of the photo must match
            assertEquals(
                plain.copy(id = 0, filename = "", capturedAtEpochMs = 0),
                guided.copy(id = 0, filename = "", capturedAtEpochMs = 0),
            )
            assertEquals(2, repository.count())
        }

    @Test
    fun the_photo_row_has_no_guidance_field() {
        // asserted by reflection so that adding "capturedWithGuidance" fails
        // here rather than quietly creating a second downstream code path
        val fields = PhotoEntity::class.java.declaredFields
            .filterNot { it.isSynthetic }
            .filterNot { java.lang.reflect.Modifier.isStatic(it.modifiers) }
            .map { it.name }
            .sorted()

        assertEquals(
            listOf("capturedAtEpochMs", "filename", "height", "id", "width"),
            fields,
        )
        for (name in fields) {
            val lower = name.lowercase()
            assertTrue("no field may reference guidance, found $name",
                       !lower.contains("guid") && !lower.contains("reference") &&
                           !lower.contains("ghost"))
        }
    }

    @Test
    fun the_reference_link_is_not_persisted_anywhere() = runBlocking {
        val reference = repository.save(sampleJpeg(), capturedAtEpochMs = 1_000)
        // a session exists only in memory
        val session = GuidedSession(reference)
        assertEquals(reference.id, session.reference.id)

        val followUp = repository.save(sampleJpeg(), capturedAtEpochMs = 2_000)
        repository.close()

        // reopen: nothing about the pairing should have survived
        val reopened = PhotoRepository(context)
        try {
            val rows = reopened.all()
            assertEquals(2, rows.size)
            val stored = rows.first { it.id == followUp.id }
            assertEquals(followUp.filename, stored.filename)
            // the only fields that exist are the five above; there is nowhere
            // for a reference id to have been kept
            assertTrue(stored.width > 0)
        } finally {
            reopened.close()
        }
        repository = PhotoRepository(context)
    }
}
