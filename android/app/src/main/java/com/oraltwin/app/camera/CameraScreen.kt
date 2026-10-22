package com.oraltwin.app.camera

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Row
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.core.content.ContextCompat
import com.oraltwin.app.R
import com.oraltwin.app.capture.FramingHintAnalyzer
import com.oraltwin.app.capture.GuidedCaptureStrings
import com.oraltwin.app.capture.GuidedSession
import com.oraltwin.app.capture.MatchIndicator
import com.oraltwin.app.data.PhotoRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.opencv.core.Mat
import java.util.concurrent.Executors
import java.util.concurrent.Executor

/**
 * Rear-camera capture. Tap to take a photograph.
 *
 * The JPEG bytes are handed straight to the caller rather than written by
 * CameraX itself: ImageCapture.OutputFileOptions can target MediaStore, and
 * routing every write through PhotoRepository is what keeps these images out
 * of the shared gallery by construction rather than by remembering to.
 */

fun hasCameraPermission(context: Context): Boolean =
    ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
        PackageManager.PERMISSION_GRANTED

/**
 * Capture, optionally guided by an earlier photograph.
 *
 * ONE CODE PATH. Guided and plain capture differ only in what is drawn over
 * the preview; the bytes travel the same way to the same [onCaptured]. A
 * photograph taken with guidance is indistinguishable downstream from one
 * taken without, which is the point: a second path would be a second thing to
 * measure, and nothing has measured it.
 */
@Composable
fun CameraScreen(
    onCaptured: (ByteArray) -> Unit,
    modifier: Modifier = Modifier,
    session: GuidedSession? = null,
    repository: PhotoRepository? = null,
    onEndGuidance: () -> Unit = {},
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    var granted by remember { mutableStateOf(hasCameraPermission(context)) }
    var askedOnce by remember { mutableStateOf(false) }
    var error by remember { mutableStateOf<String?>(null) }
    var capturing by remember { mutableStateOf(false) }

    val launcher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { result ->
        granted = result
        askedOnce = true
    }

    if (!granted) {
        Column(
            modifier = modifier.fillMaxWidth().padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(14.dp),
                       verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text("Camera access", style = MaterialTheme.typography.titleMedium)
                    // the rationale, in the user's terms, before the system dialog
                    Text(stringResourceSafe(context, R.string.camera_rationale))
                }
            }
            Button(
                onClick = { launcher.launch(Manifest.permission.CAMERA) },
                modifier = Modifier.fillMaxWidth(),
            ) { Text("Allow camera access") }

            if (askedOnce) {
                Text(
                    "Permission was not granted. You can change it in Settings > " +
                        "Apps > OralTwin > Permissions.",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }
        return
    }

    val imageCapture = remember {
        ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
            .build()
    }
    val executor = remember { ContextCompat.getMainExecutor(context) }
    val analysisExecutor = remember { Executors.newSingleThreadExecutor() }

    // --- guided capture state --------------------------------------------
    var ghostOpacity by remember(session?.reference?.id) {
        mutableFloatStateOf(session?.ghostOpacity ?: GuidedSession.DEFAULT_GHOST_OPACITY)
    }
    var matchCount by remember(session?.reference?.id) { mutableIntStateOf(0) }

    // The ghost is DISPLAY ONLY, so BitmapFactory is correct here. The same
    // file is loaded a second time, with Imgcodecs.imdecode, for the framing
    // hint, because that copy is measured. Two loads of one file, on purpose:
    // PORT-NOTES section 2.
    val ghostBitmap = remember(session?.reference?.id) {
        val reference = session?.reference
        val repo = repository
        if (reference == null || repo == null) null
        else runCatching {
            android.graphics.BitmapFactory.decodeFile(
                repo.fileFor(reference).absolutePath,
                android.graphics.BitmapFactory.Options().apply { inSampleSize = 2 },
            )
        }.getOrNull()
    }

    // descriptors for the hint, computed once per session, off the main thread
    var referenceDescriptors by remember(session?.reference?.id) {
        mutableStateOf<Mat?>(null)
    }
    LaunchedEffect(session?.reference?.id) {
        val reference = session?.reference
        val repo = repository
        referenceDescriptors = if (reference == null || repo == null) null
        else withContext(Dispatchers.Default) {
            runCatching {
                FramingHintAnalyzer.referenceDescriptors(repo, reference)
            }.getOrNull()
        }
    }

    Column(modifier = modifier.fillMaxWidth(),
           verticalArrangement = Arrangement.spacedBy(10.dp)) {

        if (session != null) {
            Column(Modifier.padding(horizontal = 20.dp),
                   verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text(GuidedCaptureStrings.TITLE,
                     style = MaterialTheme.typography.titleMedium)
                Text(GuidedCaptureStrings.REFERENCE_NOTE,
                     style = MaterialTheme.typography.bodySmall)
            }
        }

        Box(Modifier.fillMaxWidth().padding(horizontal = 20.dp)) {
            AndroidView(
                modifier = Modifier.fillMaxWidth(),
                factory = { ctx ->
                    val previewView = PreviewView(ctx).apply {
                        scaleType = PreviewView.ScaleType.FIT_CENTER
                    }
                    val providerFuture = ProcessCameraProvider.getInstance(ctx)
                    providerFuture.addListener({
                        try {
                            val provider = providerFuture.get()
                            val preview = Preview.Builder().build().also {
                                it.setSurfaceProvider(previewView.surfaceProvider)
                            }
                            provider.unbindAll()

                            val descriptors = referenceDescriptors
                            if (descriptors != null && !descriptors.empty()) {
                                val analysis = ImageAnalysis.Builder()
                                    // Drop frames rather than build a backlog.
                                    // A queue makes a slow phone fall further
                                    // behind the longer the preview runs.
                                    .setBackpressureStrategy(
                                        ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                                    .build()
                                analysis.setAnalyzer(
                                    analysisExecutor,
                                    FramingHintAnalyzer(descriptors) { count ->
                                        matchCount = count
                                    },
                                )
                                provider.bindToLifecycle(
                                    lifecycleOwner,
                                    CameraSelector.DEFAULT_BACK_CAMERA,
                                    preview, imageCapture, analysis,
                                )
                            } else {
                                provider.bindToLifecycle(
                                    lifecycleOwner,
                                    CameraSelector.DEFAULT_BACK_CAMERA,
                                    preview, imageCapture,
                                )
                            }
                        } catch (t: Throwable) {
                            error = "camera failed to start: ${t.javaClass.simpleName}"
                        }
                    }, ContextCompat.getMainExecutor(ctx))
                    previewView
                },
            )

            // the ghost, drawn over the live preview
            if (session != null && ghostBitmap != null) {
                Image(
                    bitmap = ghostBitmap.asImageBitmap(),
                    contentDescription = GuidedCaptureStrings.GHOST_LABEL,
                    modifier = Modifier.fillMaxWidth(),
                    contentScale = ContentScale.Fit,
                    alpha = ghostOpacity,
                )
            }
        }

        if (session != null) {
            Column(Modifier.padding(horizontal = 20.dp),
                   verticalArrangement = Arrangement.spacedBy(6.dp)) {

                Text(GuidedCaptureStrings.OPACITY_LABEL,
                     style = MaterialTheme.typography.bodySmall)
                Slider(
                    value = ghostOpacity,
                    onValueChange = { ghostOpacity = it },
                    // Capped at half: above this the ghost stops being a guide
                    // and starts being the picture, and the user would be
                    // framing against an image rather than against the camera.
                    valueRange = 0f..GuidedSession.MAX_GHOST_OPACITY,
                )

                if (referenceDescriptors != null) {
                    Text(GuidedCaptureStrings.FRAMING_LABEL,
                         style = MaterialTheme.typography.bodySmall)
                    // Continuous fill, no threshold, no state. The end labels
                    // are fixed: wording that changed as the bar filled would
                    // be a state transition in different clothes.
                    LinearProgressIndicator(
                        progress = { MatchIndicator.fill(matchCount) },
                        modifier = Modifier.fillMaxWidth(),
                    )
                    Row(Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween) {
                        Text(MatchIndicator.LOW_END,
                             style = MaterialTheme.typography.bodySmall)
                        Text(MatchIndicator.HIGH_END,
                             style = MaterialTheme.typography.bodySmall)
                    }
                    Text(GuidedCaptureStrings.HINT_CAVEAT,
                         style = MaterialTheme.typography.bodySmall)
                }

                TextButton(onClick = onEndGuidance) {
                    Text(GuidedCaptureStrings.CANCEL)
                }
            }
        }

        Column(Modifier.padding(horizontal = 20.dp),
               verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(
                onClick = {
                    capturing = true
                    error = null
                    takePicture(imageCapture, executor,
                        onBytes = { bytes ->
                            capturing = false
                            onCaptured(bytes)
                        },
                        onError = {
                            capturing = false
                            error = it
                        })
                },
                enabled = !capturing,
                modifier = Modifier.fillMaxWidth(),
            ) { Text(if (capturing) "Saving..." else "Take photo") }

            error?.let { Text(it, style = MaterialTheme.typography.bodySmall) }

            Text(
                stringResourceSafe(context, R.string.privacy_line),
                style = MaterialTheme.typography.bodySmall,
            )
        }
    }

    DisposableEffect(Unit) {
        onDispose {
            runCatching {
                ProcessCameraProvider.getInstance(context).get().unbindAll()
            }
        }
    }
}

/** Capture into memory, so the only writer of photo files is PhotoRepository. */
private fun takePicture(
    imageCapture: ImageCapture,
    executor: Executor,
    onBytes: (ByteArray) -> Unit,
    onError: (String) -> Unit,
) {
    imageCapture.takePicture(executor, object : ImageCapture.OnImageCapturedCallback() {
        override fun onCaptureSuccess(image: ImageProxy) {
            try {
                onBytes(image.toJpegBytes())
            } catch (t: Throwable) {
                onError("could not read the captured image: ${t.javaClass.simpleName}")
            } finally {
                image.close()
            }
        }

        override fun onError(exception: ImageCaptureException) {
            onError("capture failed: ${exception.message ?: exception.javaClass.simpleName}")
        }
    })
}

/** ImageCapture without OutputFileOptions yields JPEG in plane 0. */
private fun ImageProxy.toJpegBytes(): ByteArray {
    val buffer = planes[0].buffer
    val bytes = ByteArray(buffer.remaining())
    buffer.get(bytes)
    return bytes
}

private fun stringResourceSafe(context: Context, id: Int): String =
    runCatching { context.getString(id) }.getOrDefault("")
