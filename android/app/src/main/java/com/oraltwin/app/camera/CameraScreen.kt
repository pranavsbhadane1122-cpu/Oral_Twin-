package com.oraltwin.app.camera

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
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
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.oraltwin.app.R
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

@Composable
fun CameraScreen(
    onCaptured: (ByteArray) -> Unit,
    modifier: Modifier = Modifier,
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

    Column(modifier = modifier.fillMaxWidth(),
           verticalArrangement = Arrangement.spacedBy(10.dp)) {

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
                            provider.bindToLifecycle(
                                lifecycleOwner,
                                CameraSelector.DEFAULT_BACK_CAMERA,
                                preview,
                                imageCapture,
                            )
                        } catch (t: Throwable) {
                            error = "camera failed to start: ${t.javaClass.simpleName}"
                        }
                    }, ContextCompat.getMainExecutor(ctx))
                    previewView
                },
            )
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
