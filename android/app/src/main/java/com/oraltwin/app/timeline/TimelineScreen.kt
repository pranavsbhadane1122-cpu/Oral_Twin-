package com.oraltwin.app.timeline

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.Image
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.Checkbox
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.oraltwin.app.comparison.PairSelection
import com.oraltwin.app.data.PhotoEntity
import com.oraltwin.app.data.PhotoRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Reverse-chronological list of stored photographs.
 *
 * Backed by a Room Flow, so the list is rebuilt from the database on every
 * launch and survives a full app restart by construction: there is no
 * in-memory collection that could become the real source of truth.
 *
 * Thumbnails are decoded with BitmapFactory and inSampleSize. That is correct
 * HERE because a thumbnail is only displayed, never measured. Anywhere OpenCV
 * touches these images, decoding goes through PhotoRepository.readBgr, which
 * uses Imgcodecs.imdecode for the BGR reason set out in PORT-NOTES.md
 * section 2.
 */

private val STAMP = SimpleDateFormat("d MMM yyyy, HH:mm", Locale.getDefault())

const val PRIVACY_LINE = "Photos stay on this phone. Nothing is uploaded."

@OptIn(ExperimentalFoundationApi::class)
@Composable
fun TimelineScreen(
    repository: PhotoRepository,
    onCompare: (PairSelection) -> Unit = {},
    modifier: Modifier = Modifier,
) {
    val photos by repository.observeAll().collectAsState(initial = emptyList())
    var fullSize by remember { mutableStateOf<PhotoEntity?>(null) }
    var pendingDelete by remember { mutableStateOf<PhotoEntity?>(null) }
    var deleteError by remember { mutableStateOf<String?>(null) }
    var selecting by remember { mutableStateOf(false) }
    val selectedIds = remember { mutableStateListOf<Long>() }
    val scope = rememberCoroutineScope()

    Column(
        modifier = modifier.fillMaxWidth().padding(horizontal = 20.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text("Your photos", style = MaterialTheme.typography.titleLarge)
        // true, and the reason nothing is written outside app-internal storage
        Text(
            PRIVACY_LINE,
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.Medium,
        )

        if (photos.isEmpty()) {
            Text(
                "No photos yet. Take one to start a timeline.",
                style = MaterialTheme.typography.bodySmall,
            )
        } else {
            Text(
                "${photos.size} photo${if (photos.size == 1) "" else "s"}. " +
                    if (selecting) " Choose two to compare."
                    else " Long-press one to delete it.",
                style = MaterialTheme.typography.bodySmall,
            )

            if (photos.size >= 2) {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(onClick = {
                        selecting = !selecting
                        selectedIds.clear()
                    }) {
                        Text(if (selecting) "Cancel" else "Compare two photos")
                    }
                    if (selecting) {
                        val chosen = photos.filter { it.id in selectedIds }
                        val pair = PairSelection.from(chosen)
                        Button(
                            onClick = {
                                pair?.let {
                                    selecting = false
                                    selectedIds.clear()
                                    onCompare(it)
                                }
                            },
                            // exactly two, enforced by PairSelection.from
                            enabled = pair != null,
                        ) { Text("Compare") }
                    }
                }
                if (selecting) {
                    Text(
                        "Which is older is worked out from when each photo was " +
                            "taken, not from the order you tap them.",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }

            deleteError?.let {
                Text(it, style = MaterialTheme.typography.bodySmall)
            }

            LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                items(photos, key = { it.id }) { photo ->
                    Card(
                        Modifier
                            .fillMaxWidth()
                            .combinedClickable(
                                onClick = {
                                    if (selecting) {
                                        if (photo.id in selectedIds) {
                                            selectedIds.remove(photo.id)
                                        } else if (selectedIds.size < 2) {
                                            selectedIds.add(photo.id)
                                        }
                                    } else {
                                        fullSize = photo
                                    }
                                },
                                // delete stays on long-press even while
                                // selecting, so the gesture never changes
                                // meaning under the user
                                onLongClick = { pendingDelete = photo },
                            )
                    ) {
                        Row(
                            Modifier.padding(10.dp),
                            horizontalArrangement = Arrangement.spacedBy(12.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            if (selecting) {
                                Checkbox(
                                    checked = photo.id in selectedIds,
                                    onCheckedChange = { want ->
                                        if (want && selectedIds.size < 2) {
                                            selectedIds.add(photo.id)
                                        } else if (!want) {
                                            selectedIds.remove(photo.id)
                                        }
                                    },
                                )
                            }
                            Thumbnail(repository, photo, Modifier.size(72.dp))
                            Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                                Text(
                                    STAMP.format(Date(photo.capturedAtEpochMs)),
                                    fontWeight = FontWeight.Bold,
                                )
                                Text(
                                    "${photo.width} x ${photo.height}",
                                    style = MaterialTheme.typography.bodySmall,
                                    fontFamily = FontFamily.Monospace,
                                )
                            }
                        }
                    }
                }
            }
        }
    }

    fullSize?.let { photo ->
        AlertDialog(
            onDismissRequest = { fullSize = null },
            confirmButton = {
                TextButton(onClick = { fullSize = null }) { Text("Close") }
            },
            title = { Text(STAMP.format(Date(photo.capturedAtEpochMs))) },
            text = {
                Box(Modifier.fillMaxWidth()) { FullImage(repository, photo) }
            },
        )
    }

    pendingDelete?.let { photo ->
        AlertDialog(
            onDismissRequest = { pendingDelete = null },
            title = { Text("Delete this photo?") },
            text = {
                Text(
                    "It will be removed from this phone permanently. " +
                        "This cannot be undone."
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    pendingDelete = null
                    scope.launch {
                        val gone = repository.delete(photo)
                        deleteError =
                            if (gone) null
                            else "The photo could not be fully deleted."
                    }
                }) { Text("Delete") }
            },
            dismissButton = {
                TextButton(onClick = { pendingDelete = null }) { Text("Cancel") }
            },
        )
    }
}

@Composable
private fun Thumbnail(
    repository: PhotoRepository,
    photo: PhotoEntity,
    modifier: Modifier,
) {
    var bitmap by remember(photo.id) { mutableStateOf<Bitmap?>(null) }
    LaunchedEffect(photo.id) { bitmap = decodeSampled(repository, photo, 160) }
    val bmp = bitmap
    if (bmp != null) {
        Image(
            bitmap = bmp.asImageBitmap(),
            contentDescription = null,
            modifier = modifier,
            contentScale = ContentScale.Crop,
        )
    } else {
        Box(modifier)
    }
}

@Composable
private fun FullImage(repository: PhotoRepository, photo: PhotoEntity) {
    var bitmap by remember(photo.id) { mutableStateOf<Bitmap?>(null) }
    LaunchedEffect(photo.id) { bitmap = decodeSampled(repository, photo, 1280) }
    val bmp = bitmap
    if (bmp != null) {
        Image(
            bitmap = bmp.asImageBitmap(),
            contentDescription = null,
            modifier = Modifier.fillMaxWidth(),
            contentScale = ContentScale.Fit,
        )
    } else {
        Box(Modifier.fillMaxWidth().height(200.dp))
    }
}

/**
 * Decode for DISPLAY at roughly [targetPx] on the long edge.
 *
 * Full-resolution camera frames are tens of megapixels; decoding one per list
 * row without subsampling is how a timeline runs out of memory.
 */
private suspend fun decodeSampled(
    repository: PhotoRepository,
    photo: PhotoEntity,
    targetPx: Int,
): Bitmap? = withContext(Dispatchers.IO) {
    val file = repository.fileFor(photo)
    if (!file.exists()) return@withContext null
    val longEdge = maxOf(photo.width, photo.height).takeIf { it > 0 } ?: targetPx
    var sample = 1
    while (longEdge / sample > targetPx * 2) sample *= 2
    val options = BitmapFactory.Options().apply { inSampleSize = sample }
    runCatching { BitmapFactory.decodeFile(file.absolutePath, options) }.getOrNull()
}
