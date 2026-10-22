package com.oraltwin.app.export

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.Typeface
import android.net.Uri
import androidx.core.content.FileProvider
import com.oraltwin.app.comparison.ComparisonResult
import com.oraltwin.app.data.PhotoEntity
import com.oraltwin.app.data.PhotoRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Renders a comparison to a shareable image, and hands it to the share sheet.
 *
 * NO NETWORK, AND NONE POSSIBLE. Sharing is FileProvider plus ACTION_SEND:
 * the app writes a file, grants a one-off read permission on a content:// URI,
 * and lets the user choose what receives it. Nothing is transmitted by this
 * app. The manifest still declares no INTERNET, and if an export design ever
 * needed it, that design would be wrong rather than the manifest.
 *
 * IT RENDERS THE RESULT IT IS GIVEN. The sealed ComparisonResult comes
 * straight from the comparison the UI is already showing; this code never
 * re-derives comparability. Two code paths deciding the same question is how
 * a pair ends up with two different answers, one on screen and one in a file
 * someone keeps.
 *
 * A REFUSAL IS EXPORTABLE. It exports as a refusal - the two photographs,
 * their dates, the line saying they could not be compared, and the disclaimer.
 * No overlay, because there is no trustworthy one. Refusing to export a
 * refusal would teach the user that the export button means the app found
 * something.
 */
object ComparisonExport {

    private const val WIDTH = 1200
    private const val MARGIN = 40
    private const val GAP = 28
    private const val EXPORT_DIR = "exports"
    const val AUTHORITY_SUFFIX = ".fileprovider"

    private val dateFormat get() =
        SimpleDateFormat(ExportStrings.DATE_PATTERN, Locale.getDefault())

    /** Which template set a result uses. Dispatch happens once, here. */
    fun templatesFor(result: ComparisonResult): List<String> = when (result) {
        is ComparisonResult.Comparable -> ExportStrings.comparableVariant
        is ComparisonResult.Refused -> ExportStrings.refusedVariant
    }

    /**
     * Compose the export image.
     *
     * Photographs are decoded for DISPLAY here - this is a picture for a human
     * to look at, nothing is measured from it - so BitmapFactory is correct.
     * PORT-NOTES section 2.
     */
    fun render(
        repository: PhotoRepository,
        older: PhotoEntity,
        newer: PhotoEntity,
        result: ComparisonResult,
    ): Bitmap {
        val olderBitmap = decodeForDisplay(repository, older)
        val newerBitmap = decodeForDisplay(repository, newer)
        val overlay = (result as? ComparisonResult.Comparable)?.alignedOlder

        val photoWidth = (WIDTH - 2 * MARGIN - GAP) / 2
        val photoHeight = scaledHeight(olderBitmap, newerBitmap, photoWidth)
        val overlayHeight =
            if (overlay != null) scaledHeight(overlay, overlay, WIDTH - 2 * MARGIN)
            else 0

        val title = textPaint(34f, bold = true)
        val label = textPaint(24f)
        val body = textPaint(26f)
        val small = textPaint(22f)

        val comparedLines = when (result) {
            is ComparisonResult.Comparable -> listOf(ExportStrings.COMPARED_LINE)
            is ComparisonResult.Refused ->
                listOf(ExportStrings.REFUSED_LINE, ExportStrings.REFUSED_EXPLANATION)
        }

        val textWidth = WIDTH - 2 * MARGIN
        val comparedWrapped = comparedLines.flatMap { wrap(it, body, textWidth) }
        val disclaimerWrapped = wrap(ExportStrings.DISCLAIMER, small, textWidth)
        val projectWrapped = wrap(ExportStrings.PROJECT_NOTE, small, textWidth)

        var height = MARGIN
        height += lineHeight(title) + GAP
        height += lineHeight(label) + 8 + photoHeight + GAP
        if (overlay != null) height += lineHeight(label) + 8 + overlayHeight + GAP
        height += comparedWrapped.size * lineHeight(body) + GAP
        height += disclaimerWrapped.size * lineHeight(small) + 12
        height += projectWrapped.size * lineHeight(small) + MARGIN

        val canvasBitmap = Bitmap.createBitmap(WIDTH, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(canvasBitmap)
        // plain white. No colour coding of any kind, anywhere in this file:
        // a tint is a verdict, and there is no verdict to give.
        canvas.drawColor(Color.WHITE)

        var y = MARGIN + lineHeight(title)
        canvas.drawText(ExportStrings.TITLE, MARGIN.toFloat(), y.toFloat(), title)
        y += GAP

        // the two photographs, side by side, each under its own date
        y += lineHeight(label)
        canvas.drawText(
            "${ExportStrings.LABEL_OLDER} - ${dateFormat.format(Date(older.capturedAtEpochMs))}",
            MARGIN.toFloat(), y.toFloat(), label)
        canvas.drawText(
            "${ExportStrings.LABEL_NEWER} - ${dateFormat.format(Date(newer.capturedAtEpochMs))}",
            (MARGIN + photoWidth + GAP).toFloat(), y.toFloat(), label)
        y += 8
        olderBitmap?.let {
            canvas.drawBitmap(it, null,
                Rect(MARGIN, y, MARGIN + photoWidth, y + photoHeight), null)
        }
        newerBitmap?.let {
            canvas.drawBitmap(it, null,
                Rect(MARGIN + photoWidth + GAP, y,
                     MARGIN + 2 * photoWidth + GAP, y + photoHeight), null)
        }
        y += photoHeight + GAP

        if (overlay != null) {
            y += lineHeight(label)
            canvas.drawText(ExportStrings.LABEL_OVERLAY,
                            MARGIN.toFloat(), y.toFloat(), label)
            y += 8
            canvas.drawBitmap(overlay, null,
                Rect(MARGIN, y, WIDTH - MARGIN, y + overlayHeight), null)
            y += overlayHeight + GAP
        }

        for (line in comparedWrapped) {
            y += lineHeight(body)
            canvas.drawText(line, MARGIN.toFloat(), y.toFloat(), body)
        }
        y += GAP

        // the disclaimer lives INSIDE the image, where it cannot be separated
        // from the picture by a messaging app that drops the share text
        for (line in disclaimerWrapped) {
            y += lineHeight(small)
            canvas.drawText(line, MARGIN.toFloat(), y.toFloat(), small)
        }
        y += 12
        for (line in projectWrapped) {
            y += lineHeight(small)
            canvas.drawText(line, MARGIN.toFloat(), y.toFloat(), small)
        }

        olderBitmap?.recycle()
        newerBitmap?.recycle()
        return canvasBitmap
    }

    /** Write the rendered image and return a content:// URI for sharing. */
    suspend fun writeForSharing(
        context: Context,
        bitmap: Bitmap,
    ): Uri = withContext(Dispatchers.IO) {
        val dir = File(context.filesDir, EXPORT_DIR).apply { mkdirs() }
        // one slot, overwritten: exports are transient, and leaving a growing
        // pile of comparisons in app storage is its own privacy problem
        val file = File(dir, "comparison.png")
        file.outputStream().use { out ->
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
        }
        FileProvider.getUriForFile(
            context, context.packageName + AUTHORITY_SUFFIX, file)
    }

    /** ACTION_SEND with a one-off read grant. No network involved. */
    fun shareIntent(uri: Uri): Intent =
        Intent(Intent.ACTION_SEND).apply {
            type = "image/png"
            putExtra(Intent.EXTRA_STREAM, uri)
            putExtra(Intent.EXTRA_SUBJECT, ExportStrings.SHARE_SUBJECT)
            putExtra(Intent.EXTRA_TEXT, ExportStrings.SHARE_TEXT)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }

    // ------------------------------------------------------------- helpers

    private fun decodeForDisplay(
        repository: PhotoRepository,
        photo: PhotoEntity,
    ): Bitmap? {
        val file = repository.fileFor(photo)
        if (!file.exists()) return null
        val options = android.graphics.BitmapFactory.Options().apply {
            inSampleSize = sampleFor(photo.width, photo.height)
        }
        return runCatching {
            android.graphics.BitmapFactory.decodeFile(file.absolutePath, options)
        }.getOrNull()
    }

    private fun sampleFor(width: Int, height: Int): Int {
        var sample = 1
        val longEdge = maxOf(width, height)
        while (longEdge / sample > WIDTH) sample *= 2
        return sample
    }

    private fun scaledHeight(a: Bitmap?, b: Bitmap?, targetWidth: Int): Int {
        val source = a ?: b ?: return targetWidth
        if (source.width == 0) return targetWidth
        return (targetWidth.toFloat() * source.height / source.width).toInt()
    }

    private fun textPaint(size: Float, bold: Boolean = false) = Paint().apply {
        color = Color.BLACK          // never a status colour
        textSize = size
        isAntiAlias = true
        typeface = if (bold) Typeface.DEFAULT_BOLD else Typeface.DEFAULT
    }

    private fun lineHeight(paint: Paint): Int =
        (paint.fontMetrics.descent - paint.fontMetrics.ascent).toInt() + 6

    /** Greedy word wrap, so long sentences are not clipped at the margin. */
    internal fun wrap(text: String, paint: Paint, maxWidth: Int): List<String> {
        val words = text.split(" ")
        val lines = ArrayList<String>()
        var line = StringBuilder()
        for (word in words) {
            val candidate = if (line.isEmpty()) word else "$line $word"
            if (paint.measureText(candidate) <= maxWidth || line.isEmpty()) {
                line = StringBuilder(candidate)
            } else {
                lines.add(line.toString())
                line = StringBuilder(word)
            }
        }
        if (line.isNotEmpty()) lines.add(line.toString())
        return lines
    }
}
