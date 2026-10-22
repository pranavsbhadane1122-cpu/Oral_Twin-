package com.oraltwin.app.comparison

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

/**
 * The two comparison screens.
 *
 * [ComparisonScreen] takes the sealed result and dispatches. That is the only
 * place the choice is made, and it is a `when` over a sealed type, so a new
 * result variant is a compile error rather than a screen that silently falls
 * through to the wrong branch.
 *
 * There is NO path from a Refused to the overlay: OverlayScreen takes a
 * ComparisonResult.Comparable, not a nullable bitmap and a flag, so it cannot
 * be constructed from a refusal at all.
 */
@Composable
fun ComparisonScreen(
    result: ComparisonResult,
    onBack: () -> Unit,
    modifier: Modifier = Modifier,
    onExport: (() -> Unit)? = null,
) {
    when (result) {
        is ComparisonResult.Comparable -> OverlayScreen(result, onBack, modifier, onExport)
        // A refusal is exportable too, and exports AS a refusal. Making it
        // unexportable would teach the user that the export button means the
        // app found something.
        is ComparisonResult.Refused -> RefusalScreen(result, onBack, modifier, onExport)
    }
}

@Composable
private fun OverlayScreen(
    result: ComparisonResult.Comparable,
    onBack: () -> Unit,
    modifier: Modifier = Modifier,
    onExport: (() -> Unit)? = null,
) {
    var blend by remember { mutableFloatStateOf(0.5f) }
    var sideBySide by remember { mutableStateOf(false) }

    Column(
        modifier = modifier.fillMaxWidth().padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("Your two photos, lined up", style = MaterialTheme.typography.titleLarge)

        Row(verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            Text("Side by side", style = MaterialTheme.typography.bodyMedium)
            Switch(checked = sideBySide, onCheckedChange = { sideBySide = it })
        }

        if (sideBySide) {
            Row(Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Column(Modifier.weight(1f),
                       horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("Older (lined up)", style = MaterialTheme.typography.bodySmall)
                    Image(result.alignedOlder.asImageBitmap(), null,
                          Modifier.fillMaxWidth(), contentScale = ContentScale.Fit)
                }
                Column(Modifier.weight(1f),
                       horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("Newer", style = MaterialTheme.typography.bodySmall)
                    Image(result.newer.asImageBitmap(), null,
                          Modifier.fillMaxWidth(), contentScale = ContentScale.Fit)
                }
            }
        } else {
            BlendedImages(result, blend)
            Text(
                "Slide to fade between them: left is the newer photo only, " +
                    "right is the older one only.",
                style = MaterialTheme.typography.bodySmall,
            )
            Slider(value = blend, onValueChange = { blend = it })
        }

        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(14.dp),
                   verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text("How well they lined up", fontWeight = FontWeight.Bold)
                Mono("comparability: ${ComparabilityCode.COMPARABLE}")
                Mono("confidence:    %.2f".format(result.confidence))
                Text(
                    "This describes how well the two photographs line up. It " +
                        "says nothing about what is in them.",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }

        Text(RefusalGuidance.DISCLAIMER, fontWeight = FontWeight.Medium)
        onExport?.let {
            OutlinedButton(onClick = it, modifier = Modifier.fillMaxWidth()) {
                Text("Share these photos")
            }
        }
        Button(onClick = onBack, modifier = Modifier.fillMaxWidth()) { Text("Back") }
    }
}

/**
 * Cross-fade: 0 shows the newer photograph, 1 the aligned older one.
 *
 * Two stacked Images with an alpha on the top one. Both are already in the
 * newer photograph's frame - the older was warped there during alignment - so
 * they register pixel for pixel and need no further transform here.
 */
@Composable
private fun BlendedImages(result: ComparisonResult.Comparable, blend: Float) {
    Box(Modifier.fillMaxWidth()) {
        Image(
            bitmap = result.newer.asImageBitmap(),
            contentDescription = null,
            modifier = Modifier.fillMaxWidth(),
            contentScale = ContentScale.Fit,
        )
        Image(
            bitmap = result.alignedOlder.asImageBitmap(),
            contentDescription = null,
            modifier = Modifier.fillMaxWidth(),
            contentScale = ContentScale.Fit,
            alpha = blend,
        )
    }
}

@Composable
private fun RefusalScreen(
    result: ComparisonResult.Refused,
    onBack: () -> Unit,
    modifier: Modifier = Modifier,
    onExport: (() -> Unit)? = null,
) {
    // NO colour band here, of any shade. The Python demo shipped a green
    // "Low - nothing stood out as changed" on a refused comparison, and a
    // test asserted it. A neutral card with words is the whole design.
    Column(
        modifier = modifier.fillMaxWidth().padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("These photos could not be compared",
             style = MaterialTheme.typography.titleLarge)

        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(14.dp),
                   verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("What happened", fontWeight = FontWeight.Bold)
                Text(result.guidance)
            }
        }

        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(14.dp),
                   verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text("Why you are not seeing a result", fontWeight = FontWeight.Bold)
                Text(
                    "No comparison was made, so there is nothing to report. " +
                        "The app does not show a partial or approximate answer " +
                        "when it could not do the work.",
                    style = MaterialTheme.typography.bodySmall,
                )
                Mono("comparability: ${result.code}")
            }
        }

        Text(RefusalGuidance.DISCLAIMER, fontWeight = FontWeight.Medium)
        onExport?.let {
            OutlinedButton(onClick = it, modifier = Modifier.fillMaxWidth()) {
                Text("Share these photos")
            }
        }
        Button(onClick = onBack, modifier = Modifier.fillMaxWidth()) {
            Text("Back to my photos")
        }
    }
}

@Composable
private fun Mono(text: String) {
    Text(text, style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace)
}
