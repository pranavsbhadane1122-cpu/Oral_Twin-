package com.oraltwin.app

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.oraltwin.app.alignment.SelfCheck
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Runs the ported alignment on the bundled pairs and shows each result beside
 * the Python figure for the same pair.
 *
 * The comparison is printed rather than asserted. A gate that the app decides
 * for itself is a gate that can be quietly widened; this one is read off the
 * screen by a person.
 */
@Composable
fun SelfCheckScreen() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var running by remember { mutableStateOf(false) }
    var outcomes by remember { mutableStateOf<List<SelfCheck.Outcome>?>(null) }
    var error by remember { mutableStateOf<String?>(null) }
    val assetsPresent = remember { SelfCheck.manifestPresent(context) }

    Column(
        modifier = Modifier.fillMaxWidth().padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("Alignment self-check", style = MaterialTheme.typography.headlineSmall)
        Text(
            "Runs the Kotlin port of the Python alignment module on bundled test " +
                "pairs and prints the Python figure beside each result. JVM unit " +
                "tests cannot exercise OpenCV native code, so the comparison is " +
                "done here, on the device.",
            style = MaterialTheme.typography.bodySmall,
        )

        if (!assetsPresent) {
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(14.dp)) {
                    Text("No test pairs bundled", fontWeight = FontWeight.Bold)
                    Mono(
                        "The images are Kaggle-derived and are git-ignored, because " +
                            "the project has not confirmed that dataset allows " +
                            "redistribution. Generate them locally with:"
                    )
                    Mono("python scripts/bundle_android_testpairs.py")
                    Mono("then rebuild the APK.")
                }
            }
            return@Column
        }

        Button(
            onClick = {
                running = true
                error = null
                scope.launch {
                    try {
                        // off the main thread: five ORB runs at 4000 features
                        val result = withContext(Dispatchers.Default) {
                            SelfCheck.runAll(context)
                        }
                        outcomes = result
                    } catch (t: Throwable) {
                        error = "${t.javaClass.simpleName}: ${t.message}"
                    } finally {
                        running = false
                    }
                }
            },
            enabled = !running,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text(if (running) "Running..." else "Run alignment on the 5 test pairs")
        }

        if (running) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Center) {
                CircularProgressIndicator()
            }
        }

        error?.let {
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(14.dp)) {
                    Text("Self-check failed", fontWeight = FontWeight.Bold)
                    Mono(it)
                }
            }
        }

        outcomes?.let { results ->
            val kotlinMedian = SelfCheck.medianCornerError(results)
            val pythonMedian = SelfCheck.medianPythonCornerError(results)

            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(14.dp),
                       verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text("Summary", fontWeight = FontWeight.Bold)
                    Mono("pairs aligned:   ${results.count { it.ok }} of ${results.size}")
                    Mono("Kotlin median:   " +
                        (kotlinMedian?.let { "%.3f px".format(it) } ?: "n/a"))
                    Mono("Python median:   " +
                        (pythonMedian?.let { "%.3f px".format(it) } ?: "n/a"))
                    if (kotlinMedian != null && pythonMedian != null && pythonMedian > 0) {
                        val ratio = kotlinMedian / pythonMedian
                        Mono("ratio:           %.2fx  (gate: within 2.00x)".format(ratio))
                        Mono(if (ratio <= 2.0) "GATE: within 2x" else "GATE: OUTSIDE 2x")
                    }
                    Mono("Python median over all 200 pairs: 1.183 px")
                }
            }

            results.forEach { outcome ->
                Card(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(14.dp),
                           verticalArrangement = Arrangement.spacedBy(2.dp)) {
                        Text("${outcome.pair}  (${outcome.sourceClass})",
                             fontWeight = FontWeight.Bold)
                        Mono("image:        ${outcome.width}x${outcome.height}")
                        if (!outcome.ok) {
                            Mono("REFUSED:      ${outcome.reason ?: "no reason given"}")
                            Mono("python was:   %.3f px, confidence %.3f, %d inliers"
                                .format(outcome.python.cornerErrorPx,
                                        outcome.python.confidence,
                                        outcome.python.nInliers))
                        } else {
                            Mono("                 kotlin     python")
                            Mono("inliers:      %8d   %8d".format(
                                outcome.nInliers, outcome.python.nInliers))
                            Mono("confidence:   %8.3f   %8.3f".format(
                                outcome.confidence, outcome.python.confidence))
                            Mono("corner error: %8.3f   %8.3f  px".format(
                                outcome.cornerErrorPx ?: Double.NaN,
                                outcome.python.cornerErrorPx))
                            outcome.ratio?.let { Mono("ratio:        %8.2fx".format(it)) }
                        }
                    }
                }
            }

            Spacer(Modifier.height(4.dp))
            Text(
                "RANSAC is randomised, so these numbers will not match Python " +
                    "exactly and will vary slightly between runs. A large " +
                    "divergence means the port is wrong; a small one is expected.",
                style = MaterialTheme.typography.bodySmall,
            )
        }
    }
}

@Composable
private fun Mono(text: String) {
    Text(text, style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace)
}
