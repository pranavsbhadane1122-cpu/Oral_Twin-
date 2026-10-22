package com.oraltwin.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
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
import com.oraltwin.app.camera.CameraScreen
import com.oraltwin.app.comparison.ComparisonResult
import com.oraltwin.app.comparison.ComparisonScreen
import com.oraltwin.app.comparison.PairComparator
import com.oraltwin.app.data.PhotoRepository
import com.oraltwin.app.timeline.TimelineScreen
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {

    private lateinit var repository: PhotoRepository

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        StackCheck.ensureOpenCv()   // before any Mat is constructed
        repository = PhotoRepository(applicationContext)

        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    OralTwinApp(repository)
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        repository.close()
    }
}

private enum class Screen(val label: String) {
    CAPTURE("Capture"),
    TIMELINE("Timeline"),
    CHECKS("Checks"),
}

@Composable
fun OralTwinApp(repository: PhotoRepository) {
    var screen by remember { mutableStateOf(Screen.CAPTURE) }
    var lastSaveMessage by remember { mutableStateOf<String?>(null) }
    var comparison by remember { mutableStateOf<ComparisonResult?>(null) }
    var comparing by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()

    // A comparison result takes over the whole screen. The UI does not decide
    // which one it is: ComparisonScreen dispatches on the sealed type.
    val currentComparison = comparison
    if (currentComparison != null) {
        ComparisonScreen(result = currentComparison, onBack = { comparison = null })
        return
    }

    Column(
        modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Column(Modifier.padding(start = 20.dp, end = 20.dp, top = 20.dp)) {
            Text("OralTwin", style = MaterialTheme.typography.headlineMedium)
            Text(
                "Take a photo at each visit and compare it with your last one. " +
                    "This app does not identify dental conditions.",
                style = MaterialTheme.typography.bodySmall,
            )
        }

        TabRow(selectedTabIndex = screen.ordinal) {
            Screen.entries.forEach { entry ->
                Tab(
                    selected = screen == entry,
                    onClick = { screen = entry },
                    text = { Text(entry.label) },
                )
            }
        }

        when (screen) {
            Screen.CAPTURE -> {
                CameraScreen(onCaptured = { bytes ->
                    scope.launch {
                        val saved = runCatching { repository.save(bytes) }
                        lastSaveMessage = saved.fold(
                            onSuccess = { "Saved ${it.filename} (${it.width} x ${it.height})" },
                            onFailure = { "Could not save the photo: ${it.javaClass.simpleName}" },
                        )
                        if (saved.isSuccess) screen = Screen.TIMELINE
                    }
                })
                lastSaveMessage?.let {
                    Text(it, Modifier.padding(horizontal = 20.dp),
                         style = MaterialTheme.typography.bodySmall)
                }
            }

            Screen.TIMELINE -> {
                TimelineScreen(
                    repository = repository,
                    onCompare = { pair ->
                        comparing = true
                        scope.launch {
                            comparison = runCatching {
                                PairComparator.compare(
                                    repository, pair.older, pair.newer)
                            }.getOrElse {
                                // a failure to run the comparison is itself a
                                // refusal, never a partial result
                                ComparisonResult.refuse(
                                    com.oraltwin.app.comparison
                                        .ComparabilityCode.ALIGNMENT_FAILED)
                            }
                            comparing = false
                        }
                    },
                )
                if (comparing) {
                    Text("Lining the two photos up...",
                         Modifier.padding(horizontal = 20.dp),
                         style = MaterialTheme.typography.bodySmall)
                }
            }

            Screen.CHECKS -> {
                StackCheckPanel()
                SelfCheckScreen()
            }
        }

        Spacer(Modifier.height(8.dp))
        Text(
            "This app does not detect disease. It contains no trained model of " +
                "any kind. Any concern should be checked by a dentist.",
            modifier = Modifier.padding(horizontal = 20.dp),
            style = MaterialTheme.typography.bodySmall,
        )
        Spacer(Modifier.height(16.dp))
    }
}

/** The M0 build check: proves the native libraries actually loaded. */
@Composable
fun StackCheckPanel() {
    val context = LocalContext.current
    var orb by remember { mutableStateOf<StackCheck.OrbResult?>(null) }
    val openCvVersion = remember { StackCheck.openCvVersion() }
    val device = remember { StackCheck.deviceDescription() }
    val abis = remember { StackCheck.supportedAbis() }
    val cameraX = remember { StackCheck.cameraXCheck() }
    val room = remember { StackCheck.roomCheck(context) }

    Column(
        Modifier.fillMaxWidth().padding(horizontal = 20.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Text("Build check", style = MaterialTheme.typography.titleLarge)
        Text(
            "Proves the libraries load on this phone. Not part of normal use.",
            style = MaterialTheme.typography.bodySmall,
        )

        InfoCard("Device") {
            Line(device)
            Line("ABIs: $abis")
        }
        InfoCard("OpenCV") {
            Line("version: $openCvVersion")
            Line("read from the library at runtime, not hardcoded")
        }
        InfoCard("CameraX") { Line(cameraX) }
        InfoCard("Room") { Line(room) }

        Button(onClick = { orb = StackCheck.orbKeypointCount() },
               modifier = Modifier.fillMaxWidth()) {
            Text("Run ORB on a test image")
        }

        orb?.let { result ->
            InfoCard(if (result.ok) "ORB: ran" else "ORB: FAILED") {
                Line("keypoints found: ${result.keypoints}")
                Line(result.detail)
                if (result.ok && result.keypoints > 0) {
                    Line("OpenCV native features2d is working.")
                }
            }
        }
    }
}

@Composable
private fun InfoCard(title: String, content: @Composable () -> Unit) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(title, fontWeight = FontWeight.Bold)
            content()
        }
    }
}

@Composable
private fun Line(text: String) {
    Text(text, style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace)
}
