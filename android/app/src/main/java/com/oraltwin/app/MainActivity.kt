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
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Load OpenCV before any Compose code can touch a Mat.
        StackCheck.ensureOpenCv()
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    StackCheckScreen()
                }
            }
        }
    }
}

@Composable
fun StackCheckScreen() {
    val context = LocalContext.current
    var orb by remember { mutableStateOf<StackCheck.OrbResult?>(null) }
    val openCvVersion = remember { StackCheck.openCvVersion() }
    val device = remember { StackCheck.deviceDescription() }
    val abis = remember { StackCheck.supportedAbis() }
    val cameraX = remember { StackCheck.cameraXCheck() }
    val room = remember { StackCheck.roomCheck(context) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("OralTwin", style = MaterialTheme.typography.headlineMedium)
        Text(
            "M0 - build check. This screen exists to prove the libraries load on " +
                "a real phone. It does not take or read photographs.",
            style = MaterialTheme.typography.bodyMedium,
        )

        Spacer(Modifier.height(4.dp))

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
                    Line("OpenCV's native features2d module is working.")
                } else if (result.ok) {
                    Line("ORB ran but found nothing - unexpected for this test image.")
                }
            }
        }

        Spacer(Modifier.height(8.dp))
        Text(
            "This app does not detect disease. It contains no trained model of " +
                "any kind. Any concern should be checked by a dentist.",
            style = MaterialTheme.typography.bodySmall,
        )
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
