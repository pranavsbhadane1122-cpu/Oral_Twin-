package com.oraltwin.app.about

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

/**
 * One screen, plain language, no feature list.
 *
 * A feature list invites the reader to infer capability from length. What
 * matters here is the opposite: what this does not do, and why someone should
 * not act on it.
 */
object AboutText {

    const val WHAT_IT_DOES =
        "OralTwin keeps photographs of your mouth and lines up two of them so " +
            "you can see them together. That is all it does."

    const val WHAT_IT_DOES_NOT =
        "It does not identify dental conditions. It cannot tell you whether " +
            "anything is wrong, whether something has got better or worse, or " +
            "whether you need treatment. It contains no trained model of any " +
            "kind, and it gives no result to act on."

    const val WHY_NOT =
        "Earlier versions of this project did try to recognise conditions from " +
            "photographs. Those attempts were tested and they did not work well " +
            "enough to be trusted, so they were left out rather than shipped " +
            "with a caveat."

    const val PROJECT =
        "This is a student research project. It has not been reviewed or " +
            "approved by any medical or dental body, and it has not been tested " +
            "in a clinic."

    const val PRIVACY =
        "Your photographs stay in this app on this phone. The app has no " +
            "permission to use the internet, so it cannot send them anywhere. " +
            "If you share one yourself, it leaves the phone at that point and " +
            "the app no longer controls where it goes."

    const val DENTIST = "Have this checked by a dentist."

    val all: List<String> = listOf(
        WHAT_IT_DOES, WHAT_IT_DOES_NOT, WHY_NOT, PROJECT, PRIVACY, DENTIST,
    )
}

@Composable
fun AboutScreen(modifier: Modifier = Modifier) {
    Column(
        modifier = modifier.fillMaxWidth().padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("About OralTwin", style = MaterialTheme.typography.titleLarge)

        Section("What it does", AboutText.WHAT_IT_DOES)
        Section("What it does not do", AboutText.WHAT_IT_DOES_NOT)
        Section("Why not", AboutText.WHY_NOT)
        Section("Where your photos go", AboutText.PRIVACY)
        Section("What this is", AboutText.PROJECT)

        Text(AboutText.DENTIST, fontWeight = FontWeight.Medium)
    }
}

@Composable
private fun Section(heading: String, body: String) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(14.dp),
               verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(heading, fontWeight = FontWeight.Bold)
            Text(body, style = MaterialTheme.typography.bodyMedium)
        }
    }
}
