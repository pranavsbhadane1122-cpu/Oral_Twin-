// Top-level build file. Plugin versions are declared here and applied in :app.
plugins {
    id("com.android.application") version "8.7.3" apply false
    id("org.jetbrains.kotlin.android") version "2.0.21" apply false
    // Kotlin 2.0 moved the Compose compiler into a Kotlin plugin; without this
    // the build fails with "Compose Compiler requires Kotlin ..."
    id("org.jetbrains.kotlin.plugin.compose") version "2.0.21" apply false
    // Room's annotation processor. KSP versions are pinned to a Kotlin version:
    // the prefix here (2.0.21) must match the Kotlin version above exactly.
    id("com.google.devtools.ksp") version "2.0.21-1.0.28" apply false
}
