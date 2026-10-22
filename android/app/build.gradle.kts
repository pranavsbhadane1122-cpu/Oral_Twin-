plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
    id("com.google.devtools.ksp")
}

android {
    namespace = "com.oraltwin.app"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.oraltwin.app"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0-M0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // OpenCV ships native .so files for these ABIs. Real phones are all
        // arm64-v8a; x86_64 is kept so the app also runs on an emulator.
        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86_64")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"),
                          "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        compose = true
    }
    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

dependencies {
    // --- Compose -----------------------------------------------------------
    val composeBom = platform("androidx.compose:compose-bom:2024.10.01")
    implementation(composeBom)
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    // declared explicitly rather than relied on transitively: the
    // self-check runs ORB off the main thread via Dispatchers.Default
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    debugImplementation("androidx.compose.ui:ui-tooling")

    // --- OpenCV ------------------------------------------------------------
    // The official Android AAR, published to Maven Central since 4.9.0. It
    // carries features2d (ORB), calib3d (findHomography/RANSAC) and imgproc
    // (warpPerspective) - the four things the alignment port will need.
    // No manual SDK download or module import. See README-android.md.
    implementation("org.opencv:opencv:4.11.0")

    // --- CameraX -----------------------------------------------------------
    val cameraX = "1.4.0"
    implementation("androidx.camera:camera-core:$cameraX")
    implementation("androidx.camera:camera-camera2:$cameraX")
    implementation("androidx.camera:camera-lifecycle:$cameraX")
    implementation("androidx.camera:camera-view:$cameraX")

    // --- Room --------------------------------------------------------------
    val room = "2.6.1"
    implementation("androidx.room:room-runtime:$room")
    implementation("androidx.room:room-ktx:$room")
    ksp("androidx.room:room-compiler:$room")

    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.6.1")
}
