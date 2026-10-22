# OralTwin Android — M0

A native Kotlin app that lives inside the OralTwin repository at `android/`,
so it can later port the alignment algorithms from `../src/alignment/` without
a second repository to keep in step.

**M0 is a build check and nothing more.** One screen that proves the libraries
actually work on a real phone. It does not open the camera, store a photograph
or compare anything.

---

## What this app does not do, and will not do

- **No disease detection.** It does not identify caries, gum disease, oral
  cancer or anything else, and it never will. The research project's
  classifier reached 0.491 sensitivity on its own held-out data and 0.072 on
  photographs from another clinic, and it failed its attention gate. It is
  parked and it is not in this app.
- **No lesion outlining.** The segmentation model failed its acceptance gate:
  it learned to draw around the inside of the mouth rather than find the area
  of concern. It is not in this app either.
- **No models trained on licence-restricted data.** The Piyarathne dataset is
  CC BY-NC-ND. No weights derived from it are bundled here, and none will be.
  The APK ships no trained model of any kind.
- **No network permission.** The manifest declares no `INTERNET` permission.
  Photographs, once M1 adds them, stay on the phone.

What it *will* do, across later milestones: guided capture, a photo timeline,
alignment between visits, and refusing clearly when a comparison cannot be
trusted. Those are the parts of the research project that passed their gates.

---

## Installing Android Studio

As of 22 Oct 2026 this machine has no Android toolchain: no Android Studio, no
JDK, no SDK, no `adb`. Checked against the registry, the Start Menu, `winget`
and a full scan of `C:\` for `studio64.exe` and `adb.exe`.

Two things that look like an install and are not:

- **`D:\Android`** is a connected TCL Android TV's storage
  (`D:\Android\data\com.tcl.*`), not an installation.
- **`RStudio-...exe`** in Downloads is RStudio, the R statistics IDE. Despite
  the name it is an unrelated product and provides no JDK or Android SDK.

To install:

1. <https://developer.android.com/studio> and download for Windows (~1.1 GB)
2. Run the installer, accepting the defaults
3. **Open it once and let the first-run wizard finish.** This is the step that
   downloads the SDK, the platform and the build-tools, another 2-3 GB. Until
   it has run there is no SDK for Gradle to find.
4. Then either press Run in Studio, or from Git Bash:

   ```
   cd android
   ./configure-and-build.sh
   ```

   That script finds the bundled JDK and the SDK, writes `local.properties`,
   obtains a Gradle if the wrapper jar is missing, and builds the debug APK.
   It prints what it found before using it, so a wrong guess is visible
   immediately rather than surfacing as a confusing Gradle error.

## Where photographs are stored, and why it matters

Every JPEG is written under `Context.filesDir/photos` - app-internal storage.
Not MediaStore, not `getExternalFilesDir`, not the shared Pictures collection.
These are intraoral photographs: they must not appear in the camera roll, must
not be picked up by a gallery scanner, and must not be swept into a cloud photo
backup.

Three mechanisms, none of which rely on remembering to do the right thing:

1. **No `INTERNET` permission.** The manifest does not request it, and
   declares `tools:node="remove"` so a dependency cannot merge one in
   silently. Without it the process cannot open a socket, so no code path -
   ours, a library's, or one added later by mistake - can upload anything.
   `PermissionsTest` asserts this against the installed package, so a
   regression fails a test instead of shipping.
2. **`allowBackup="false"` plus explicit data-extraction rules.** Android's
   auto-backup would otherwise copy `filesDir` to the user's Google Drive.
   `res/xml/data_extraction_rules.xml` excludes files, databases and
   preferences from both cloud backup and device-to-device transfer.
3. **One writer.** CameraX captures into memory and hands the bytes to
   `PhotoRepository`, which is the only thing that writes a photo file.
   `ImageCapture.OutputFileOptions` can target MediaStore directly, so not
   using it is what keeps these images out of the gallery by construction.

Deleting a photo removes the row *and* the file, and `PhotoRepository.delete`
returns true only after confirming the file is genuinely gone from disk.

## Requirements

| | |
|---|---|
| Android Studio | Ladybug (2024.2) or newer |
| JDK | 17 — bundled with Android Studio, you do not install it separately |
| Gradle | 8.9 — downloaded automatically on first sync |
| Android Gradle Plugin | 8.7.3 |
| Kotlin | 2.0.21 |
| compileSdk / targetSdk | 35 (Android 15) |
| minSdk | 26 (Android 8.0) |

---

## How OpenCV was added — the fiddly step, written out

Most OpenCV-on-Android guides tell you to download a ~1 GB SDK zip, unpack it,
`File > New > Import Module`, point at the `sdk` folder, fix its
`build.gradle`, and add `implementation(project(":opencv"))`. **That is not
what this project does**, and you do not need to do it.

### What this project actually does

Since version 4.9.0, OpenCV publishes an official Android AAR to Maven
Central. So the whole of the setup is two things:

**1.** `mavenCentral()` in the repositories block — `android/settings.gradle.kts`:

```kotlin
dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}
```

**2.** One dependency line — `android/app/build.gradle.kts`:

```kotlin
implementation("org.opencv:opencv:4.11.0")
```

That is it. No download, no module import, no `local.properties` edit.

### The two things that still catch people out

**You must initialise it before touching a `Mat`.** The Java classes load
happily without the native `.so` files behind them; the failure only appears
the first time you construct a `Mat`, as an `UnsatisfiedLinkError`. So call
this once, early — `MainActivity.onCreate` does it:

```kotlin
OpenCVLoader.initLocal()
```

If you find a tutorial mentioning `initAsync()` and an "OpenCV Manager" app
from the Play Store, it is out of date by years. `initLocal()` loads the
library bundled inside the AAR. There is nothing to install alongside.

**Keep the ABI filters.** `android/app/build.gradle.kts` has:

```kotlin
ndk { abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86_64") }
```

Every real phone is `arm64-v8a`. `x86_64` is kept only so the app also runs on
an emulator. Removing these makes the APK considerably larger; removing the
wrong one makes it crash on launch with `UnsatisfiedLinkError` and no useful
message.

### If you ever do need the manual SDK import

Only necessary for a custom OpenCV build (extra contrib modules, a smaller
binary). For the record:

1. Download `opencv-4.x.y-android-sdk.zip` from <https://opencv.org/releases/>
2. Unzip it somewhere outside this repository
3. `File > New > Import Module`, select the `sdk` folder, name it `opencv`
4. In `opencv/build.gradle`, match `compileSdk`, `minSdk` and the
   `namespace` to this project's values
5. `include(":opencv")` in `settings.gradle.kts`
6. Replace the Maven line with `implementation(project(":opencv"))`

`android/opencv/` is already in the repository's `.gitignore`, so an imported
SDK will not be committed by accident.

---

## Opening and building the project

1. Open **Android Studio**
2. **Open**, and select the `android` folder inside the OralTwin project —
   that exact folder, *not* the repository root. The repository root is a
   Python project; Android Studio will not understand it.
3. Wait for **Gradle sync**. The first one downloads Gradle 8.9 and all the
   dependencies and can take several minutes on a slow connection.
4. If it offers to install a missing SDK platform or build-tools, accept.

### A note on `gradle-wrapper.jar`

This repository contains `gradle/wrapper/gradle-wrapper.properties` but **not**
`gradle-wrapper.jar`, because that file is a binary and was deliberately not
fabricated by hand. Android Studio regenerates it on first sync and you will
not notice.

If you want to build from a terminal instead, generate the wrapper once:

```
cd android
gradle wrapper --gradle-version 8.9
```

(that needs Gradle installed), after which `./gradlew assembleDebug` works
normally.

---

## Getting it onto your phone

Two routes. **Route A is much easier and is the one to use.**

### Route A — straight from Android Studio over USB

**Step 1 — turn on Developer options on the phone**

1. **Settings > About phone**
2. Find **Build number**. On Samsung it is under *Software information*; on
   Pixel it is at the bottom of *About phone*.
3. **Tap Build number seven times.** It counts down at you — "you are 3 steps
   away from being a developer" — then asks for your PIN.
4. You will see "You are now a developer".

**Step 2 — turn on USB debugging**

1. **Settings > System > Developer options** (on Samsung: Settings >
   Developer options, near the bottom)
2. Switch on **USB debugging**

**Step 3 — plug the phone in**

1. Connect it with a USB cable. **It must be a data cable** — many charging
   cables carry power only, and a power-only cable looks exactly like a
   working one. If the phone charges but nothing else happens, suspect the
   cable first.
2. The phone shows **"Allow USB debugging?"** with a computer fingerprint.
   Tick *Always allow from this computer* and press **Allow**. If this never
   appears, unplug and replug.
3. Some phones also need the USB mode changed from *Charging* to
   *File transfer (MTP)* — swipe down the notification shade and tap the USB
   notification.

**Step 4 — run it**

1. In Android Studio, your phone's name appears in the device dropdown in the
   toolbar, beside the green play button
2. Press **Run** (the green ▶, or Shift+F10)
3. First run takes a minute or two. The app launches by itself.

### Route B — build an APK and install it by hand

Use this if USB debugging will not cooperate.

1. **Build > Build Bundle(s)/APK(s) > Build APK(s)**
2. When it finishes, click **locate** in the notification. The file is
   `android/app/build/outputs/apk/debug/app-debug.apk`
3. Copy it to the phone (USB, Google Drive, or email it to yourself)
4. On the phone, open the file. Android will say installing from this source
   is blocked — tap **Settings**, allow it for whichever app you used, then
   go back and tap install again.

APKs are git-ignored, so a built APK will not be committed.

---

## What you should see when it runs

A scrolling screen headed **OralTwin**, then:

- **Device** — your phone's manufacturer, model, Android version and API
  level, plus the supported ABIs (expect `arm64-v8a` first)
- **OpenCV** — `version: 4.11.0`, read out of the library at runtime rather
  than printed from a constant in our code
- **CameraX** — `OK` with two class names
- **Room** — `OK - wrote and read back 1 row`
- A button, **Run ORB on a test image**

Press the button. It builds a small greyscale image with a grid of bright
rectangles, runs ORB over it and reports the keypoint count. **A non-zero
count is the thing that matters**: it means OpenCV's native `features2d`
module genuinely ran on the phone, which is what is actually in doubt. A
dependency that merely resolved would get this far and then fail here.

Expect a few hundred keypoints. The exact number does not matter.

### If something is wrong

| What you see | What it means |
|---|---|
| OpenCV version `not loaded` | `initLocal()` returned false — the native library is missing for your phone's ABI. Check `abiFilters`. |
| ORB card says **FAILED** with `UnsatisfiedLinkError` | Same cause: the Java classes loaded, the native ones did not. |
| `keypoints found: 0` but ORB ran | Unexpected. The detector ran but found nothing in an image built to be full of corners — worth telling me about. |
| Room `FAILED` | KSP did not generate the database implementation. Usually a Kotlin/KSP version mismatch — the two version numbers in `android/build.gradle.kts` must share the same Kotlin prefix. |

---

## Status

**M0 has not been built or run by me**, because there is no toolchain on this
machine to build it with (see *Installing Android Studio* above). Gradle has
never executed here and no APK exists. The code is written to be correct but
it is **unverified**, and a first Android sync commonly wants a version nudge.

The versions pinned here - AGP 8.7.3, Kotlin 2.0.21, compileSdk 35 - are
deliberately conservative, chosen because they could not be tested. If Gradle
asks for newer ones, that is expected and they should be taken.

Tell me what the screen says — or paste the build error — and I will fix it
before M1.
