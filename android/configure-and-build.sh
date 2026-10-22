#!/usr/bin/env bash
# Locate Android Studio's bundled JDK and the Android SDK, write
# local.properties, and build the debug APK.
#
# Run from the android/ folder:   ./configure-and-build.sh
#
# Everything it needs comes from an Android Studio installation. It does not
# install anything except, if necessary, a Gradle distribution into
# android/.gradle-dist (git-ignored) to bootstrap the wrapper - a fresh clone
# has gradle-wrapper.properties but no gradle-wrapper.jar, because that file is
# a binary and was not fabricated by hand.
#
# It prints what it found before using it, so a wrong guess is visible rather
# than buried in a Gradle error forty lines later.

set -u

GRADLE_VERSION="8.9"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1

say() { printf '%s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------
# 1. the JDK bundled with Android Studio
# --------------------------------------------------------------------------
JAVA_CANDIDATES=(
  "${JAVA_HOME:-}"
  "/c/Program Files/Android/Android Studio/jbr"
  "/c/Program Files/Android/Android Studio1/jbr"
  "${LOCALAPPDATA:-}/Programs/Android Studio/jbr"
  "${LOCALAPPDATA:-}/JetBrains/Toolbox/apps/AndroidStudio/ch-0/jbr"
  "/d/Program Files/Android/Android Studio/jbr"
  "/c/Program Files/Java/jdk-17"
  "/c/Program Files/Microsoft/jdk-17"
)

JDK=""
for candidate in "${JAVA_CANDIDATES[@]}"; do
  [ -n "$candidate" ] || continue
  if [ -x "$candidate/bin/java.exe" ] || [ -x "$candidate/bin/java" ]; then
    JDK="$candidate"
    break
  fi
done

[ -n "$JDK" ] || fail "no JDK found.
Install Android Studio (it bundles one at 'Android Studio/jbr'), or set
JAVA_HOME to a JDK 17 and run this again. Checked:
$(printf '  %s\n' "${JAVA_CANDIDATES[@]}")"

export JAVA_HOME="$JDK"
export PATH="$JAVA_HOME/bin:$PATH"
say "JAVA_HOME = $JAVA_HOME"
say "           $(java -version 2>&1 | head -1)"

# --------------------------------------------------------------------------
# 2. the Android SDK
# --------------------------------------------------------------------------
SDK_CANDIDATES=(
  "${ANDROID_SDK_ROOT:-}"
  "${ANDROID_HOME:-}"
  "${LOCALAPPDATA:-}/Android/Sdk"
  "${LOCALAPPDATA:-}/Android/sdk"
  "/c/Android/Sdk"
  "${USERPROFILE:-}/AppData/Local/Android/Sdk"
)

SDK=""
for candidate in "${SDK_CANDIDATES[@]}"; do
  [ -n "$candidate" ] || continue
  # platforms/ is the marker: an SDK directory can exist while still empty
  if [ -d "$candidate/platforms" ] || [ -d "$candidate/platform-tools" ]; then
    SDK="$candidate"
    break
  fi
done

[ -n "$SDK" ] || fail "no Android SDK found.
Open Android Studio once and let it finish its first-run setup, which
downloads the SDK. Checked:
$(printf '  %s\n' "${SDK_CANDIDATES[@]}")"

say "SDK      = $SDK"
say "           platforms: $(ls "$SDK/platforms" 2>/dev/null | tr '\n' ' ')"

# --------------------------------------------------------------------------
# 3. local.properties - machine-specific, never committed
# --------------------------------------------------------------------------
# Gradle wants a Windows path with escaped separators here.
SDK_WIN="$(cd "$SDK" && pwd -W 2>/dev/null || echo "$SDK")"
SDK_ESCAPED="${SDK_WIN//\//\\\\}"
cat > local.properties <<EOF
# Machine-specific. Git-ignored on purpose: this path is not portable.
# Regenerate with ./configure-and-build.sh
sdk.dir=$SDK_ESCAPED
EOF
say "wrote local.properties -> sdk.dir=$SDK_ESCAPED"

# --------------------------------------------------------------------------
# 4. a Gradle to run with
# --------------------------------------------------------------------------
RUN_GRADLE=""
if [ -f "gradle/wrapper/gradle-wrapper.jar" ] && [ -x "gradlew" ]; then
  RUN_GRADLE="./gradlew"
  say "using the existing Gradle wrapper"
elif command -v gradle >/dev/null 2>&1; then
  RUN_GRADLE="gradle"
  say "using Gradle from PATH: $(gradle --version 2>/dev/null | awk '/^Gradle/{print $2}')"
else
  DIST_DIR="$HERE/.gradle-dist"
  GRADLE_BIN="$DIST_DIR/gradle-$GRADLE_VERSION/bin/gradle"
  if [ ! -x "$GRADLE_BIN" ]; then
    say ""
    say "No Gradle wrapper jar and no gradle on PATH."
    say "Downloading Gradle $GRADLE_VERSION into .gradle-dist (git-ignored, ~130 MB)..."
    mkdir -p "$DIST_DIR" || fail "could not create $DIST_DIR"
    ZIP="$DIST_DIR/gradle-$GRADLE_VERSION-bin.zip"
    curl -fL --progress-bar -o "$ZIP" \
      "https://services.gradle.org/distributions/gradle-$GRADLE_VERSION-bin.zip" \
      || fail "download failed. Open the project in Android Studio once instead:
its first sync generates gradle-wrapper.jar, after which ./gradlew works."
    (cd "$DIST_DIR" && unzip -q -o "$(basename "$ZIP")") || fail "unzip failed"
    rm -f "$ZIP"
  fi
  RUN_GRADLE="$GRADLE_BIN"
  say "using $GRADLE_BIN"
  # leave a real wrapper behind so ./gradlew works from now on
  "$GRADLE_BIN" wrapper --gradle-version "$GRADLE_VERSION" >/dev/null 2>&1 \
    && say "generated the Gradle wrapper for future runs"
fi

# --------------------------------------------------------------------------
# 5. build
# --------------------------------------------------------------------------
say ""
say "=== $RUN_GRADLE assembleDebug ==="
"$RUN_GRADLE" assembleDebug --stacktrace
STATUS=$?

say ""
APK="app/build/outputs/apk/debug/app-debug.apk"
if [ $STATUS -eq 0 ] && [ -f "$APK" ]; then
  SIZE=$(du -h "$APK" | cut -f1)
  say "BUILD OK  ->  android/$APK  ($SIZE)"
  say ""
  say "Install it on a connected phone with:"
  say "  \"$SDK/platform-tools/adb\" install -r \"$APK\""
else
  say "BUILD FAILED (exit $STATUS). The Gradle output above says why."
fi
exit $STATUS
