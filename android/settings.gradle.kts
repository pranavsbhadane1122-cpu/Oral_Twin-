pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\.android.*")
                includeGroupByRegex("com\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        // OpenCV publishes org.opencv:opencv to Maven Central from 4.9.0 onwards.
        // That is why this project needs no manual SDK import - see README-android.md.
        mavenCentral()
    }
}

rootProject.name = "OralTwin"
include(":app")
