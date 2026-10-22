package com.oraltwin.app.comparison

import com.oraltwin.app.data.PhotoEntity

/**
 * Exactly two photographs, ordered by when they were taken.
 *
 * ORDER COMES FROM THE TIMESTAMP, NEVER FROM TAP ORDER. Tapping the newer one
 * first is an ordinary thing to do, and if tap order decided which was
 * "older", every subsequent comparison would be inverted - the overlay would
 * warp the wrong direction and any future change measurement would have its
 * sign flipped. Nothing about the result would look wrong.
 *
 * Pure, so this is unit-testable on a desktop JVM without OpenCV or Android.
 */
data class PairSelection(
    val older: PhotoEntity,
    val newer: PhotoEntity,
) {
    init {
        require(older.capturedAtEpochMs <= newer.capturedAtEpochMs) {
            "PairSelection must be ordered oldest first"
        }
        require(older.id != newer.id) { "a photo cannot be compared with itself" }
    }

    companion object {
        /**
         * Build a selection from exactly two photographs, in whatever order
         * they were tapped. Returns null for any other count, so "exactly two"
         * is enforced here rather than by the UI remembering to.
         */
        fun from(selected: Collection<PhotoEntity>): PairSelection? {
            if (selected.size != 2) return null
            val distinct = selected.distinctBy { it.id }
            if (distinct.size != 2) return null
            val sorted = distinct.sortedBy { it.capturedAtEpochMs }
            return PairSelection(older = sorted[0], newer = sorted[1])
        }
    }
}
