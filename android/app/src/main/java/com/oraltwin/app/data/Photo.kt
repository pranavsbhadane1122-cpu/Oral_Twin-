package com.oraltwin.app.data

import androidx.room.Dao
import androidx.room.Database
import androidx.room.Entity
import androidx.room.Insert
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.RoomDatabase

/**
 * Photo metadata only. The image file itself stays in app-private storage and
 * this row points at it by relative path.
 *
 * Nothing here records a condition, a severity or a model output, because the
 * app produces none: the classifier and the lesion segmenter both failed their
 * acceptance gates and are not shipped. See README-android.md.
 */
@Entity(tableName = "photos")
data class PhotoEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val capturedAtEpochMs: Long,
    val relativePath: String,
    /** Set in M1 by the guided-capture checks: focus, framing, mouth visible. */
    val usable: Boolean? = null,
    val note: String? = null,
)

@Dao
interface PhotoDao {
    @Insert
    fun insert(photo: PhotoEntity): Long

    @Query("SELECT COUNT(*) FROM photos")
    fun count(): Int

    @Query("SELECT * FROM photos ORDER BY capturedAtEpochMs DESC")
    fun all(): List<PhotoEntity>
}

@Database(entities = [PhotoEntity::class], version = 1, exportSchema = false)
abstract class OralTwinDatabase : RoomDatabase() {
    abstract fun photoDao(): PhotoDao
}
