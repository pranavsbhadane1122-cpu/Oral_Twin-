package com.oraltwin.app.data

import androidx.room.Dao
import androidx.room.Database
import androidx.room.Delete
import androidx.room.Entity
import androidx.room.Insert
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.RoomDatabase
import kotlinx.coroutines.flow.Flow

/**
 * Photo metadata only. The JPEG lives in app-internal storage and this row
 * points at it by filename.
 *
 * Nothing here records a condition, a severity or a model output, because the
 * app produces none: the classifier and the lesion segmenter both failed their
 * acceptance gates and are not shipped. See README-android.md.
 */
@Entity(tableName = "photos")
data class PhotoEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    /** Bare filename within filesDir/photos. Never an absolute path: those
     *  change when the app is reinstalled or moved to another device. */
    val filename: String,
    val capturedAtEpochMs: Long,
    val width: Int,
    val height: Int,
)

@Dao
interface PhotoDao {
    @Insert
    suspend fun insert(photo: PhotoEntity): Long

    @Delete
    suspend fun delete(photo: PhotoEntity)

    @Query("DELETE FROM photos WHERE id = :id")
    suspend fun deleteById(id: Long): Int

    @Query("SELECT * FROM photos ORDER BY capturedAtEpochMs DESC")
    fun observeAll(): Flow<List<PhotoEntity>>

    /** Same ordering, but a one-shot read - used by tests and by cleanup. */
    @Query("SELECT * FROM photos ORDER BY capturedAtEpochMs DESC")
    suspend fun all(): List<PhotoEntity>

    @Query("SELECT * FROM photos WHERE id = :id")
    suspend fun byId(id: Long): PhotoEntity?

    @Query("SELECT COUNT(*) FROM photos")
    suspend fun count(): Int

    // blocking variants, for the M0 stack check and instrumented assertions
    @Insert
    fun insertBlocking(photo: PhotoEntity): Long

    @Query("SELECT COUNT(*) FROM photos")
    fun countBlocking(): Int
}

@Database(entities = [PhotoEntity::class], version = 1, exportSchema = false)
abstract class OralTwinDatabase : RoomDatabase() {
    abstract fun photoDao(): PhotoDao
}
