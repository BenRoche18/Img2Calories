package org.tensorflow.lite.examples.detection.fooddata;

import androidx.room.Dao;
import androidx.room.Delete;
import androidx.room.Insert;
import androidx.room.OnConflictStrategy;
import androidx.room.Query;

import java.util.List;

@Dao
public interface FoodDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    void insertAll(Food... foods);

    @Delete
    void delete(Food food);

    @Query("SELECT * FROM food WHERE name = :name")
    Food getByName(String name);

    @Query("SELECT * FROM food")
    List<Food> getAll();
}