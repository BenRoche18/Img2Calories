package org.tensorflow.lite.examples.detection.fooddata;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.room.ColumnInfo;
import androidx.room.Entity;
import androidx.room.PrimaryKey;

@Entity
public class Food {
    @PrimaryKey(autoGenerate = true)
    public int id;

    @ColumnInfo
    @NonNull
    public String name;

    @ColumnInfo(name = "energy", defaultValue = "0")
    public int energy;

    @ColumnInfo(name = "protein", defaultValue = "0")
    public float protein;

    @ColumnInfo(name = "fats", defaultValue = "0")
    public float fats;

    @ColumnInfo(name = "carbohydrates", defaultValue = "0")
    public float carbohydrates;

    @ColumnInfo(name = "sugars", defaultValue = "0")
    public float sugars;

    @ColumnInfo(name = "density", defaultValue = "1")
    public float density;

    @ColumnInfo(name = "depth", defaultValue = "5")
    public float avgDepth;
}