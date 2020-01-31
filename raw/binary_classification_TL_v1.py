import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

import os
from matplotlib import pyplot as plt
import numpy as np


# ENABLE GPU
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print('GPU device not found')
print('Found GPU at: {}'.format(device_name))


# TRAINING PARAMETERS
INPUT_SIZE = 299
SHUFFLE_BUFFER_SIZE = 1000
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001

optimizer = 'adam'
loss_func = 'binary_crossentropy'


# DOWNLOAD 'NON-FOOD' DATASET
non_food_train_raw = tfds.load("downsampled_imagenet/64x64", split="train")
non_food_val_raw = tfds.load("downsampled_imagenet/64x64", split="validation")


# DOWNLOAD 'FOOD' DATASET
food_train_raw = tfds.load("food101", split="train[:80]")
food_val_raw = tfds.load("food101", split="train[80:]")


# REDUCE DATASETS
TRAINING_SIZE = 10000
VAL_SIZE = 1000

food_train_raw = food_train_raw.shuffle(SHUFFLE_BUFFER_SIZE).take(TRAINING_SIZE//2)
non_food_train_raw = non_food_train_raw.shuffle(SHUFFLE_BUFFER_SIZE).take(TRAINING_SIZE//2)
food_val_raw = food_val_raw.shuffle(SHUFFLE_BUFFER_SIZE).take(VAL_SIZE//2)
non_food_val_raw = non_food_val_raw.shuffle(SHUFFLE_BUFFER_SIZE).take(VAL_SIZE//2)


# FORMAT AND LABEL IMAGES
def formatImage(img):
  img = tf.cast(img, tf.float32)
  img /= 255
  img = tf.image.resize_with_pad(img, INPUT_SIZE, INPUT_SIZE)
  return img

def formatNonFood(features):
  img = features["image"]
  img = formatImage(img)
  return img, 0

def formatFood(features):
  img = features["image"] 
  img = formatImage(img)
  return img, 1

food_train_raw = food_train_raw.map(formatFood)
food_val_raw = food_val_raw.map(formatFood)
non_food_train_raw = non_food_train_raw.map(formatNonFood)
non_food_val_raw = non_food_val_raw.map(formatNonFood)


# MERGE DATASETS
train_raw = food_train_raw.concatenate(non_food_train_raw)
val_raw = food_val_raw.concatenate(non_food_val_raw)


# PREPARE DATASETS
train_batches = train_raw.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
val_batches = val_raw.batch(BATCH_SIZE, drop_remainder=True)


# SHOW EXAMPLE IMAGES
cols, rows = 4, 5
fig = plt.figure(figsize=(20,10))
fig.suptitle("Random images from dataset")

examples = train_batches.unbatch().batch(20).as_numpy_iterator()

images, labels = next(examples)

for i in range(1, cols*rows+1):
  fig.add_subplot(rows, cols, i)

  img = images[i-1]

  if labels[i-1] == 1:
    plt.title("food")
  else:
    plt.title("non-food")

  plt.axis("off")
  plt.imshow(img, interpolation="nearest")
plt.show()


# DEFINE MODEL
base_model = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4", output_shape=[2048], trainable=False)

model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.Dense(1, activation='sigmoid')
])

model.build([None, INPUT_SIZE, INPUT_SIZE, 3])

model.compile(optimizer=optimizer, loss=loss_func, metrics=["accuracy"])

model.summary()


# TRAIN MODEL
history = model.fit(train_batches, validation_data=val_batches, epochs=EPOCHS)


# SAVE MODEL WEIGHTS
MODEL_PATH = os.path.join("models", "binary_classifier_TL_v1.h5")
model.save(MODEL_PATH, overwrite=True)


# EVALUATE
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()