import tensorflow as tf
from tensorflow import keras
import numpy as np


TRAIN_PATH = "images/train"
VALIDATION_PATH = "images/validate"

# labels
TRAIN_LABELS = np.array([0, 1], dtype='float')
VALIDATION_LABELS = np.array([0, 1], dtype='float')

# TRAINING
training_data = keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")
train = training_data.flow_from_directory(directory=TRAIN_PATH, shuffle=True, target_size=(226, 226), class_mode="binary")


# VALIDATION
validation_data = keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

validation = training_data.flow_from_directory(directory=VALIDATION_PATH, shuffle=False, target_size=(226, 226), class_mode="binary")

model = keras.Sequential([
    keras.layers.Conv2D(64, (24, 24), activation="relu", input_shape=(226, 226, 3)),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Conv2D(64, (24, 24), activation="relu", input_shape=(226, 226, 3)),
    keras.layers.MaxPooling2D(2, 2),


    keras.layers.Flatten(),

    keras.layers.Dense(128, activation="relu"),

    keras.layers.Dense(128, activation="relu"),

    keras.layers.Dense(2, activation="softmax")

])

# COMPILING THE MODEL
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

model.summary()

# TRAINING THE MODEL
model.fit(train, epochs=2)

# EVALUATING THE MODEL
model.evaluate(validation)

model.save("mask_detect.model")

print("MODEL SAVED")
