import os
import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#
# Process and augments the data from the dirs using ImageDataGenerator
def process_data(train_data_dir, test_data_dir, img_size, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=60,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255
    )

    train_gen = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_gen = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    test_gen = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
    )

    return train_gen, val_gen, test_gen

#
#
def balance_data(train_gen, emotion_labels, img_size, batch_size):
    images, labels = [], []

    for i in range(len(train_gen)):
        batch_images, batch_labels = train_gen[i]
        images.append(batch_images)
        labels.append(batch_labels)
        if len(images) * batch_size >= train_gen.samples:
            break

    images = np.vstack(images)
    labels = np.argmax(np.vstack(labels), axis=1)

    x_train_flat = images.reshape((images.shape[0], -1))
    smote = SMOTE(random_state=42)
    x_resampled, y_resampled = smote.fit_resample(x_train_flat, labels)

    x_resampled = x_resampled.reshape((-1, img_size[0], img_size[1], 3))
    y_resampled = tf.keras.utils.to_categorical(y_resampled, num_classes=len(emotion_labels))

    return x_resampled, y_resampled
import os
import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#
# Process and augments the data from the dirs using ImageDataGenerator
def process_data(train_data_dir, test_data_dir, img_size, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=60,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255
    )

    train_gen = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_gen = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    test_gen = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
    )

    return train_gen, val_gen, test_gen

#
#
def balance_data(train_gen, emotion_labels, img_size, batch_size):
    images, labels = [], []

    for i in range(len(train_gen)):
        batch_images, batch_labels = train_gen[i]
        images.append(batch_images)
        labels.append(batch_labels)
        if len(images) * batch_size >= train_gen.samples:
            break

    images = np.vstack(images)
    labels = np.argmax(np.vstack(labels), axis=1)

    x_train_flat = images.reshape((images.shape[0], -1))
    smote = SMOTE(random_state=42)
    x_resampled, y_resampled = smote.fit_resample(x_train_flat, labels)

    x_resampled = x_resampled.reshape((-1, img_size[0], img_size[1], 3))
    y_resampled = tf.keras.utils.to_categorical(y_resampled, num_classes=len(emotion_labels))

    return x_resampled, y_resampled