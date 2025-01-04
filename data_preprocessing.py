import os
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#
# Process and augments the data from the dirs using ImageDataGenerator
def process_dataset(train_data_dir, test_data_dir, img_size, batch_size):
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
def balance_dataset(train_gen, emotion_labels, img_size, batch_size):
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

#
#
def align_data(image_dir, au_csv, img_size):
    au_features = pd.read_csv(au_csv)
    au_features.set_index('image_id', inplace=True)

    image_data, au_data, labels = [], [], []
    for root, _, files, in os.walk(image_dir):
        if root == image_dir:
            continue
        emotion = os.path.basename(root)

        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_id = os.path.splitext(file)[0]

                if image_id in au_features.index:
                    img_path = os.path.join(root, file)
                    img = tf.keras.utils.load_img(img_path, target_size=img_size)
                    img_array = tf.keras.utils.img_to_array(img) / 255.0

                    au = au_features.loc[image_id].values
                    image_data.append(img_array)
                    au_data.append(au)
                    labels.append(emotion)

    image_data = np.asarray(image_data)
    au_data = np.array(au_data)
    labels = np.asarray(labels)
    label_map = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    encoded_labels = np.array([label_map[label] for label in labels])
    encoded_labels = to_categorical(encoded_labels)

    x_train_img, x_val_img, x_train_au, x_val_au, y_train, y_val = train_test_split(
        image_data, au_data, encoded_labels, test_size=0.2, random_state=42
    )

    x_train_img = x_train_img.astype('float32')
    x_train_au = x_train_au.astype('float32')
    y_train = y_train.astype('float32')
    x_val_img = x_val_img.astype('float32')
    x_val_au = x_val_au.astype('float32')
    y_val = y_val.astype('float32')

    return x_train_img, x_val_img, x_train_au, x_val_au, y_train, y_val, label_map, au_data