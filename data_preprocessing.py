import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from keras.src.utils import to_categorical
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence

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

    augmentor = ImageDataGenerator(
        rotation_range=60,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='nearest'
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

    return train_gen, val_gen, test_gen, augmentor

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

                if image_id not in au_features.index:
                    continue

                img_path = os.path.join(root, file)
                img = tf.keras.utils.load_img(img_path, target_size=img_size)
                img_array = tf.keras.utils.img_to_array(img) / 255.0

                au = au_features.loc[image_id].values
                image_data.append(img_array)
                au_data.append(au)
                labels.append(emotion)

    image_data = np.asarray(image_data, dtype='float32')
    au_data = np.array(au_data, dtype='float32')
    labels = np.asarray(labels)

    label_map = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    encoded_labels = to_categorical([label_map[label] for label in labels])

    x_train_img, x_val_img, x_train_au, x_val_au, y_train, y_val = train_test_split(
        image_data, au_data, encoded_labels, test_size=0.2, random_state=42
    )

    class_weights = compute_class_weight(
        'balanced', classes=np.unique(y_train.argmax(axis=1)), y=y_train.argmax(axis=1)
    )

    class_weights = dict(enumerate(class_weights))

    return label_map, au_data, image_data


def align_data_au(image_dir, au_csv, img_size):
    au_features = pd.read_csv(au_csv)
    au_features.set_index('image_id', inplace=True)

    image_data, au_data, labels = [], [], []
    emotion = os.path.basename(image_dir)

    for file in os.listdir(image_dir):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            image_id = os.path.splitext(file)[0]

            if image_id not in au_features.index:
                continue

            img_path = os.path.join(image_dir, file)
            img = tf.keras.utils.load_img(img_path, target_size=img_size)
            img_array = tf.keras.utils.img_to_array(img) / 255.0

            au = au_features.loc[image_id].values
            image_data.append(img_array)
            au_data.append(au)
            labels.append(emotion)

    image_data = np.asarray(image_data, dtype='float32')
    au_data = np.array(au_data, dtype='float32')
    labels = np.asarray(labels)

    label_map = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    # encoded_labels = to_categorical([label_map[label] for label in labels])

    return au_data, image_data


class AugmentWithAU(Sequence):
    """
    Combines ImageDataGenerator with AU feature alignment.

    Args:
        images (numpy array): Array of images.
        au_features (numpy array): Array of AU features.
        labels (numpy array): Array of labels.
        batch_size (int): Batch size for training.
        augmentor (ImageDataGenerator): Augmentation instance.
    """
    def __init__(self, images, au_features, labels, batch_size, augmentor):
        self.images = images
        self.au_features = au_features
        self.labels = labels
        self.batch_size = batch_size
        self.augmentor = augmentor
        self.steps = math.ceil(len(images) / batch_size)  # Total steps per epoch

    def __len__(self):
        return self.steps  # Number of batches per epoch

    def __getitem__(self, idx):
        # Calculate the start and end indices for this batch
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.images))  # Avoid going out of bounds

        # Extract the batch of images, AUs, and labels
        img_batch = self.images[start:end]
        label_batch = self.labels[start:end]
        au_batch = self.au_features[start:end]

        # Apply augmentation to the image batch
        augmented_images = next(self.augmentor.flow(
            img_batch, label_batch, batch_size=len(img_batch), shuffle=False
        ))

        return [augmented_images, au_batch], label_batch