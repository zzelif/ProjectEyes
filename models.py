from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add, Input, Dense, Conv2D, Flatten, Concatenate, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.metrics import AUC

def custom_model(input_shape_image, input_shape_au, emotion_labels):
    """
    Custom cnn model. Initally a microexpression integrated with extracted action units.

    Args:
        input_shape_image: Input shape of (48, 48, 3).
        input_shape_au: Input shape of extracted AU Features
        emotion_labels: Contains the emotion labels or the classes. len will be used as the last dense layer

    Returns:
        model:
    """
    img_input = Input(shape=input_shape_image, name="image_input")

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    residual = Conv2D(64, (1, 1), activation='relu', padding='same')(img_input)
    x= Add()([x, residual])

    # x = Flatten()(x)
    # x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
    # x = Dropout(0.5)(x)
    # x = Dense(32, activation='relu')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)

    au_input = Input(shape=input_shape_au, name="au_input")
    y = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(au_input)
    y = BatchNormalization()(y)
    y = Dense(32, activation='relu')(y)

    merged = Concatenate()([x, y])
    merged = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(merged)
    merged = Dropout(0.5)(merged)
    outputs = Dense(emotion_labels, activation='softmax')(merged)

    model = Model(inputs=[img_input, au_input], outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'Precision', 'Recall', AUC()]
    )

    model.summary()
    return model

def frozen_mobilenetv2_model(input_shape_image, emotion_labels):
    """
    Freezing the first layers of the mobilenetv2 for initial training on dataset.

    Args:
        input_shape_image: Input shape of (228, 228, 3).
        emotion_labels (list): Contains the emotion labels or the classes. len will be used as the last dense layer

    Returns:
        model: Hisotry object after training pretrained model after freezing some layers.
    """
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape_image)
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(len(emotion_labels), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'Precision', 'Recall', AUC()])

    # model.summary()
    return model

def finetuned_mobilenetv2_model(model, unfrozen_layers, lrn_rate):
    """
    Fine-tune a pre-trained MobileNetV2 model by unfreezing layers starting from a specified index.

    Args:
        model: The pre-trained MobileNetV2 model.
        unfrozen_layers (int): Index of the layer from which to start unfreezing layers.
        lrn_rate: adjust depending on the

    Returns:
        model: Hisotry object after finetuning the model.
    """

    for layer in model.layers[-unfrozen_layers:]:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=lrn_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'Precision', 'Recall', AUC()]
    )

    # model.summary()
    return model


def final_cnn_model(model_1, model_2, emotion_labels):
    combined_input = Concatenate()([model_1.output, model_2.output])
    x = Dense(128, activation='relu')(combined_input)
    x = Dense(64, activation='relu')(x)

    final_output = Dense(emotion_labels, activation='softmax')(x)
    model = Model(inputs=[model_1.input, model_2.input], outputs=final_output)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'Precision', 'Recall', AUC()]
    )

    # model.summary()
    return model