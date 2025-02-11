from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def train_custom(model, train_inputs, val_inputs, y_val, epochs, save_path, steps):
    """
    Custom cnn model. Initally a microexpression integrated with extracted action units.

    Args:
        model: Custom CNN with dataset image and action units as inputs
        train_inputs: Two inputs: Input shape of (48, 48, 3) and extracted Action Units' shape
        val_inputs: x_val_img and x_val_au which are returned from aligning image data with au_features then train_test_split
        y_val
        epochs
        save_path
        steps
       # class_weights
    Returns:
        model: History object after training custom model
    """

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, mode='auto', min_lr=1e-5, verbose=1)

    return model.fit(
        train_inputs,
        validation_data=(val_inputs, y_val),
        epochs=epochs,
        steps_per_epoch=steps,
       # class_weight=class_weights,
        callbacks=[early_stopping, checkpoint, lr_scheduler]
    )

def train_frozen(model, train_inputs, y_train, val_inputs, y_val, batch_size, epochs, class_weights, save_path):
    """
    Frozen MobileNetV2 CNN Model

    Args:
        model: Base MobileNetV2 with frozen layers
        train_inputs: input is a returned value from aligning image data with au_features then train_test_split
        y_train: input is a returned value from aligning image data with au_features then train_test_split
        val_inputs: inputs are returned values from aligning image data with au_features then train_test_split
        y_val: inputs are returned values from aligning image data with au_features then train_test_split
        batch_size: 32
        epochs: 50
        class_weights
        save_path: location

    Returns:
        model: History object after training pretrained model after freezing layers.
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    return model.fit(
        train_inputs, y_train,
        validation_data=(val_inputs, y_val),
        epochs=epochs,
        batch_size=batch_size,
       class_weight=class_weights,
        callbacks=[early_stopping, checkpoint, lr_scheduler]
    )

def finetune_mobilenet(model, train_inputs, y_train, val_inputs, y_val, batch_size, epochs, class_weights, save_path):
    """
    Finetune the MobileNetV2 CNN Model

    Args:
        model: frozen mobilenet
        train_inputs: input is a returned value from aligning image data with au_features then train_test_split
        y_train: Y input for training
        val_inputs:
        y_val:
        batch_size: 32
        epochs: 15 epochs. Shorter because it is finetuning
        class_weights
        save_path: location

    Returns:
        model: History object after finetuning the model.
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    checkpoint = ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, mode='auto', min_lr=1e-7, verbose=1)

    return model.fit(
        train_inputs, y_train,
        validation_data=(val_inputs, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[early_stopping, checkpoint, lr_scheduler]
    )

def build_final(model, train_inputs, y, val_inputs, y_val, epochs, save_path, steps):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, mode='auto', min_lr=1e-7, verbose=1)

    return model.fit(
        train_inputs, y,
        validation_data=(val_inputs, y_val),
        epochs=epochs,
        steps_per_epoch=steps,
        callbacks=[early_stopping, checkpoint, lr_scheduler]
    )