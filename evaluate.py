import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# import os
# import cv2
# from sklearn.metrics import confusion_matrix

def plot_training(history, label, output):
    plt.figure(figsize=(16, 8))

    plt.subplot(2, 2, 1)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs_range = range(len(acc))
    plt.plot(epochs_range, acc, label=f'{label} Training Accuracy')
    plt.plot(epochs_range, val_acc, label=f'{label} Validation Accuracy')

    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 2, 2)
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(loss))
    plt.plot(epochs_range, loss, label=f'{label} Training Loss')
    plt.plot(epochs_range, val_loss, label=f'{label} Validation Loss')

    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.subplot(2, 2, 3)
    rec = history.history['recall']
    val_rec = history.history['val_recall']
    epochs_range = range(len(rec))
    plt.plot(epochs_range, rec, label=f'{label} Training Recall')
    plt.plot(epochs_range, val_rec, label=f'{label} Validation Recall')

    plt.legend(loc='lower left')
    plt.title('Training and Validation Recall')

    plt.subplot(2, 2, 4)
    pre = history.history['precision']
    val_pre = history.history['val_precision']
    epochs_range = range(len(pre))
    plt.plot(epochs_range, pre, label=f'{label} Training Precision')
    plt.plot(epochs_range, val_pre, label=f'{label} Validation Precision')

    plt.legend(loc='lower right')
    plt.title('Training and Validation Precision')

    plt.tight_layout()
    plt.savefig(output)
    plt.close()

def eval_model(model, x, y):
    loss, accuracy, precision, recall, auc = model.evaluate(x, y)

    print(f"Validation Accuracy: {accuracy:.2f}")
    print(f"Validation Precision: {precision:.2f}")
    print(f"Validation Recall: {recall:.2f}")
    print(f"Validation AUC: {auc:.2f}")
