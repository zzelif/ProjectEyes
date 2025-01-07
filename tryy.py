import numpy as np
import pandas as pd
import os
import cv2
from collections import Counter
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from data_preprocessing import process_dataset, align_data_au
from action_units import extract_au
from run_openface import run_openface

m_timestamp = datetime.now().strftime("%Y-%m-%d_%H")

specific_aus = {
    "Angry": [4, 9, 5, 17],
    "Happy": [6, 7, 10, 12, 14, 20],
    "Neutral": [2, 5],
    "Sad": [1, 4, 6, 7, 9, 12, 15, 17, 20],
    "Surprised": [1, 2, 5, 25, 26]
}

pr_model = load_model("models/custom/custom-2025-01-06_03-11.h5")

train_data_dir = "Dataset/Train"
train_au_output_path = "utils/train_consolidated_au_2025-01-04_22.csv"
test_data_dir = "Dataset/Test/Surprised"
openface_path = "OpenFace/FeatureExtraction.exe"
test_au_out_dir = "au features"
cons_au_in_dir = os.path.join(test_au_out_dir, m_timestamp)
test_au_extr = f"utils/consolidated_data-{m_timestamp}.csv"
img_size = (48, 48)
batch_size = 32

train_gen, val_gen, test_gen, augment = process_dataset(train_data_dir, test_data_dir, img_size, batch_size)
# _, _, _, _, _, _, label_map, _, _ = align_data(train_data_dir, train_au_output_path, img_size)

# def preprocess_image(image_path, target_size=(224, 224)):
#     """
#     Preprocess the image for prediction.
#
#     Args:
#         image_path (str): Path to the image file.
#         target_size (tuple): Target size for the image.
#
#     Returns:
#         numpy array: Preprocessed image.
#     """
#     image = load_img(image_path, target_size=target_size)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = img_to_array(image)
#     image = np.expand_dims(image, axis=0) / 255.0
#     return image.astype('float32')
#
# def predict_image(model, image_path):
#     """
#     Predict the emotion of an image using the trained model.
#
#     Args:
#         model: Trained Keras model.
#         image_path (str): Path to the image file.
#
#     Returns:
#         str: Predicted emotion.
#     """
#     image = preprocess_image(image_path)
#     predictions = model.predict(image)
#     class_idx = np.argmax(predictions)
#
#     confidence = predictions[0][class_idx]
#     predicted_emotion = list(label_map.keys())[list(label_map.values()).index(class_idx)]
#     return predicted_emotion, confidence
#
# def predict_images_in_directory(model1, directory_path):
#     """
#     Predict emotions for all images in a directory.
#
#     Args:
#         model1: Trained Keras model.
#         directory_path (str): Path to the directory containing images.
#
#     Returns:
#         list: Predictions with confidence scores for each image.
#     """
#     predictions = []
#     tally = Counter()
#
#     for filename in os.listdir(directory_path):
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#             image_path = os.path.join(directory_path, filename)
#             label, confidence = predict_image(model1, image_path)
#
#             if label is not None:
#                 tally[label] += 1
#                 predictions.append((filename, label, confidence))
#                 print(f"Image: {filename}, Predicted: {label}, Confidence: {confidence:.2f}")
#             else:
#                 continue
#     print("\nEmotion Tally:")
#     for emotion, count in tally.items():
#         print(f"{emotion}: {count}")
#     return tally

def refine_predictions(aus_data, model_preds, spec_aus, emotion_label):
    """
    Refine model predictions using AU relevance.
    Args:
        aus_data: CSV containing the AU Features.
        model_preds: Probabilities predicted by the model.
        spec_aus: Dictionary of relevant AUs for each emotion.
        emotion_label: Mapping of class indices to emotion labels.
    Returns:
        Refined emotion label.
    """
    df = pd.read_csv(aus_data)

    preds_label = np.argmax(model_preds, axis=1)[0]
    pred_emotion = emotion_label[preds_label]
    conf = model_preds[0][preds_label]

    relevant_aus = spec_aus[pred_emotion]
    au_scores = []

    for au in relevant_aus:
        try:
            au_intensity = df[f"AU{au:02d}_r"]
            au_occurences = df[f"AU{au:02d}_c"]

            au_scores.append(au_intensity * au_occurences)

        except KeyError:
            print(f"Warning: AU{au:02d} not found in data.")
            au_scores.append(0)

    if len(au_scores) > 0:
        relevance_score = sum(au_scores) / len(relevant_aus)

    else:
        relevance_score = 0

    final_score = 0.7 * conf + 0.3 * relevance_score

    return final_score, pred_emotion


# Assuming the original emotion_map is:
emotion_map = {0: "angry", 1: "happy", 2: "neutral", 3: "sad", 4: "surprised"}

# Capitalize values using a dictionary comprehension
emotion_map = {key: value.capitalize() for key, value in emotion_map.items()}

print("Running OpenFace FeatureExtraction")
run_openface(test_data_dir, test_au_out_dir, openface_path, m_timestamp)
print("Consolidating the extracted Action Units")
extract_au(cons_au_in_dir, test_au_extr)
print("Aligning the images and the consolidated extracted aus")
au_data, image_data = align_data_au(test_data_dir, test_au_extr, img_size)
print(f"Shapes of the data: {au_data.shape}, {image_data.shape}")
preds = pr_model.predict([image_data, au_data])
print(preds)
emotion_tally = {emotion: 0 for emotion in emotion_map.values()}
# Iterate over each row in `preds` to get predictions for all instances
for i, row in enumerate(preds):  # `row` corresponds to predictions for each input
    class_idx = np.argmax(row)  # Find the class index with the highest score for this row
    emotion_label = emotion_map[class_idx]  # Get the label from the emotion map
    confidence_score = row[class_idx]  # Confidence for the predicted class
    emotion_tally[emotion_label] += 1  # Increment the count for this emotion

    # Print results for each row
    print(f"Row {i}: Predicted emotion: {emotion_label}")
    print(f"Row {i}: Confidence score: {confidence_score:.2f} ({confidence_score * 100:.2f}%)")

# Print the tally
print("Emotion Tally:")
for emotion, count in emotion_tally.items():
    print(f"{emotion}: {count}")

# print("Making predictions")
# # Iterate over each image and corresponding AU data
# for i in range(len(image_data)):
#     # Select the ith image and corresponding AU data
#     single_image = np.expand_dims(image_data[i], axis=0)  # Add batch dimension for the image
#     single_au = np.expand_dims(au_data[i], axis=0)  # Add batch dimension for AU data
#
#     # Predict for the single sample
#     preds = pr_model.predict([single_image, single_au])
#
#     # Refine predictions for the single row of AU data
#     single_au_row = test_au_extr.iloc[i] if isinstance(test_au_extr, pd.DataFrame) else test_au_extr[i]
#     fin_score, predict_emotion = refine_predictions(test_au_extr, preds, specific_aus, emotion_map)
#
#     # Print prediction for this image
#     print(f"Image {i+1}: Final prediction: {predict_emotion} with confidence: {fin_score:.2f}")
