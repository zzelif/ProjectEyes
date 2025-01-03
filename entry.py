#
#Entry point for the project
#

import os
# from datetime import datetime
from data_preprocessing import process_data, balance_data
from tensorflow.keras.models import load_model
from models import custom_model, frozen_mobilenetv2_model, finetuned_mobilenetv2_model
# from run_openface import run_openface
# from action_units import extract_au

#Paths anc Constants
train_data_dir = "Dataset/Train"
test_data_dir = "Dataset/Test"
output_path = "au features"
predict_path = "Dataset/"
cache_path = "utils/cached_data.pkl"
model_path = "models/"
openface_path = "OpenFace/FeatureExtraction.exe"
input_path = "au features"
au_output_path = "utils/consolidated_au_features.csv"
img_size = (48, 48)
batch_size = 32
epochs = 25

# timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#Load and preprocess dataset
train_gen, val_gen, test_gen = process_data(train_data_dir, test_data_dir, img_size, batch_size)
emotion_labels = list(train_gen.class_indices.keys())
x_resampled, y_resampled = balance_data(train_gen, emotion_labels, img_size, batch_size)

# Run openface to extract AU features from images
# run_openface(test_data_dir, output_path, openface_path)

# Consolidate extracted AU Features into 1 CSV
# extract_au(input_path, au_output_path)

#Load the trained model to start image directory or realtime emotion prediction
if os.path.exists(model_path):
    print(f"Loading the trained {model_path}")
    model = load_model(model_path)


else:
    # Build the model
    print("Building the model...")
#
#Entry point for the project
#

# from datetime import datetime
from data_preprocessing import process_data, balance_data
from tensorflow.keras.models import load_model
import os

#Paths anc Constants
train_data_dir = "Dataset/Train"
test_data_dir = "Dataset/Test"
predict_path = "Dataset/"
cache_path = "utils/cached_data.pkl"
model_path = "models/"
img_size = (48, 48)
batch_size = 32
epochs = 25
unfrozen_layers = 35

# timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#Load and preprocess data
train_gen, val_gen, test_gen = process_data(train_data_dir, test_data_dir, img_size, batch_size)
emotion_labels = list(train_gen.class_indices.keys())
x_resampled, y_resampled = balance_data(train_gen, emotion_labels, img_size, batch_size)

#Load the trained model to start image directory or realtime emotion prediction
if os.path.exists(model_path):
    print(f"Loading the trained {model_path}")
    model = load_model(model_path)

else:
    # Build the models
    # print("Building the custom model...")
    # custom_model = custom_model(img_size + (3,), emotion_labels)

    print("Building the frozen mobilenetv2 model")
    frozen_model = frozen_mobilenetv2_model(input_shape_image=(224, 224, 3), emotion_labels=emotion_labels)

    print("Finetuning the mobilenetv2 model")
    finetuned_model = finetuned_mobilenetv2_model(frozen_model, unfrozen_layers)
