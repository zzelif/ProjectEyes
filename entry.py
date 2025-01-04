import os
# from datetime import datetime
from data_preprocessing import process_dataset, balance_dataset, align_data
from tensorflow.keras.models import load_model
from models import frozen_mobilenetv2_model, finetuned_mobilenetv2_model, custom_model
# from run_openface import run_openface
# from action_units import extract_au

#Paths and Constants
train_data_dir = "Dataset/Train"
test_data_dir = "Dataset/Test"
predict_path = "Dataset/"
au_extr_path = "au features"
cache_path = "utils"
model_path = "models/"
openface_path = "OpenFace/FeatureExtraction.exe"
train_au_output_path = "utils/train_consolidated_au_2025-01-04_22.csv"
rt_frames_path = "processed"
img_size = (48, 48)
batch_size = 32
epochs = 25
unfrozen_layers = 35

# timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#Load and preprocess dataset
train_gen, val_gen, test_gen = process_dataset(train_data_dir, test_data_dir, img_size, batch_size)
emotion_labels = list(train_gen.class_indices.keys())
x_resampled, y_resampled = balance_dataset(train_gen, emotion_labels, img_size, batch_size)
x_train_img, x_val_img, x_train_au, x_val_au, y_train, y_val, label_map, au_data = align_data(train_data_dir, train_au_output_path, img_size)

#Load the trained model to start image directory or realtime emotion prediction
if os.path.exists(model_path):
    print(f"Loading the trained {model_path}")
    model = load_model(model_path)

    # Run openface to extract AU features from images
    # run_openface(test_data_dir, output_path, openface_path)

    # Consolidate extracted AU Features into 1 CSV
    # extract_au(input_path, au_output_path)

else:
    # Build the models
    print("Building the custom model...")
    custom_model = custom_model(img_size + (3,), (au_data.shape[1],), len(label_map))

    print("Building the frozen mobilenetv2 model")
    frozen_model = frozen_mobilenetv2_model(input_shape_image=(224, 224, 3), emotion_labels=emotion_labels)

    print("Finetuning the mobilenetv2 model")
    finetuned_model = finetuned_mobilenetv2_model(frozen_model, unfrozen_layers)

    # print("Building the final cnn model")
    # final_model = final_cnn_model(model_1, finetuned_model, emotion_labels)