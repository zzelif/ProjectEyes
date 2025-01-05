import os
from datetime import datetime
from data_preprocessing import process_dataset, balance_dataset, align_data
from tensorflow.keras.models import load_model
from models import frozen_mobilenetv2_model, finetuned_mobilenetv2_model, custom_model, final_cnn_model
from training import train_custom, train_frozen, finetune_mobilenet, build_final
from evaluate import plot_training, eval_model
# from run_openface import run_openface
# from action_units import extract_au

#Paths and Constants
train_data_dir = "Dataset/Train"
test_data_dir = "Dataset/Test"
predict_path = "Dataset/"
au_extr_path = "au features"
cache_path = "utils"
model_path = "models/model.h5"
openface_path = "OpenFace/FeatureExtraction.exe"
train_au_output_path = "utils/train_consolidated_au_2025-01-04_22.csv"
rt_frames_path = "processed"
img_size = (48, 48)
batch_size = 32
epochs = 25
unfrozen_layers = 40

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
m_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

#Load and preprocess dataset
train_gen, val_gen, test_gen = process_dataset(train_data_dir, test_data_dir, img_size, batch_size)
fr_train, fr_val, fr_test = process_dataset(train_data_dir, test_data_dir, img_size=(224, 224), batch_size=batch_size)
emotion_labels = list(train_gen.class_indices.keys())

# x_resampled, y_resampled = balance_dataset(train_gen, emotion_labels, img_size, batch_size)
frozen_x, frozen_y = balance_dataset(fr_train, emotion_labels, img_size=(224, 224), batch_size=batch_size)

x_train_img, x_val_img, x_train_au, x_val_au, y_train, y_val, label_map, au_data = align_data(train_data_dir, train_au_output_path, img_size)
fr_x_tr, fr_x_val, _, _, fr_y_tr, fr_y_val, fr_lb_map, _ = align_data(train_data_dir, train_au_output_path, img_size=(224, 224))

#Load the trained model to start image directory or realtime emotion prediction
if os.path.exists(model_path):
    print(f"Loading the trained {model_path}")
    model = load_model(model_path)

    # Run openface to extract AU features from images
    # run_openface(test_data_dir, output_path, openface_path)

    # Consolidate extracted AU Features into 1 CSV
    # extract_au(input_path, au_output_path)

else:
    # Build and Train the models
    print("Building the custom model...")
    custom_model = custom_model(img_size + (3,), (au_data.shape[1],), len(label_map))
    print("Training the custom model...")
    custom_history = train_custom(
        custom_model, [x_train_img, x_train_au], y_train,[x_val_img, x_val_au],
        y_val, batch_size=16, epochs=50, save_path=f"models/custom-{timestamp}.h5"
    )
    print("Evaluating the model...")
    eval_model(custom_model, [x_val_img, x_val_au], y_val)
    plot_training(custom_history, "custom model", output=f"metrics/custom_metrics-{m_timestamp}.png")

    print("Building the frozen mobilenetv2 model")
    frozen_model = frozen_mobilenetv2_model(input_shape_image=(224, 224, 3), emotion_labels=emotion_labels)
    print("Training the frozen model...")
    frozen_history = train_frozen(
        frozen_model, fr_x_tr, fr_y_tr, fr_x_val, fr_y_val, batch_size, epochs, save_path=f"models/frozen-{timestamp}.h5"
    )
    print("Evaluating the model...")
    eval_model(frozen_model, fr_x_val, fr_y_val)
    plot_training(frozen_history, "frozen model", output=f"metrics/frozen_metrics-{m_timestamp}.png")

    print("Starting to finetune the mobilenetv2 model")
    finetuned_model = finetuned_mobilenetv2_model(frozen_model, unfrozen_layers, lrn_rate=0.0001)
    print("Finetuning the mobilenet model...")
    fine_history = finetune_mobilenet(
        finetuned_model, frozen_x, frozen_y, fr_x_val, fr_y_val, batch_size, epochs=40, save_path=f"models/tuned_mb-{timestamp}.h5"
    )
    print("Evaluating the model...")
    eval_model(finetuned_model, fr_x_val, fr_y_val)
    plot_training(fine_history, "finetuned model", output=f"metrics/tuned_metrics-{m_timestamp}.png")

    print("Building the final cnn model")
    final_model = final_cnn_model(custom_model, finetuned_model, len(emotion_labels))
    print("Training the final model...")
    # No training concept yet...
