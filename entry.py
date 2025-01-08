import math
import os
from datetime import datetime
from data_preprocessing import process_dataset, balance_dataset, align_data, align_data_au, AugmentWithAU
from tensorflow.keras.models import load_model
from models import frozen_mobilenetv2_model, finetuned_mobilenetv2_model, custom_model, final_cnn_model
from training import train_custom, train_frozen, finetune_mobilenet, build_final
from evaluate import plot_training, eval_model, plot_matrix
# from run_openface import run_openface
# from action_units import extract_au_directory

#Paths and Constants
train_data_dir = "Dataset/Train"
test_data_dir = "Dataset/Test"
predict_path = "Dataset/"
au_extr_path = "au_features"
cache_path = "utils"
model_path = "models/model.h5"
openface_path = "OpenFace/FeatureExtraction.exe"
train_au_output_path = "utils/train_consolidated_au_2025-01-04_22.csv"
rt_frames_path = "processed"
img_size = (48, 48)
batch_size = 32
cust_batch = 16
epochs = 25

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
m_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

#Load and preprocess dataset
_, augment, _ = process_dataset(train_data_dir, test_data_dir, img_size, batch_size)
fr_train, _, fine_gen = process_dataset(train_data_dir, test_data_dir, img_size=(224, 224), batch_size=batch_size)
emotion_labels = list(fine_gen.class_indices.keys())

# x_resampled, y_resampled = balance_dataset(train_gen, emotion_labels, img_size, batch_size)
frozen_x, frozen_y, _ = balance_dataset(fr_train, emotion_labels, img_size=(224, 224), batch_size=batch_size)
fine_x, fine_y, fine_weight = balance_dataset(fine_gen, emotion_labels, img_size=(224, 224), batch_size=batch_size)

x_train_img, x_val_img, x_train_au, x_val_au, y_train, y_val, label_map, au_data, image_data, weight = align_data_au(train_data_dir, train_au_output_path, img_size)
fr_x_tr, fr_x_val, fr_y_tr, fr_y_val, fr_lb_map, _, fr_weight = align_data(train_data_dir, img_size=(224, 224))

augmentor = AugmentWithAU(x_train_img, x_train_au, y_train, cust_batch, augment)
gen_inputs = augmentor

stp = math.ceil(len(x_train_img) / cust_batch)

#Load the trained model to start image directory or realtime emotion prediction
if os.path.exists(model_path):
    print(f"Loading the trained {model_path}")
    model = load_model(model_path)

else:
    # Build and Train the models
    # print("Building the custom model...")
    # custom_model = custom_model(img_size + (3,), (au_data.shape[1],), len(label_map))
    # print("Training the custom model...")
    # custom_history = train_custom(
    #     model=custom_model, train_inputs=gen_inputs, val_inputs=[x_val_img, x_val_au],
    #     y_val=y_val, epochs=50, save_path=f"models/custom/custom-{timestamp}.h5", steps=stp, #class_weight=class_weight
    # )
    # print("Evaluating the model...")
    # eval_model(custom_model, [x_val_img, x_val_au], y_val)
    # plot_training(custom_history, "custom model", output=f"metrics/custom/custom_metrics-{m_timestamp}.png")

    print("Building the frozen mobilenetv2 model")
    frozen_model = frozen_mobilenetv2_model(input_shape_image=(224, 224, 3), emotion_labels=emotion_labels)
    print("Training the frozen model...")
    frozen_history = train_frozen(
        frozen_model, frozen_x, frozen_y, fine_x, fine_y, batch_size, epochs=50, class_weights=fr_weight,
        save_path=f"models/frozen/frozen-{timestamp}.h5"
    )
    print("Evaluating the model...")
    eval_model(frozen_model, fine_x, fine_y)
    plot_training(frozen_history, "frozen model", output=f"metrics/frozen/frozen_metrics-{m_timestamp}.png")
    plot_matrix(frozen_model, fine_x, fine_y,
                out=f"metrics/frozen/conf_matrix-frozen-{m_timestamp}.png", emotion_labels=emotion_labels)

    unfrozen_layers = [20, 50, 100]
    learning_rates = [1e-5, 5e-6, 1e-6]
    batch = [32, 32, 32]
    epc = [25, 30, 35]

    for i, unfrozen_layers in enumerate(unfrozen_layers):
        print(f"Stage {i + 1}: Unfreezing {unfrozen_layers} layers")
        tuned_model = finetuned_mobilenetv2_model(frozen_model, unfrozen_layers, lrn_rate=learning_rates[i])

        print(f"Training stage {i + 1} mobilenetv2 model...")
        tuned_history = finetune_mobilenet(
            tuned_model, fr_x_tr, fr_y_tr, fine_x, fine_y, batch_size=batch[i], epochs=epc[i], class_weights=fr_weight,
            save_path=f"models/tuned/stage_{i + 1}-{timestamp}.h5")

        print(f"Evaluating the stage {i + 1} model...")
        eval_model(tuned_model, fine_x, fine_y)
        plot_training(tuned_history, f"stage {i + 1} model",
                      output=f"metrics/tuned/tuned_stage_{i + 1}_metrics-{m_timestamp}.png")
        plot_matrix(tuned_model, fine_x, fine_y,
                    out=f"metrics/tuned/conf_matrix-stage_{i + 1}-{m_timestamp}.png", emotion_labels=emotion_labels)

    # print("Starting to finetune the mobilenetv2 model")
    # finetuned_stage_1 = finetuned_mobilenetv2_model(frozen_model, unfrozen_layers=20, lrn_rate=1e-4)
    # print("Finetuning the mobilenet model...")
    # fine_stage_1_history = finetune_mobilenet(
    #     finetuned_stage_1, fr_x_tr, fr_y_tr, fr_x_val, fr_y_val, batch_size=64, epochs=20, save_path=f"models/tuned/tuned_stage_1-{timestamp}.h5"
    # )
    # print("Evaluating the model...")
    # eval_model(finetuned_stage_1, fr_x_val, fr_y_val)
    # plot_training(fine_stage_1_history, "finetuned model", output=f"metrics/tuned/tuned_stage_1_metrics-{m_timestamp}.png")
    #
    # print("Finetuning again the mobilenetv2 model")
    # finetuned_stage_2 = finetuned_mobilenetv2_model(frozen_model, unfrozen_layers=50, lrn_rate=1e-5)
    # print("Finetuning the mobilenet model...")
    # fine_stage_2_history = finetune_mobilenet(
    #     finetuned_stage_2, fr_x_tr, fr_y_tr, fr_x_val, fr_y_val, batch_size=32, epochs=20,
    #     save_path=f"models/tuned/tuned_stage_2-{timestamp}.h5"
    # )
    # print("Evaluating the model...")
    # eval_model(finetuned_stage_2, fr_x_val, fr_y_val)
    # plot_training(fine_stage_2_history, "finetuned model",
    #               output=f"metrics/tuned/tuned_stage_2_metrics-{m_timestamp}.png")
    #
    # print("Finetuning again the mobilenetv2 model")
    # finetuned_stage_3 = finetuned_mobilenetv2_model(frozen_model, unfrozen_layers=50, lrn_rate=1e-5)
    # print("Finetuning the mobilenet model...")
    # fine_stage_2_history = finetune_mobilenet(
    #     finetuned_stage_2, fr_x_tr, fr_y_tr, fr_x_val, fr_y_val, batch_size=32, epochs=20,
    #     save_path=f"models/tuned/tuned_stage_2-{timestamp}.h5"
    # )
    # print("Evaluating the model...")
    # eval_model(finetuned_stage_2, fr_x_val, fr_y_val)
    # plot_training(fine_stage_2_history, "finetuned model",
    #               output=f"metrics/tuned/tuned_stage_2_metrics-{m_timestamp}.png")
    #
    # print("Building the final cnn model")
    # final_model = final_cnn_model(custom_model, finetuned_stage_1, len(emotion_labels))
    # print("Training the final model...")
    # No training concept yet...
