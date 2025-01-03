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
    # Build the model
    print("Building the model...")
