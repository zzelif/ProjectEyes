from concurrent.futures import ThreadPoolExecutor
from run_openface import _realtime_openface
from haar_face import FaceDetectionPipeline
from data_preprocessing import align_data_au
from tensorflow.keras.models import load_model
from datetime import datetime
from threading import Lock
import numpy as np
import queue
import cv2
import os

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
save_lock = Lock()

#Paths anc Constants
pr_model = load_model("models/custom/custom-2025-01-06_12-22.h5")
openface_path = "OpenFace/FeatureExtraction.exe"
au_extract_path = f"au features/{timestamp}"
out = f"processed/{timestamp}"
if not os.path.exists(out):
    os.makedirs(out)


emotion_map = {0: "angry", 1: "happy", 2: "neutral", 3: "sad", 4: "surprised"}
emotion_map = {key: value.capitalize() for key, value in emotion_map.items()}
emotion_tally = {emotion: 0 for emotion in emotion_map.values()}

pipeline = FaceDetectionPipeline(
    classifier_path=".venv\lib\site-packages\cv2\data\haarcascade_frontalface_default.xml",
    output_dir=out,
    timer_threshold=0.5,
    rectangle_color=(0, 255, 0),
    rectangle_thickness=3
)

f_name = ""
state = False
frame_queue = queue.Queue(maxsize=10)
processed_queue = queue.Queue()

def webcam_capture():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        exit()
    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))
        if not ret:
            break

        if frame_queue.full():
            continue

        frame_queue.put(frame)
        cv2.imshow("Realtime Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    frame_queue.put(None)

def process_frame():
    while True:
        global state
        global f_name

        frame = frame_queue.get()
        if frame is None:
            processed_queue.put(None)
            break

        with save_lock:
            f_name, state = pipeline.process_frame(frame)

        if state:
            processed_queue.put(out)

def extract_features_and_predict():
    while True:
        global state
        global f_name

        directory = processed_queue.get()
        print(f"length of directory: {len(directory)}")
        if directory is None:
            break

        if state and os.path.exists(f_name):
            print("Running openface on the frames")
            _realtime_openface(directory, au_extract_path, openface_path)
            au_data, image_data = align_data_au(out, au_extract_path, img_size=(48,48))
            print(f"Shapes of the data: {au_data.shape}, {image_data.shape}")
            preds = pr_model.predict([image_data, au_data])
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





with ThreadPoolExecutor() as executor:
    executor.submit(webcam_capture)
    executor.submit(process_frame)
    executor.submit(extract_features_and_predict)