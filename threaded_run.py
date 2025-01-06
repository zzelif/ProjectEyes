from concurrent.futures import ThreadPoolExecutor
from run_openface import _realtime_openface
from haar_face import detect_face_haar, haar_features
from data_preprocessing import align_data_au
from datetime import datetime
import queue
import cv2
import os

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

#Paths anc Constants
openface_path = "OpenFace/FeatureExtraction.exe"
au_extract_path = f"au features/{timestamp}"
out = f"processed/{timestamp}"
if not os.path.exists(out):
    os.makedirs(out)

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
        frame = cv2.resize(frame, (480, 360))
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

        frame = frame_queue.get()
        if frame is None:
            processed_queue.put(None)
            break

        faces, orig_frames = detect_face_haar(frame)
        print(f"Detected faces: {faces}")
        haar_features(frame, faces, orig_frames, out)

        processed_queue.put(out)

def extract_features_and_predict():
    while True:

        directory = processed_queue.get()
        print(f"length of directory: {len(directory)}")
        if directory is None:
            break

        _realtime_openface(directory, au_extract_path, openface_path)
        _, _ = align_data_au(out, au_extract_path, img_size=(48,48))


with ThreadPoolExecutor() as executor:
    executor.submit(webcam_capture)
    executor.submit(process_frame)
    executor.submit(extract_features_and_predict)