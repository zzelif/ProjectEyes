import cv2
import os
import threading
import subprocess
import tempfile
import time
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

# Path to Haar Cascade and OpenFace FeatureExtraction
CASCADE_PATH = ".venv\lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
FEATURE_EXTRACTION_PATH = "OpenFace/FeatureExtraction.exe"
OUTPUT_DIR = f"processed/{timestamp}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize face detection with Haar cascades
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def extract_features(frame, output_dir, feature_extraction_path):
    """
    Saves a frame as an image and runs OpenFace FeatureExtraction on it.
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_image_path = temp_file.name
        cv2.imwrite(temp_image_path, frame)  # Save the frame as a temp image

    # Run OpenFace on the saved image
    subprocess.run([feature_extraction_path, "-f", temp_image_path, "-aus", "-out_dir", output_dir])
    os.remove(temp_image_path)  # Clean up temporary file

def process_frame(frame, face_detected, last_detection_time):
    """
    Process a single frame: detect faces, visualize them, and trigger feature extraction.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Draw rectangle around detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract features if not recently detected
        current_time = time.time()
        if current_time - last_detection_time > 5 and not face_detected.is_alive():
            face_detected = threading.Thread(target=extract_features, args=(frame, OUTPUT_DIR, FEATURE_EXTRACTION_PATH))
            face_detected.start()
            last_detection_time = current_time

    return frame, face_detected, last_detection_time


def real_time_detection():
    """
    Real-time face detection and feature extraction loop.
    """
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    print("Press 'q' to exit.")
    face_detected = threading.Thread()
    last_detection_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Process the frame
        frame, face_detected, last_detection_time = process_frame(frame, face_detected, last_detection_time)

        # Display the frame
        cv2.imshow("Real-Time Detection", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Start real-time detection
real_time_detection()


# import cv2
# import os
# import threading
# import subprocess
# import time
#
# # Paths
# CASCADE_PATH = ".venv\lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
# FEATURE_EXTRACTION_PATH = "OpenFace/FeatureExtraction.exe"
# OUTPUT_DIR = "processed"
# os.makedirs(OUTPUT_DIR, exist_ok=True)
#
# # Initialize Haar cascade for face detection
# face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
#
# def run_openface_for_duration(duration, output_dir, feature_extraction_path):
#     """
#     Runs OpenFace for a specified duration on the webcam.
#     Args:
#         duration (int): Duration in seconds for which OpenFace processes frames.
#         feature_extraction_path (str): Path to OpenFace FeatureExtraction.exe.
#         output_dir (str): Directory to save OpenFace output.
#     """
#     process = subprocess.Popen([feature_extraction_path, '-device', '0', '-out_dir', output_dir, '-aus', '-g'])
#     time.sleep(duration)
#     process.terminate()
#
# def analyze_openface_output(output_dir):
#     """
#     Analyzes OpenFace output and computes the mean of AUs.
#     Args:
#         output_dir (str): Directory containing OpenFace output files.
#     Returns:
#         dict: Mean values of extracted Action Units (AUs).
#     """
#     import pandas as pd
#     au_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".csv")]
#     if not au_files:
#         return {}
#
#     # Read all AU files and compute mean
#     combined_df = pd.concat([pd.read_csv(f) for f in au_files])
#     mean_aus = combined_df.filter(regex="AU").mean().to_dict()
#     return mean_aus
#
# def process_frame_with_haar(frame):
#     """
#     Detect faces using Haar cascades and draw rectangles around them.
#     Args:
#         frame: Input frame from OpenCV.
#     Returns:
#         bool: True if a face is detected, False otherwise.
#         frame: Processed frame with rectangles drawn around detected faces.
#     """
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
#     if len(faces) > 0:
#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         return True, frame
#     return False, frame
#
# def real_time_detection_with_openface():
#     """
#     Real-time detection and feature extraction using OpenCV and OpenFace.
#     """
#     cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#     if not cap.isOpened():
#         print("Error: Unable to access the webcam.")
#         return
#
#     print("Press 'q' to exit.")
#     last_run_time = time.time()
#     face_detected = False
#     duration = 2  # OpenFace runtime in seconds
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame.")
#             break
#
#         # Process frame with Haar cascades
#         face_detected, processed_frame = process_frame_with_haar(frame)
#
#         # Trigger OpenFace if a face is detected and sufficient time has passed
#         if face_detected and time.time() - last_run_time >= 5:  # 5-second gap between OpenFace runs
#             print("Face detected! Running OpenFace...")
#             cap.release()  # Release camera to allow OpenFace access
#             threading.Thread(target=run_openface_for_duration, args=(duration, OUTPUT_DIR, FEATURE_EXTRACTION_PATH)).start()
#             last_run_time = time.time()
#             cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#
#         # Display the frame
#         cv2.imshow("Real-Time Detection", processed_frame)
#
#         # Exit on pressing 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     # After exiting the loop, analyze OpenFace output
#     cap.release()
#     cv2.destroyAllWindows()
#     print("Computing mean AUs from OpenFace output...")
#     mean_aus = analyze_openface_output(OUTPUT_DIR)
#     print("Mean AUs:", mean_aus)
#
# # Run the detection loop
# real_time_detection_with_openface()
