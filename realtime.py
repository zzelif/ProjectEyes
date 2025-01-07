from run_openface import _realtime_openface
from haar_face import FaceDetectionPipeline
from data_preprocessing import align_data_au
from datetime import datetime
import os
import cv2

#Paths and Constants
emotion_map = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}
specific_aus = {
    "Angry": [4, 9, 5, 17],
    "Happy": [6, 7, 10, 12, 14, 20],
    "Neutral": [2, 5],
    "Sad": [1, 4, 6, 7, 9, 12, 15, 17, 20],
    "Surprised": [1, 2, 5, 25, 26]
}

img_size = (48, 48)
timestamp = datetime.now().strftime("%Y-%m-%d_%H")
model_path = "models/model.h5"
openface_path = "OpenFace/FeatureExtraction.exe"
au_extract_path = f"au features/{timestamp}"
# mean_au = f"utils/train_consolidated_au_{timestamp}.csv"
# mean_rt_au_path = f"utils/rt_consolidated_au_{timestamp}.csv"

out = f"processed/{timestamp}"
if not os.path.exists(out):
    os.makedirs(out)

pipeline = FaceDetectionPipeline(
    classifier_path=".venv\lib\site-packages\cv2\data\haarcascade_frontalface_default.xml",
    output_dir=out,
    timer_threshold=0.5,
    rectangle_color=(0, 255, 0),
    rectangle_thickness=3
)

# #Initialize the model
# model = load_model(model_path)

#Start the video feed
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()
print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    f_name, state = pipeline.process_frame(frame)

    _realtime_openface(out, au_extract_path, openface_path)

    # au_data, image_data = align_data_au(out, au_extract_path, img_size)

    # preds = model.predict([au_data, image_data])
    #
    # fin_score, predict_emotion = refine_predictions(mean_rt_au_path, preds, specific_aus, emotion_map)

    cv2.imshow("Realtime Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break

cap.release()
cv2.destroyAllWindows()
