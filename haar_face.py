import cv2
import time
import os

face_detected = False
centered_face = False
start_time = None
frame_count = 0

#
# Detect if face is visible
def detect_face_haar(frame):
    """
    Detects a face in the given frame.

    Args:
        frame: The input video frame.
    Returns:
        face: Cropped face region if detected, else None.
    """
    classifier_path = ".venv\lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(classifier_path)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    orig_frame = frame.copy()

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    return faces, orig_frame

def is_face_centered(face, frame_width, frame_height):
    """
    Detects a face in the given frame.

    Args:
        face: The input video frame.
        frame_width: frame.shape[1]
        frame_height: frame.shape[0]

    Returns:
        face: boolean value whether face is centered or not
    """
    global centered_face

    x, y, w, h = face
    center_x = frame_width // 2
    center_y = frame_height // 2

    tolerance = 0.2

    center_region_x1 = center_x - (w // 2) - int(tolerance * frame_width)
    center_region_y1 = center_y - (h // 2) - int(tolerance * frame_height)
    center_region_x2 = center_x + (w // 2) + int(tolerance * frame_width)
    center_region_y2 = center_y + (h // 2) + int(tolerance * frame_height)

    # Check if the detected face lies inside the defined center region (with tolerance)
    if x >= center_region_x1 and y >= center_region_y1 and x + w <= center_region_x2 and y + h <= center_region_y2:
        return True

    return False

def haar_features(frame, faces, orig_frame, out):
    """
    Detects a face in the given frame.

    Args:
        frame: The input video frame.
        faces: Number of faces detected
        orig_frame: The video frame without the rectangle
        out: Output directory

    Returns:

    """
    global face_detected
    global start_time
    global frame_count
    global centered_face

    for (x, y, w, h) in faces:
        if is_face_centered((x, y, w, h), frame.shape[1], frame.shape[0]):
            centered_face = True
            break

    if not face_detected and centered_face:
        face_detected = True
        start_time = time.time()
        print(f"Face detected: {face_detected}, Timer started at: {start_time}")

    if face_detected and ((time.time() - start_time) < 3):
        print("Timer condition satisfied. Drawing rectangle.")
        for (x, y, w, h) in faces:
            print(f"Drawing rectangle at: x={x}, y={y}, w={w}, h={h}")
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 5)

        frame_filename = os.path.join(out, f'frame_{frame_count}.jpg')
        cv2.imwrite(frame_filename, orig_frame)
        frame_count += 1

    elif not centered_face:
        face_detected = False
        print("No detected face in the center. Resetting the timer.")

    # Display the frame
    cv2.imshow("Realtime Detection", frame)