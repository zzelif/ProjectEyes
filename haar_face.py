import cv2
import time
import os


def is_face_centered(face, frame_width, frame_height):
    """
    Checks if a face is centered in the frame.

    Args:
        face: Bounding box of the detected face.
        frame_width: Width of the frame.
        frame_height: Height of the frame.

    Returns:
        bool: True if the face is centered, False otherwise.
    """
    x, y, w, h = face
    center_x = frame_width // 2
    center_y = frame_height // 2

    tolerance = 0.2

    center_region_x1 = center_x - (w // 2) - int(tolerance * frame_width)
    center_region_y1 = center_y - (h // 2) - int(tolerance * frame_height)
    center_region_x2 = center_x + (w // 2) + int(tolerance * frame_width)
    center_region_y2 = center_y + (h // 2) + int(tolerance * frame_height)

    return (
        x >= center_region_x1
        and y >= center_region_y1
        and x + w <= center_region_x2
        and y + h <= center_region_y2
    )


class FaceDetectionPipeline:
    def __init__(self, classifier_path, output_dir, timer_threshold, rectangle_color, rectangle_thickness):
        self.face_cascade = cv2.CascadeClassifier(classifier_path)
        self.output_dir = output_dir
        self.timer_threshold = timer_threshold
        self.rectangle_color = rectangle_color
        self.rectangle_thickness = rectangle_thickness

        self.face_detected = False
        self.centered_face = False
        self.start_time = None
        self.frame_count = 0

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

    def detect_faces(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        orig_frame = frame.copy()
        faces = self.face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces, orig_frame

    def process_frame(self, frame):
        faces, orig_frame = self.detect_faces(frame)

        self.centered_face = any(
            is_face_centered((x, y, w, h), frame.shape[1], frame.shape[0])
            for (x, y, w, h) in faces
        )

        if not self.face_detected and self.centered_face:
            self.face_detected = True
            self.start_time = time.time()
            print(f"Face detected: {self.face_detected}, Timer started at: {self.start_time}")

        if self.face_detected and ((time.time() - self.start_time) <= self.timer_threshold):
            print("Timer condition satisfied. Drawing rectangle.")
            for (x, y, w, h) in faces:
                cv2.rectangle(
                    frame, (x, y), (x + w, y + h), self.rectangle_color, self.rectangle_thickness
                )

            # Remove cv2.imwrite and instead push `orig_frame` directly to processed_queue
            return orig_frame

        elif not self.centered_face:
            self.face_detected = False
            print("No detected face in the center. Resetting the timer.")

        # Display the frame
        cv2.imshow("Realtime Detection", frame)

        return True


