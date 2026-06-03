"""
01_mediapipe_landmarks.py
Setup MediaPipe Hand Landmarker detection using OpenCV.
It captures webcam frames, detect hands, extract 21 landmarks,
and visualize them.
"""
# Imports
import os
import cv2
import time
import mediapipe as mp

# Function that draws hand landmarks on the frame
from utils.drawing_landmarks import draw_landmarks_on_image

# MediaPipe Hand Landmarker Configuration
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Store most recent detection result
latest_result = None

def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
  # Callback function that stores the most recent
  global latest_result
  latest_result = result
  print('hand landmarker result: {}'.format(result))

def main():
  # Model
  BASE_DIR = os.path.dirname(__file__)
  model_path = os.path.join(BASE_DIR, "model", "hand_landmarker.task")

  options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=print_result
  )

  # Open webcam
  cap = cv2.VideoCapture(0)

  # Create Mediapipe hand landmarker
  with HandLandmarker.create_from_options(options) as landmarker:

    while True:

      # Read frame from webcam
      ok, frame_bgr = cap.read()
      if not ok:
        break

      # Flip fram horizontally from mirror view
      frame_bgr = cv2.flip(frame_bgr, 1)

      # Convert BGR (OpenCV format) -> RBG (MediaPipe format)
      frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

      # Create MediaPipe image object
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
      timestamp_ms = int(time.time() * 1000)

      # async callback (result comes to print_result)
      landmarker.detect_async(mp_image, timestamp_ms)

      # Draw detected landmarks on the frame
      annotated_rgb = draw_landmarks_on_image(frame_rgb, latest_result)

      # Covert RGB -> to BGR for OpenCV
      annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

      # Show the video window
      cv2.imshow("MediaPipe Landmarker", annotated_bgr)

      # Pres ESC to exit
      if cv2.waitKey(1) & 0xFF == 27:
        break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
    main()