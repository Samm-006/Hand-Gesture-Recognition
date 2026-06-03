"""
02_collect_dataset.py
The script collects hand gesture data using MediaPipe.
First it detects hand landmarks using webcam
then saves processed landmarks data to csv file.
"""
# Imports
import os
import cv2
import csv
import time
import mediapipe as mp

# Functions for preprocessing and drawing landmarks
from utils.preprocessing import preprocess_landmarks_xy
from utils.drawing_landmarks import draw_landmarks_on_image
from utils.drawing_landmarks import extract_xy_landmarks

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

def append_row(csv_path, label, features):
  # Opens CSV
    with open(csv_path, "a", newline="") as f:
      # Writes one training dataset
        writer = csv.writer(f)
        writer.writerow([label, *features])

# Handles keyboard input for labels and data collection mode
def select_mode(key, mode):
  number = -1          # Default: no label selected

  # ASCII codes for labels
  if 48 <= key <= 57:  # 0 - 9
    number = key - 48

  label_keys = {       # 10 - 14
    ord('a'): 10,
    ord('s'): 11,
    ord('d'): 12,
    ord('f'): 13,
    ord('g'): 14
  }
  if key in label_keys:
    number = label_keys[key]

  # Mode control
  if key == 110:       # n - Stop saving (Turn off data collection)
    mode = 0
  if key == 107:       # k - Start saving (Turn on data collection)
    mode = 1
  return number, mode


def main():
    """
    Open the camera → detect both hand → collect landmarks for a specific gesture → save 21 (x,y) coordinates data

    """
    # To capture 250 frames when k is pressed then turn off data collection
    count_frames = 250

    # Save CSV File
    os.makedirs("data", exist_ok=True)
    csv_path = "data/dataset.csv"

    # Load MediaPipe hand landmark model
    BASE_DIR = os.path.dirname(__file__)
    model_path = os.path.join(BASE_DIR, "model", "hand_landmarker.task")

    # MediaPipe Hands
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=print_result
    )

    # Open Webcam
    cap = cv2.VideoCapture(0)

    # Logging state
    saving = False
    label = -1
    mode = 0

    # Counter state
    saved_count = 0

    with HandLandmarker.create_from_options(options) as landmarker:
      while True:
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

        # Handle keyboard input
        key = cv2.waitKey(10)

        if key == 27:  # ESC
          break

        prev_mode = mode
        number, mode = select_mode(key, mode)

        # If user start saving (k) -> reset to 0 for next capture set.
        if prev_mode == 0 and mode == 1:
            saved_count =0

        # Update label when user presses 0–14
        if number != -1:
          label = number

        # Extract both hands
        right, left = extract_xy_landmarks(latest_result)

        # Preprocess
        saving = (mode == 1)
        if right is not None:
          features_right = preprocess_landmarks_xy(right)
        else:
          features_right = [0.0] * 42

        if left is not None:
          features_left = preprocess_landmarks_xy(left)
        else:
          features_left = [0.0] * 42
        # Save as one training dataset (label + the preprocessed landmarks) to csv file
        if saving and (0 <= label <= 14):
            features = features_right + features_left
            append_row(csv_path, label, features)
            saved_count += 1
            # Stop automatically after 250 frames
            if saved_count >= count_frames:
                mode = 0
                saving = False

        # Draw detected landmarks on the frame
        annotated_rgb = draw_landmarks_on_image(frame_rgb, latest_result)
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

        # Display recording status
        status1 = f"Saving: {'YES' if saving else 'NO'}  (k/n) {saved_count}/{count_frames}"
        status2 = f"Label: {label}  (0-14)"
        cv2.putText(annotated_bgr, status1, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated_bgr, status2, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("02_collect_dataset", annotated_bgr)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
