import numpy as np

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

def draw_landmarks_on_image(rgb_image, detection_result):
  annotated_image = np.copy(rgb_image)

  if detection_result is None:
    return annotated_image

  # Loop through the detected hands to visualize.
  hand_landmarks_list = detection_result.hand_landmarks
  if not hand_landmarks_list:
    return annotated_image

  # Draw the hand landmarks.
  for hand_landmarks in hand_landmarks_list:
    # Convert Landmarks to MediaPipe format for drawing
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    # Draw landmarks and connections between them
    mp.solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      mp.solutions.hands.HAND_CONNECTIONS,
      mp.solutions.drawing_utils.DrawingSpec(
        color=(255, 0, 0),  # landmarks colour
        thickness=2,
        circle_radius=3
      ),
      mp.solutions.drawing_utils.DrawingSpec(
        color=(0, 255, 0),  # connections colour
        thickness=2)
      )

  return annotated_image

def extract_xy_landmarks(result):
  """
  Extract x and y coordinates for each hand
  Returns:
    (left_xy, right_xy)
    Each is a list of 21 [x,y] coordinates
  From hand landmarks I get 21 landmarks for each hand (so 42)
  And from handedness I get the info about right or left
  """
  if result is None or not result.hand_landmarks:
    return None, None

  right_xy, left_xy = None, None

  for i, hand_landmarks in enumerate(result.hand_landmarks):
    # Left or Right
    hand_label = None
    if result.handedness and len(result.handedness) > i:
      hand_label = result.handedness[i][0].category_name

    xy = [[landmark.x, landmark.y] for landmark in hand_landmarks]

    if hand_label == 'Right':
      right_xy = xy
    elif hand_label == 'Left':
      left_xy = xy
    else:
      pass

  return right_xy, left_xy
