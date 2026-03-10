import numpy as np

def preprocess_landmarks_xy(xy_landmarks):
  """
  Preprocess had landmark coordinates before saving it to dataset

  1) Translation invariance: Convert landmarks to be relative to wrist (landmark 0)
  2) Flatten: Convert [[x1,y1],[x2,y2],...] -> [x1,y1,x2,y2, ...]
  3) Normalisation: Scale values by max abs value so hand size and camera distance dont matter(scale invariance)
  """
  pts = np.array(xy_landmarks, dtype=np.float32)

  # 1 relative to wrist
  pts = pts - pts[0]
  # 2 flatten
  flat = pts.flatten()
  # 3 normalise
  max_val = np.max(np.abs(flat))
  if max_val > 0:
    flat = flat / max_val
  return flat.tolist()
