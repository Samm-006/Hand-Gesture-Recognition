# Imports
import io
import os
import av
import cv2
import time
import threading
import numpy as np

# TTS
from gtts import gTTS

import mediapipe as mp
import tensorflow as tf

# Streamlit for User interface
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

from utils.preprocessing import preprocess_landmarks_xy
from utils.drawing_landmarks import draw_landmarks_on_image, extract_xy_landmarks

# Page config (title + layout)
st.set_page_config(
    page_title="Emergency Sign Translator",
    layout="wide",
)
# WebRTC configuration to connect to webcam stream
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
# App Styling (visual layout + appearance)
st.markdown(
    """
    <style>
      .stApp {
        background: radial-gradient(1200px 600px at 20% 0%, #2a2f3a 0%, #0f1116 60%, #0b0c10 100%);
        color: #EDEFF3;
      }
      .block-container {
        max-width: 100% !important;
        padding-left: 3rem;
        padding-right: 3rem;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
      }

      section[data-testid="stSidebar"] {
        background: rgba(18, 21, 28, 0.9);
        border-right: 1px solid rgba(255,255,255,0.06);
      }

      .section-header {
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
      }

      .card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        backdrop-filter: blur(8px);
        margin-bottom: 1.25rem; 
      }

      .muted { color: rgba(237,239,243,0.75); font-size: 0.95rem; }

      .pill {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(255, 80, 80, 0.14);
        border: 1px solid rgba(255, 80, 80, 0.35);
        color: #ffd0d0;
        font-size: 0.85rem;
      }

      .big {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.15;
        margin-top: 6px;
      }

      .stButton button {
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.14) !important;
        background: rgba(255,255,255,0.06) !important;
        width: 100%;
      }
      
      .history-item {
       padding: 6px 10px;
       border-radius: 8px;
       background: rgba(255,255,255,0.06);
       margin-bottom: 4px;
       font-size: 0.9rem;
       color: #EDEFF3;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# Emergency phrase -> map words to full phrases.
# For demonstration purposes
EMERGENCY_PHRASES = {
    "help": "I need help",
    "Police": "Call the police",
    "fireman": "Call the fireman",
    "Ambulance": "Call an ambulance",
    "Fire": "There is a fire",
    "hurt": "I am hurt",
    "stop": "STOP",
    "short of breath": "I am short of breath",
    "emergency": "It's an emergency",
    "Accident": "There has been an accident",
    "Doctor": "a doctor",
    "Yes": "Yes",
    "No": "No"
}

# Gesture List for user interface
GESTURE_REFERENCE = [
    "I", "Need", "Accident", "Fire", "Yes",
    "No", "Ambulance", "Doctor", "Police", "Fireman",
    "Hurt", "Emergency", "Short of breath", "Help", "Stop"
]

# MediaPipe Hand Landmarker Configuration
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Load labeled csv from dataset_labels.csv
def load_labels(labels_csv_path):
    labels = []
    with open(labels_csv_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    return labels

@st.cache_resource
def load_label_list(labels_path: str):
    return load_labels(labels_path)

# Map = Predicted label into emergency phrase
def label_to_phrase(label: str) -> str:
    if not label:
        return ""
    return EMERGENCY_PHRASES.get(label, label)

# Load trained Tensorflow gesture classification model
@st.cache_resource
def load_tf_model(model_path: str):
    return tf.keras.models.load_model(model_path)

# Store MediaPipe model path
@st.cache_resource
def load_mp_landmarker(task_path: str):
    return task_path

# Convert a phrase to MP3 audio using Google TTS
def tts_bytes(text: str) -> bytes:
    """
    Convert text -> MP3 bytes using gTTS.
    Streamlit will play it using st.audio().
    """
    tts = gTTS(text=text, lang="en", slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    return buf.getvalue()


# VideoProcessor -> Processes each webcam frames:
# 1- Detect hands using MediaPipe
# 2- Extract hand landmarks
# 3- Preprocess
# 4- Predict the gestures using the trained TensorFlow model

class VideoProcessor:
    def __init__(self, task_path, model, labels, min_conf=0.60, smooth_window=8, max_low_frames=10):
        self.model = model
        self.labels = labels
        self.min_conf = min_conf

        # Stores recent prediction for smoothing
        self.smooth_window = int(smooth_window)
        self.recent_preds = []

        # Track low confidence frames
        self.max_low_frames = int(max_low_frames)
        self.low_conf_frames = 0

        self.lock = threading.Lock()

        # Latest predication results
        self.latest_phrase = ""
        self.latest_label = ""
        self.latest_conf = 0.0
        self.latest_status = "No hand detected"

        self.latest_conf_raw = 0.0     # Track confidence for progress bar
        self.latest_has_hand = False   # Track if hand is detected or not
        self._result_lock = threading.Lock()
        self._latest_result = None


        def _callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            with self._result_lock:
                self._latest_result = result

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=task_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=_callback,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def recv(self, frame):
        frame_bgr = frame.to_ndarray(format="bgr24")
        frame_bgr = cv2.flip(frame_bgr, 1)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Convert frame to MediaPipe format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(time.time() * 1000)

        # async
        self.landmarker.detect_async(mp_image, timestamp_ms)

        # grab latest landmarks
        with self._result_lock:
            result = self._latest_result

        # Extract both hands
        right, left = extract_xy_landmarks(result)

        has_hand = not (right is None and left is None)

        # Preprocess
        if right is not None:
            features_right = preprocess_landmarks_xy(right)
        else:
            features_right = [0.0] * 42

        if left is not None:
            features_left = preprocess_landmarks_xy(left)
        else:
            features_left = [0.0] * 42

        features = features_right + features_left

        pred_text = ""
        phrase = ""
        status = "No hand detected"
        shown_conf = 0.0
        raw_conf = 0.0

        # To check if there is hands
        if has_hand:
            x = np.array(features, dtype=np.float32).reshape(1, 84)
            probs = self.model.predict(x, verbose=0)[0]
            pred_id = int(np.argmax(probs))
            confidence = float(np.max(probs))
            raw_conf = confidence

            # if confidence is >0.6 then give translation of SL of the given hand gesture
            if confidence >= self.min_conf:
                status = "Hand detected"
                shown_conf = confidence
                self.low_conf_frames = 0

                # smoothing (keep last N confident predictions and choose most common)
                self.recent_preds.append(pred_id)
                if len(self.recent_preds) > self.smooth_window:
                    self.recent_preds.pop(0)

                pred_id_smooth = max(set(self.recent_preds), key=self.recent_preds.count)

                if 0 <= pred_id_smooth < len(self.labels):
                    pred_text = self.labels[pred_id_smooth]
                else:
                    pred_text = str(pred_id_smooth)

                phrase = label_to_phrase(pred_text)
            else:
                # Low confidence: don't show label/phrase (<60%)
                status = f"Low confidence (<{self.min_conf:.2f})"
                self.low_conf_frames += 1
                if self.low_conf_frames >= self.max_low_frames:
                    self.recent_preds.clear()
        else:
            # no hands so nothing clear
            self.low_conf_frames = 0
            self.recent_preds.clear()

        with self.lock:
            self.latest_label = pred_text
            self.latest_phrase = phrase
            self.latest_conf = shown_conf
            self.latest_status = status
            self.latest_conf_raw = raw_conf
            self.latest_has_hand = has_hand

        # Draw
        annotated_rgb = draw_landmarks_on_image(frame_rgb, result)
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")

# App
def main():

    # App title and description displayed at the top of the page
    st.markdown(
        """
        <div class="card">
          <span class="pill">🚨 Emergency Translator</span>
          <h1 style="margin-top:10px;">Emergency Sign Language Interpreter</h1>
          <div class="muted">Point your camera. Sign an emergency gesture. The app shows a clear phrase</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")

    left_col, right_col = st.columns([1.55, 1.0], gap="large")

    # Define paths
    base_dir = os.path.dirname(__file__)
    task_path = os.path.join(base_dir, "model", "hand_landmarker.task")
    keras_model_path = os.path.join(base_dir, "model", "gesture_classifier.keras")
    labels_path = os.path.join(base_dir, "data", "dataset_labels.csv")

    # Load trained TensorFlow model and label files
    model = load_tf_model(keras_model_path)
    labels = load_label_list(labels_path)

    # Store last phrase so your tts button always has something to speak
    if "last_phrase" not in st.session_state:
        st.session_state.last_phrase = ""

    if "phrase_history" not in st.session_state:
        st.session_state.phrase_history = []

    # Left Column-> the camera
    with left_col:
        st.markdown(
            """
            <div class="card section-header">
              <h3>Live Camera</h3>
              <div class="muted">Press "Start" to see translation here with a real-time camera capturing interpreter. Keep your hands visible and steady for a second.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Start webcam streaming using WebRTC
        webrtc_ctx = webrtc_streamer(
            key="WYH",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=lambda: VideoProcessor(
                task_path=task_path,
                model=model,
                labels=labels,
                min_conf=0.60,
                smooth_window=8,
                max_low_frames=10
            ),
            async_processing=True,
        )

        # Gestures classes list
        with st.expander("View all 15 supported gestures"):
            cols = st.columns(3)
            for i, word in enumerate(GESTURE_REFERENCE):
                cols[i % 3].markdown(
                    f"<div class='history-item'>{word}</div>",
                    unsafe_allow_html=True
                )

    # Right Column -> Translation output and audio
    with right_col:
        st.markdown(
            """
            <div class="card section-header">
              <h3>Translation</h3>
              <div class="muted">This is what you show or read out loud.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        panel = st.empty()
        audio_placeholder = st.empty()
        speak_placeholder = st.empty()
        conf_label_placeholder = st.empty()
        conf_bar_placeholder = st.empty()
        history_panel = st.empty()
        clear_placeholder = st.empty()

        if not (webrtc_ctx and webrtc_ctx.state.playing):
            panel.markdown(
                """
                <div class="card">
                  <div class="muted">Press "Start" to see the translation here.</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        if webrtc_ctx and  webrtc_ctx.state.playing:

            with speak_placeholder:
                speak_clicked = st.button(
                    "Translator Audio",
                    key="speak_btn",
                    help="Press to get audio on the last detect phrase",
                    use_container_width = True
                )

            with clear_placeholder:
                if st.button("Clear history", key="clear_btn", use_container_width=True):
                    st.session_state.phrase_history = []
                    st.session_state.last_phrase = ""

            while webrtc_ctx.state.playing:
                # Get latest prediction results from VideoProcessor
                vp = webrtc_ctx.video_processor

                if vp is not None:
                    with vp.lock:
                        phrase = vp.latest_phrase
                        label = vp.latest_label
                        conf = vp.latest_conf
                        status = vp.latest_status
                        conf_raw = vp.latest_conf_raw
                        has_hand = vp.latest_has_hand
                else:
                    phrase, label, conf, conf_raw, has_hand, status = "", "", 0.0, 0.0, False, "Starting..."

                # Save last valid phrase (so Speak can work even if next frame becomes blank)
                if phrase:
                    st.session_state.last_phrase = phrase
                    # Append phrase history
                    history = st.session_state.phrase_history
                    if not history or history[-1] != phrase:
                        history.append(phrase)
                        if len(history) > 5:
                            history.pop(0)
                        st.session_state.phrase_history = history

                phrase_show = phrase if phrase else "—"
                label_show = label if label else "—"

                # Display prediction results
                panel.markdown(
                    f"""
                    <div class="card">
                      <div class="muted">Status</div>
                      <div style="font-size: 1.00rem; font-weight: 650;">{status}</div>

                      <div style="height: 12px;"></div>

                      <div class="muted">Emergency phrase</div>
                      <div class="big">{phrase_show}</div>

                      <div style="height: 12px;"></div>

                      <div class="muted">Detected label</div>
                      <div style="font-size: 1.00rem; font-weight: 650;">{label_show}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Show for all confidence values including detected cong values under 60
                if has_hand and conf_raw > 0:
                    conf_label_placeholder.markdown(
                        f"<div class='muted' style='margin-bottom:4px;'>Confidence: {conf_raw*100:.0f}%</div>",
                        unsafe_allow_html=True
                    )
                    conf_bar_placeholder.progress(min(conf_raw, 1.0))
                else:
                    conf_label_placeholder.empty()
                    conf_bar_placeholder.empty()

                if st.session_state.phrase_history:
                    history_items = "".join(
                        f"<div class='history-item'> {p}</div>"
                        for p in reversed(st.session_state.phrase_history)
                    )
                    history_panel.markdown(
                        f"""
                        <div class="card">
                          <div class="muted">Recent phrase</div>
                          <div style="margin-top:8px;">{history_items}</div>
                          <div style="margin-top:10px; padding-top:8px; border-top:1px solid rgba(255,255,255,0.06);">
                          <span class="muted" style="font-size:0.82rem;">Press "Clear history" below to reset</span>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # If Speak clicked, play LAST saved phrase (TTS)
                if speak_clicked:
                    last = st.session_state.last_phrase
                    already_spoken = st.session_state.get("_last_spoken", "")
                    if last and last != already_spoken:
                        st.session_state._last_spoken = last
                        try:
                            mp3 = tts_bytes(last)
                            audio_placeholder.audio(mp3, format="audio/mp3")
                        except Exception as e:
                            st.error(f"TTS failed: {e}")
                    elif not last:
                        if not st.session_state.get("_warned", False):
                            st.warning("No gesture detected yet to speak.")
                            st.session_state._warned = True
                else:
                    st.session_state._warned = False
                time.sleep(0.1)

        st.write("")
        st.markdown(
            """
            <div class="card section-header">
              <h3>Quick actions</h3>
              <ul>
                <li>Call emergency services (UK: 999)</li>
                <li>Or visit: <a href="https://999bsl.co.uk" target="_blank">https://999bsl.co.uk</a></li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
