# Emergency Sign Language Interpreter

> A real-time hand gesture recognition system that translates emergency sign language into spoken phrases — built for Deaf and Hard-of-Hearing (DHH) individuals.


[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://emergency-gesture-recognition.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.2-FF6F00?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.21-0097A7?style=for-the-badge)](https://ai.google.dev/edge/mediapipe)

---

## Live Demo

**Try it now →** [https://emergency-gesture-recognition.streamlit.app/](https://emergency-gesture-recognition.streamlit.app/)

1. Click **Start** to enable your webcam
2. Perform one of the 15 emergency gestures
3. The app displays the translated phrase in real time
4. Press **Translator Audio** to hear the phrase spoken aloud

---

## Overview

This project bridges the communication gap between DHH individuals and hearing emergency responders. It recognises **15 emergency sign language gestures** and maps them to full emergency phrases such as:

| Gesture | Emergency Phrase |
|---------|-----------------|
| `help` | I need help |
| `Ambulance` | Call an ambulance |
| `Fire` | There is a fire |
| `hurt` | I am hurt |
| `Police` | Call the police |
| `short of breath` | I am short of breath |

**Final model: 95.46% test accuracy · Macro F1-score: 0.96**

---

## Project Structure

```
Hand-Gesture-Recognition/
│
├── 01_mediapipe_landmarks.py     # Webcam + hand landmark visualisation
├── 02_collect_dataset.py         # Gesture data collection tool
├── 03_trained_model.ipynb        # Model training notebook
├── app.py                        # Streamlit web application
├── requirements.txt              # Project dependencies
│
├── model/
│   ├── hand_landmarker.task      # MediaPipe pretrained model
│   └── gesture_classifier.keras  # Trained gesture classifier
│
├── data/
│   └── dataset_labels.csv        # Gesture class labels (0–14)
│     # Note: dataset.csv (60,000 samples) not included — available on request
│
└── utils/
    ├── preprocessing.py          # Landmark normalisation pipeline
    └── drawing_landmarks.py      # Landmark drawing + extraction helpers
```

---

## Installation

> **Python 3.12 is required.** The exact versions of MediaPipe and TensorFlow used in this project only work with Python 3.12.

### 1. Clone the repository

```bash
git clone https://github.com/Samm-006/Hand-Gesture-Recognition.git
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the MediaPipe model

Download `hand_landmarker.task` from [MediaPipe Models](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) and place it inside the `model/` folder.

---

## Requirements

**`requirements.txt`**

```
mediapipe==0.10.21
tensorflow==2.16.2
opencv-python==4.11.0.86
streamlit==1.53.0
streamlit-webrtc==0.64.5
numpy==1.26.4
pandas==2.3.3
scikit-learn==1.8.0
matplotlib==3.10.8
seaborn==0.13.2
gTTS==2.5.4
av==16.1.0
```

> These are the exact versions tested to work together on Python 3.12

---

## Running the Project

### Run the web application

```bash
streamlit run app.py
```

### Collect your own gesture data

```bash
python 02_collect_dataset.py
```

| Key | Action |
|-----|--------|
| `k` | Start recording frames |
| `n` | Stop recording |
| `0`–`9` | Select gesture label 0–9 |
| `a` `s` `d` `f` `g` | Select gesture label 10–14 |
| `ESC` | Exit |

### Train the model

Open and run `03_trained_model.ipynb` in Jupyter or VS Code.

---

## How It Works

```
Webcam → MediaPipe → Preprocessing → Feature Vector → Neural Network → Output
  |           |             |               |                |             |
RGB        21 kpts      Translate       84-dim           Softmax(15)   Phrase
frames    × 2 hands     Flatten        vector            + Smoothing   + TTS
                       Normalise
```

**Model architecture:**

- Input: 84 features (21 landmarks × 2 coordinates × 2 hands)
- `Dropout(0.1)` → `Dense(256, ReLU)` → `Dropout(0.2)` → `Dense(128, ReLU)` → `Dropout(0.1)` → `Dense(64, ReLU)` → `Softmax(15)`
- Trained on 60,000 samples across varied lighting, distance, and hand orientations

---

## Dataset — EmergencySL

- **15 gesture classes:** I, Need, Accident, Fire, Yes, No, Ambulance, Doctor, Police, Fireman, Hurt, Emergency, Short of breath, Help, Stop
- **60,000 samples** (~4,000 per class)
- Captured across: poor/good lighting · close/far distance · left/right hand · front/side orientation
- Dataset available on request

---

## Resources

- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## Author

**Sama Sultan A Alzahrani** · BSc Artificial Intelligence and Computer Science  
University of Birmingham · Supervisor: Samuel Montero Hernandez
