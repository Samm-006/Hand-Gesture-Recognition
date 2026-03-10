# Real-Time Vision-Based Hand Gesture Recognition for Emergency Sign Language Interpretation

### Overview

A Real-Time Vision-Based Hand Gesture Recognition for Emergency Sign Language Interpretation is a computer vision application designed to translate emergency hand gestures into clear text and spoken messages.

The system uses MediaPipe hand tracking to detect hand landmarks from a webcam in real time. These landmarks are processed and passed into a TensorFlow-based gesture classification model that predicts the gesture being performed. Once a gesture is recognised, it is mapped to a predefined emergency phrase and displayed on screen. The system can also convert the phrase into speech using text-to-speech technology.

The purpose of this project is to explore how machine learning and real-time computer vision can support communication during emergency situations where verbal communication may not be possible.

The application is implemented using Python, with a Streamlit web interface that allows users to interact with the system through a browser.
