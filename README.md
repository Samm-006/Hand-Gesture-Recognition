# Real-Time Vision-Based Hand Gesture Recognition for Emergency Sign Language Interpretation

### <ins>Overview</ins>

A Real-Time Vision-Based Hand Gesture Recognition for Emergency Sign Language Interpretation is a computer vision application designed to translate emergency hand gestures into clear text and spoken messages.

The system uses MediaPipe hand tracking to detect hand landmarks from a webcam in real time. These landmarks are processed and passed into a TensorFlow-based gesture classification model that predicts the gesture being performed. Once a gesture is recognised, it is mapped to a predefined emergency phrase and displayed on screen. The system can also convert the phrase into speech using text-to-speech technology.

The purpose of this project is to explore how machine learning and real-time computer vision can support communication during emergency situations where verbal communication may not be possible.

The application is implemented using Python, with a Streamlit web interface that allows users to interact with the system through a browser.




### <ins>Live Application Demo</ins>

> A live demonstration of the system is available online: [App Demo](https://emergency-gesture-recognition.streamlit.app/]).

This deployed application allows users to test the gesture recognition system directly from their browser using a webcam.


#### The system performs the following steps in real time:

1. Captures webcam frames from the user's device.
2. Detects hands using MediaPipe.
3. Extracts 21 hand landmarks per hand.
4. Preprocesses the landmark coordinates.
5. Uses a trained TensorFlow model to classify the gesture.
6. Maps the gesture to a predefined emergency phrase.
7. Displays the phrase and optionally generates audio output.
