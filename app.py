import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import numpy as np

# Dummy emotion detector — replace with your model logic
def detect_emotion(frame):
    return "😊 Happy"

class EmotionTransformer(VideoTransformerBase):
    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")
        emotion = detect_emotion(img)

        # Display the emotion on the frame
        cv2.putText(img, emotion, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

st.title("🧠 Emotion Detector Web App")
st.write("This app captures webcam feed and detects emotion in real-time.")

webrtc_streamer(key="emotion", video_transformer_factory=EmotionTransformer)
