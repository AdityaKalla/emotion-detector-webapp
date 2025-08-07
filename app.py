import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Load emotion model and Haar cascade
model = load_model("emotion_model.h5")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ---------- UI -----------
st.set_page_config(page_title="Face Emotion Recognition", layout="centered")
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 2rem;
        border-radius: 15px;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        color: #2c3e50;
    }
    .upload-section {
        text-align: center;
        padding-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title">😊 Real-Time Face Emotion Recognition</div>', unsafe_allow_html=True)
st.write("Detect facial emotions using your **webcam** or by uploading an image.")

# ---------- Webcam Mode ----------
class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (48, 48))
            face = face / 255.0
            face = face.reshape(1, 48, 48, 1)
            prediction = model.predict(face)
            emotion = emotion_labels[np.argmax(prediction)]

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return img

st.subheader("📷 Live Webcam Emotion Detection")
webrtc_streamer(key="emotion", video_processor_factory=EmotionDetector)

# ---------- Image Upload Mode ----------
st.subheader("🖼️ Upload Image for Emotion Detection")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = face.reshape(1, 48, 48, 1)
        prediction = model.predict(face)
        emotion = emotion_labels[np.argmax(prediction)]
        cv2.rectangle(img_np, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img_np, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    st.image(img_np, caption="Detected Emotions", use_column_width=True)

st.markdown('</div>', unsafe_allow_html=True)

