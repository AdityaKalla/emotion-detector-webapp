import streamlit as st

st.set_page_config(page_title="Emotion Detector", layout="centered")
st.title("🧠 Real-Time Emotion Detection App")

import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Load emotion model and Haar cascade
# Note: You need to have 'emotion_model.h5' and 'haarcascade_frontalface_default.xml' in the same directory.
try:
    model = load_model("emotion_model.h5")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
except Exception as e:
    st.error(f"Error loading model or cascade: {e}")
    st.stop()


# --- Custom CSS for a professional look ---
st.set_page_config(page_title="Pro Emotion Recognition", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    
    body {
        font-family: 'Poppins', sans-serif;
    }
    .main {
        background-color: #f0f2f6; /* Soft gray background */
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .stApp {
        background-color: #ffffff;
    }
    .title {
        font-size: 2.5rem;
        font-weight: 600;
        text-align: center;
        color: #1e3a8a; /* Dark blue title */
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1rem;
        font-weight: 400;
        text-align: center;
        color: #6b7280; /* Gray subtitle */
        margin-bottom: 2rem;
    }
    .stSubheader {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e3a8a;
        border-left: 5px solid #3b82f6; /* Blue border for a modern touch */
        padding-left: 10px;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .section-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
    }
    .st-emotion-detector-section {
        background-color: #f7f9fd;
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid #e0e7ff;
    }
    .stFileUploader {
        border: 2px dashed #9ca3af;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
    }
    .image-caption {
        font-style: italic;
        text-align: center;
        color: #4b5563;
        margin-top: 1rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #9ca3af;
        font-size: 0.8rem;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<h1 class="title">AI-Powered Face Emotion Recognition</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Experience real-time emotion detection using your webcam or by uploading a photo.</p>', unsafe_allow_html=True)

# --- Webcam Mode ---
class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Adjust parameters for better detection if needed
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            try:
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
                face = face / 255.0
                face = face.reshape(1, 48, 48, 1)
                
                prediction = model.predict(face, verbose=0)
                emotion = emotion_labels[np.argmax(prediction)]

                # Dynamic color based on emotion (can be customized)
                color = (0, 255, 0)
                if emotion == 'Sad': color = (255, 0, 0)
                elif emotion == 'Angry': color = (0, 0, 255)
                elif emotion == 'Fear': color = (0, 255, 255)
                elif emotion == 'Surprise': color = (255, 255, 0)
                
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            except Exception as e:
                # Handle cases where face resizing might fail
                print(f"Error processing face: {e}")
                continue
                
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.markdown('<div class="section-container">', unsafe_allow_html=True)
st.markdown('<h2 class="stSubheader">📷 Live Webcam Emotion Detection</h2>', unsafe_allow_html=True)
st.markdown('<div class="st-emotion-detector-section">', unsafe_allow_html=True)
webrtc_streamer(key="emotion", video_processor_factory=EmotionDetector)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Image Upload Mode ---
st.markdown('<div class="section-container">', unsafe_allow_html=True)
st.markdown('<h2 class="stSubheader">🖼️ Upload an Image for Emotion Detection</h2>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a face image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Convert the uploaded file to a format OpenCV can use
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_np = cv2.imdecode(file_bytes, 1)
        
        # Ensure image is in RGB format for consistent processing
        img_np_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            st.warning("No faces detected in the uploaded image. Please try another one.")
        else:
            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
                face = face / 255.0
                face = face.reshape(1, 48, 48, 1)
                
                prediction = model.predict(face, verbose=0)
                emotion = emotion_labels[np.argmax(prediction)]
                
                # Use a specific color for the uploaded image box
                cv2.rectangle(img_np_rgb, (x, y), (x + w, y + h), (255, 128, 0), 2)
                cv2.putText(img_np_rgb, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 128, 0), 2)

            st.image(img_np_rgb, caption="Detected Emotions", use_column_width=True)
            st.markdown('<p class="image-caption">The detected emotion is shown above the face.</p>', unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="footer">Built with Streamlit and Keras</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
