import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load model
MODEL_PATH = "emotion_model.h5"
model = load_model(MODEL_PATH)

# Emotion labels (adjust according to your training)
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

st.set_page_config(page_title="Emotion Detection App", layout="centered")
st.title("😊 Emotion Detection from Image")

# Upload or Webcam
option = st.radio("Select Input Method:", ["Upload Image", "Use Webcam"])

def detect_emotion(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    detections = faces.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(detections) == 0:
        return None, None

    for (x, y, w, h) in detections:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        prediction = model.predict(roi)[0]
        label = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction)
        return label, confidence

    return None, None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img = np.array(img.convert("RGB"))
        label, confidence = detect_emotion(img)
        if label:
            st.image(img, caption=f"Detected Emotion: {label} ({confidence*100:.2f}%)")
        else:
            st.warning("No face detected.")

elif option == "Use Webcam":
    st.write("Click 'Start' to capture")
    picture = st.camera_input("Take a picture")
    if picture:
        img = Image.open(picture)
        img = np.array(img.convert("RGB"))
        label, confidence = detect_emotion(img)
        if label:
            st.image(img, caption=f"Detected Emotion: {label} ({confidence*100:.2f}%)")
        else:
            st.warning("No face detected.")
