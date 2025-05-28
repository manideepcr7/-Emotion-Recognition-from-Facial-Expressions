
import gradio as gr
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tempfile

# Load pre-trained model (you should replace this with your own trained model)
model = load_model("emotion_model.h5")

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = model.predict(roi, verbose=0)[0]
        label = emotion_labels[np.argmax(preds)]
        return label
    return "No face detected"

iface = gr.Interface(fn=preprocess_face,
                     inputs=gr.Image(type="numpy", label="Upload Face Image"),
                     outputs="text",
                     title="Emotion Recognition from Facial Expressions",
                     description="Upload a face image to predict the emotion.")
iface.launch()
