import logging
import os

import cv2
import numpy as np
from keras._tf_keras.keras.models import load_model
from keras_preprocessing.image import img_to_array

log_dir = os.path.join(os.path.dirname(__file__), 'Logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(filename=os.path.join(log_dir, 'face_recognition.log'), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def log_and_print(message):
    print(message)
    logging.info(message)


model_path = 'face_classifier.h5'
model = load_model(model_path)
log_and_print(f"Model loaded from {model_path}")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

data_dir = 'C:\\DARAM-ai-Archive'
categories = [name for name in os.listdir(os.path.join(data_dir, 'knows_faces')) if
              os.path.isdir(os.path.join(data_dir, 'knows_faces', name))]


def preprocess_face(face):
    face = cv2.resize(face, (64, 64))
    face = face.astype("float") / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    return face


def recognize_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        processed_face = preprocess_face(face)
        prediction = model.predict(processed_face)[0]
        max_index = np.argmax(prediction)
        confidence = prediction[max_index]

        if confidence > 0.5:
            label = categories[max_index]
            color = (255, 255, 0)  # 하늘색
            log_and_print(f"Recognized {label} with confidence {confidence:.2f}")
        else:
            label = "Unknown"
            color = (0, 0, 255)  # 빨간색
            log_and_print("Detected unknown face")

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return frame


cap = cv2.VideoCapture(0)

log_and_print("Starting video stream...")

while True:
    ret, frame = cap.read()
    if not ret:
        log_and_print("Failed to capture frame")
        break

    frame = recognize_faces(frame)
    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        log_and_print("Exiting video stream...")
        break

cap.release()
cv2.destroyAllWindows()
log_and_print("Video stream ended")