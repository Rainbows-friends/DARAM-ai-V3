import os

import cv2
import numpy as np
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.utils import to_categorical
from keras_preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_faces(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    detected_faces = []
    for (x, y, w, h) in faces:
        face = img[y:y + h, x:x + w]
        detected_faces.append(cv2.resize(face, (64, 64)))
    return detected_faces


def load_data(data_dir, categories, img_size=(64, 64)):
    data = []
    labels = []
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        for subdir in os.listdir(path):
            subdir_path = os.path.join(path, subdir)
            if os.path.isdir(subdir_path):
                for img in os.listdir(subdir_path):
                    try:
                        img_path = os.path.join(subdir_path, img)
                        if category == 'knows_faces' and subdir != 'Other':
                            faces = detect_faces(img_path)
                            for face in faces:
                                data.append(face)
                                labels.append(class_num)
                        elif category == 'non_faces':
                            img = load_img(img_path, target_size=img_size)
                            img_array = img_to_array(img)
                            data.append(img_array)
                            labels.append(class_num)
                    except Exception as e:
                        print(e)
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    return data, labels


data_dir = 'C:\\DARAM-ai-Archive'
categories = ['knows_faces', 'non_faces']

data, labels = load_data(data_dir, categories)
labels = to_categorical(labels, num_classes=len(categories))
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.2, random_state=42)

model = Sequential([Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)), MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'), MaxPooling2D((2, 2)), Flatten(), Dense(128, activation='relu'),
    Dense(len(categories), activation='softmax')])

model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(trainX, trainY, epochs=10, validation_data=(testX, testY), batch_size=32)

model.save('face_classifier.h5')

loss, accuracy = model.evaluate(testX, testY)
print(f"Test accuracy: {accuracy}")