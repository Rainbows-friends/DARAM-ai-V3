import os

import cv2
import numpy as np
from keras._tf_keras.keras import Sequential
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

KNOWN_FACES_DIR = 'C:\\DARAM-ai-V2\\knows_faces'
OTHER_FACES_DIR = 'C:\\DARAM-ai-V2\\non_faces'


def load_images_from_folder(folder, label, img_size=(128, 128)):
    images = []
    labels = []
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            img_path = os.path.join(subdir, file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(label)
    return images, labels


def detect_faces(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    detected_faces = []
    for (x, y, w, h) in faces:
        face = img[y:y + h, x:x + w]
        detected_faces.append(cv2.resize(face, (128, 128)))
    return detected_faces


all_images = []
all_labels = []

known_faces, known_labels = load_images_from_folder(KNOWN_FACES_DIR, 1)
all_images.extend(known_faces)
all_labels.extend(known_labels)

non_faces, non_labels = load_images_from_folder(OTHER_FACES_DIR, 0)
all_images.extend(non_faces)
all_labels.extend(non_labels)

all_images = np.array(all_images, dtype="float") / 255.0
all_labels = to_categorical(np.array(all_labels), num_classes=2)

X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
    zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

model = Sequential(
    [Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)), BatchNormalization(), MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'), BatchNormalization(), MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'), BatchNormalization(), MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu'), BatchNormalization(), MaxPooling2D((2, 2)), Flatten(),
        Dense(512, activation='relu'), Dropout(0.5), Dense(2, activation='softmax')])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_face_detection_model.keras', save_best_only=True, monitor='val_loss')

model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=20, validation_data=(X_test, y_test),
          callbacks=[early_stopping, model_checkpoint])

model.save('face_detection_model.keras')
print("모델 저장 완료: face_detection_model.keras")

predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

print("테스트 셋에서의 정확도: ", np.mean(predicted_labels == true_labels))
for i in range(10):
    true_label = true_labels[i]
    predicted_label = predicted_labels[i]
    print(f"실제 라벨: {true_label}, 예측 라벨: {predicted_label}")