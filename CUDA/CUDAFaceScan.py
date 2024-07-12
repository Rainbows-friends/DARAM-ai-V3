import os
import json
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# GPU 설정을 제거하고 CPU만 사용하도록 변경
device = torch.device("cpu")
print(f"Using device: {device}")

# 얼굴 검출 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 256 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 얼굴 분류 모델 정의
class FaceRecognitionCNN(nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognitionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 256 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 모델 로드 함수
def load_model(model_path, model_class, num_classes=None):
    if num_classes:
        model = model_class(num_classes)
    else:
        model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# 얼굴 검출 함수
def detect_faces(image, face_detector):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    detected_faces = []
    for (x, y, w, h) in faces:
        face = image[y:y + h, x:x + w]
        face = cv2.resize(face, (128, 128))
        face_tensor = transforms.ToTensor()(face).unsqueeze(0).to(device)
        output = face_detector(face_tensor)
        _, predicted = torch.max(output.data, 1)
        if predicted.item() == 1:  # 얼굴 검출된 경우
            detected_faces.append((x, y, w, h, face_tensor))
    return detected_faces

# 얼굴 분류 함수
def classify_faces(detected_faces, face_recognizer, label_mapping):
    results = []
    for (x, y, w, h, face_tensor) in detected_faces:
        output = face_recognizer(face_tensor)
        _, predicted = torch.max(output.data, 1)
        label = label_mapping[str(predicted.item())]
        results.append((x, y, w, h, label))
    return results

# 웹캠에서 사진 찍기
def capture_image_from_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Could not read frame.")
        return None

    return frame

# 메인 함수
def main():
    # 모델 로드
    face_detector = load_model('face_detection_model.pth', SimpleCNN)
    with open('label_mapping.json', 'r') as f:
        label_mapping = json.load(f)
    face_recognizer = load_model('face_recognition_model.pth', FaceRecognitionCNN, num_classes=len(label_mapping))

    # 웹캠에서 사진 찍기
    image = capture_image_from_webcam()
    if image is None:
        return

    # 얼굴 검출
    detected_faces = detect_faces(image, face_detector)
    if not detected_faces:
        print("No faces detected.")
        return

    # 얼굴 분류
    results = classify_faces(detected_faces, face_recognizer, label_mapping)

    # 결과 출력 및 시각화
    for (x, y, w, h, label) in results:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    cv2.imshow('Face Detection and Recognition', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()