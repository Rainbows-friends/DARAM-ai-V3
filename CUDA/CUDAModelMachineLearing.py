import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BASE_DIR = 'C:\\DARAM-ai-Archive'
KNOWN_FACES_DIR = os.path.join(BASE_DIR, 'knows_faces')
OTHER_FACES_DIR = os.path.join(BASE_DIR, 'non_faces')


class FaceDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def load_images_from_folder(folder, label, img_size=(128, 128)):
    images = []
    labels = []
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            img_path = os.path.join(subdir, file)
            try:
                img = cv2.imread(img_path)
                if img is not None and img.size > 0:
                    img = cv2.resize(img, img_size)
                    images.append(img)
                    labels.append(label)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return images, labels


def train_face_detection_model():
    all_images = []
    all_labels = []

    known_faces, known_labels = load_images_from_folder(KNOWN_FACES_DIR, 1)
    all_images.extend(known_faces)
    all_labels.extend(known_labels)

    non_faces, non_labels = load_images_from_folder(OTHER_FACES_DIR, 0)
    all_images.extend(non_faces)
    all_labels.extend(non_labels)

    all_images = np.array(all_images, dtype="float32") / 255.0
    all_labels = np.array(all_labels, dtype="int64")

    X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomResizedCrop(128, scale=(0.8, 1.0)), ])

    train_dataset = FaceDataset(X_train, y_train, transform=transform)
    test_dataset = FaceDataset(X_test, y_test, transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
            self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.fc1 = nn.Linear(512 * 4 * 4, 1024)
            self.fc2 = nn.Linear(1024, 2)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = self.pool(F.relu(self.conv4(x)))
            x = self.pool(F.relu(self.conv5(x)))
            x = x.view(-1, 512 * 4 * 4)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        scheduler.step(val_loss / len(test_loader))

    if os.path.exists('face_detection_model.pth'):
        os.remove('face_detection_model.pth')
    torch.save(model.state_dict(), 'face_detection_model.pth')
    print("모델 저장 완료: face_detection_model.pth")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("테스트 셋에서의 정확도: ", 100 * correct / total)


def load_images_for_recognition(folder, label=None, sample_size=None):
    images = []
    labels = []
    file_list = os.listdir(folder)

    if sample_size is not None and len(file_list) > sample_size:
        file_list = random.sample(file_list, sample_size)

    for filename in file_list:
        img_path = os.path.join(folder, filename)
        try:
            img = cv2.imread(img_path)
            if img is not None and img.size > 0:
                img = cv2.resize(img, (128, 128))
                images.append(img)
                if label is not None:
                    labels.append(label)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    return images, labels


def train_face_recognition_model():
    all_images = []
    all_labels = []
    valid_classes = []
    label_mapping = {}

    for label, face_dir in enumerate(os.listdir(KNOWN_FACES_DIR)):
        face_path = os.path.join(KNOWN_FACES_DIR, face_dir)
        if face_dir == 'Other':
            continue
        valid_classes.append(face_dir)
        label_mapping[len(valid_classes) - 1] = face_dir
        images, labels = load_images_for_recognition(face_path, len(valid_classes) - 1)
        all_images.extend(images)
        all_labels.extend(labels)

    num_classes = len(valid_classes)
    all_images = np.array(all_images, dtype="float32") / 255.0
    all_labels = np.array(all_labels, dtype="int64")

    X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomResizedCrop(128, scale=(0.8, 1.0)), ])

    train_dataset = FaceDataset(X_train, y_train, transform=transform)
    test_dataset = FaceDataset(X_test, y_test, transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    class FaceRecognitionCNN(nn.Module):
        def __init__(self, num_classes):
            super(FaceRecognitionCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
            self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.fc1 = nn.Linear(512 * 4 * 4, 1024)
            self.fc2 = nn.Linear(1024, num_classes)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = self.pool(F.relu(self.conv4(x)))
            x = self.pool(F.relu(self.conv5(x)))
            x = x.view(-1, 512 * 4 * 4)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    model = FaceRecognitionCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        scheduler.step(val_loss / len(test_loader))

    if os.path.exists('face_recognition_model.pth'):
        os.remove('face_recognition_model.pth')
    torch.save(model.state_dict(), 'face_recognition_model.pth')
    print("모델 저장 완료: face_recognition_model.pth")

    with open('label_mapping.json', 'w') as f:
        json.dump(label_mapping, f)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("테스트 셋에서의 정확도: ", 100 * correct / total)


if __name__ == "__main__":
    print("얼굴 검출 모델 학습 시작...")
    train_face_detection_model()
    print("얼굴 검출 모델 학습 완료")
    print("얼굴 인식 모델 학습 시작...")
    train_face_recognition_model()
    print("얼굴 인식 모델 학습 완료")