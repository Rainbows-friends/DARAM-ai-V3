import json
import logging
import os
import time

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BASE_DIR = 'C:\\DARAM-ai-Archive'
KNOWN_FACES_DIR = os.path.join(BASE_DIR, 'knows_faces')
OTHER_FACES_DIR = os.path.join(KNOWN_FACES_DIR, 'Other')
NON_FACES_DIR = os.path.join(BASE_DIR, 'non_faces')

logging.basicConfig(filename='error_log.txt', level=logging.ERROR, format='%(asctime)s:%(levelname)s:%(message)s')


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


def load_images_from_folder(folder, label=None, img_size=(224, 224)):
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
                    if label is not None:
                        labels.append(label)
                else:
                    warning_message = f"Warning: Skipping invalid image {img_path}"
                    print(warning_message)
                    logging.warning(f"Warning Code 10003: {warning_message}")
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return images, labels


def save_metrics_to_excel(metrics, filename):
    df = pd.DataFrame(metrics)
    df.to_excel(filename, index=False)


def plot_metrics(metrics, title, filename):
    plt.figure()
    plt.plot(metrics['epoch'], metrics['train_loss'], label='Train Loss', marker='o')
    plt.plot(metrics['epoch'], metrics['val_loss'], label='Validation Loss', marker='o')
    plt.plot(metrics['epoch'], metrics['train_accuracy'], label='Train Accuracy', marker='o')
    plt.plot(metrics['epoch'], metrics['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def monitor_system():
    mem = psutil.virtual_memory()
    mem_available = mem.available / (1024 ** 3)  # Convert to GB
    return mem_available


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def log_error(message, error_code):
    logging.error(f"Error Code {error_code}: {message}")


def load_faces_and_labels():
    known_faces = []
    known_labels = []
    non_faces, _ = load_images_from_folder(NON_FACES_DIR, 0)
    other_faces, _ = load_images_from_folder(OTHER_FACES_DIR, 0)

    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if os.path.isdir(person_dir) and person_name != "Other":
            person_images, _ = load_images_from_folder(person_dir, 1)
            known_faces.extend(person_images)
            known_labels.extend([person_name] * len(person_images))

    return known_faces, known_labels, non_faces, other_faces


def train_face_detection_model():
    known_faces, known_labels, non_faces, other_faces = load_faces_and_labels()

    all_images = known_faces + non_faces + other_faces
    all_labels = known_labels + [0] * (len(non_faces) + len(other_faces))

    if len(all_images) == 0:
        log_error(f"No images found in {KNOWN_FACES_DIR} and {OTHER_FACES_DIR} and {NON_FACES_DIR}", 10000)
        return

    X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
         transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
         transforms.Resize((224, 224)), ])

    train_dataset = FaceDataset(X_train, y_train, transform=transform)
    test_dataset = FaceDataset(X_test, y_test,
                               transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))]))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = False

    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    num_epochs = 50
    early_stopping = EarlyStopping(patience=5, min_delta=0.01)
    metrics = {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}

    under_memory_duration = 0

    for epoch in range(num_epochs):
        mem_available = monitor_system()

        while mem_available < 1.5:
            under_memory_duration += 1

            if under_memory_duration >= 2:
                print("System is low on memory, pausing training...")
                while mem_available < 1.5:
                    time.sleep(60)
                    mem_available = monitor_system()
                under_memory_duration = 0
                print("Resuming training...")

            time.sleep(60)
            mem_available = monitor_system()

        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss /= len(test_loader)
        val_accuracy = 100 * correct_val / total_val

        scheduler.step(val_loss)

        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['train_accuracy'].append(train_accuracy)
        metrics['val_accuracy'].append(val_accuracy)

        with open('training_log.txt', 'a') as log_file:
            log_file.write(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Validation Loss: {val_loss}, '
                           f'Train Accuracy: {train_accuracy}, Validation Accuracy: {val_accuracy}\n')

        print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Validation Loss: {val_loss}, '
              f'Train Accuracy: {train_accuracy}, Validation Accuracy: {val_accuracy}')

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    if os.path.exists('face_detection_model.pth'):
        os.remove('face_detection_model.pth')
    torch.save(model.state_dict(), 'face_detection_model.pth')
    print("모델 저장 완료: face_detection_model.pth")

    save_metrics_to_excel(metrics, 'face_detection_metrics.xlsx')
    plot_metrics(metrics, 'Face Detection Model Metrics', 'face_detection_metrics.png')

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


def train_face_recognition_model():
    known_faces = []
    known_labels = []

    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if os.path.isdir(person_dir) and person_name != "Other":
            person_images, _ = load_images_from_folder(person_dir, person_name)
            known_faces.extend(person_images)
            known_labels.extend([person_name] * len(person_images))

    if len(known_faces) == 0:
        log_error(f"No images found in {KNOWN_FACES_DIR} except 'Other'", 10000)
        return

    valid_classes = list(set(known_labels))
    label_mapping = {i: valid_classes[i] for i in range(len(valid_classes))}

    num_classes = len(valid_classes)
    all_labels = [label_mapping[label] for label in known_labels]

    X_train, X_test, y_train, y_test = train_test_split(known_faces, all_labels, test_size=0.2, random_state=42)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
         transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
         transforms.Resize((224, 224)), ])

    train_dataset = FaceDataset(X_train, y_train, transform=transform)
    test_dataset = FaceDataset(X_test, y_test,
                               transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))]))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = False

    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
    early_stopping = EarlyStopping(patience=5, min_delta=0.01)

    num_epochs = 50
    metrics = {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}

    under_memory_duration = 0

    for epoch in range(num_epochs):
        mem_available = monitor_system()

        while mem_available < 1.5:
            under_memory_duration += 1

            if under_memory_duration >= 2:
                print("System is low on memory, pausing training...")
                while mem_available < 1.5:
                    time.sleep(60)
                    mem_available = monitor_system()
                under_memory_duration = 0
                print("Resuming training...")

            time.sleep(60)
            mem_available = monitor_system()

        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss /= len(test_loader)
        val_accuracy = 100 * correct_val / total_val

        scheduler.step(val_loss)

        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['train_accuracy'].append(train_accuracy)
        metrics['val_accuracy'].append(val_accuracy)

        with open('training_log.txt', 'a') as log_file:
            log_file.write(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Validation Loss: {val_loss}, '
                           f'Train Accuracy: {train_accuracy}, Validation Accuracy: {val_accuracy}\n')

        print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Validation Loss: {val_loss}, '
              f'Train Accuracy: {train_accuracy}, Validation Accuracy: {val_accuracy}')

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    if os.path.exists('face_recognition_model.pth'):
        os.remove('face_recognition_model.pth')
    torch.save(model.state_dict(), 'face_recognition_model.pth')
    print("모델 저장 완료: face_recognition_model.pth")

    save_metrics_to_excel(metrics, 'face_recognition_metrics.xlsx')
    plot_metrics(metrics, 'Face Recognition Model Metrics', 'face_recognition_metrics.png')

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