import argparse
import json
import os
import random
import logging
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

logging.basicConfig(filename='training_log.txt', level=logging.INFO, format='%(asctime)s %(message)s')

DATA_DIR = r"C:\DARAM-ai-Archive\knows_faces"


# 얼굴 이미지 데이터셋 클래스
class FaceDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logging.error(f"이미지 로드 오류: {img_path}, {e}")
            image = Image.new('RGB', (224, 224))
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def load_images_for_recognition(folder, target_folders=None):
    image_paths = []
    labels = []
    class_names = sorted([subdir for subdir in os.listdir(folder) if os.path.isdir(os.path.join(folder, subdir))])

    if target_folders is not None:
        class_names = [subdir for subdir in class_names if subdir in target_folders]
        if not class_names:
            raise ValueError("입력한 폴더 식별자가 데이터 디렉토리에 없습니다.")

    label_mapping = {class_name: idx for idx, class_name in enumerate(class_names)}

    for class_name in class_names:
        label = label_mapping[class_name]
        subdir_path = os.path.join(folder, class_name)
        for filename in os.listdir(subdir_path):
            img_path = os.path.join(subdir_path, filename)
            image_paths.append(img_path)
            labels.append(label)

    return image_paths, labels, label_mapping


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


# 조기 종료를 관리하는 클래스
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def monitor_gpu(log_once=False):
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        log_message = f"GPU Memory: Allocated={mem_allocated:.2f} MB, Reserved={mem_reserved:.2f} MB"
        if log_once:
            print(log_message)
        logging.info(log_message)


def train_face_recognition_model(args, use_dropout, target_folders):
    image_paths, labels, label_mapping = load_images_for_recognition(DATA_DIR, target_folders)

    if len(image_paths) == 0:
        print("학습에 사용할 이미지가 없습니다.")
        return

    X_train, X_test, y_train, y_test = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

    class_counts = Counter(y_train)
    class_weights = [1.0 / class_counts[i] for i in range(len(class_counts))]
    sample_weights = [class_weights[y] for y in y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = FaceDataset(X_train, y_train, transform=transform_train)
    test_dataset = FaceDataset(X_test, y_test, transform=transform_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.5) if use_dropout else nn.Identity(),
        nn.Linear(num_features, len(label_mapping))
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
    early_stopping = EarlyStopping(patience=args.patience)

    monitor_gpu(log_once=True)

    metrics = {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}
    for epoch in range(args.epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
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
        val_loss, correct_val, total_val = 0.0, 0, 0
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

        logging.info(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                     f"Train Acc={train_accuracy:.2f}, Val Acc={val_accuracy:.2f}")

        if early_stopping(val_loss):
            print("조기 종료")
            break

    torch.save(model.state_dict(), 'face_recognition_model.pth')
    save_metrics_to_excel(metrics, 'face_recognition_metrics.xlsx')
    plot_metrics(metrics, 'Face Recognition Model Metrics', 'face_recognition_metrics.png')

    with open('label_mapping.json', 'w', encoding='utf-8') as f:
        json.dump({v: k for k, v in label_mapping.items()}, f)

    print("테스트 정확도:", val_accuracy)


def get_user_inputs():
    print("========== 학습 설정 ==========")
    while True:
        use_dropout_input = input("드롭아웃을 사용하시겠습니까? (yes/no): ").strip().lower()
        if use_dropout_input in ["yes", "no"]:
            use_dropout = use_dropout_input == "yes"
            break
        else:
            print("잘못된 입력입니다. 'yes' 또는 'no'로 입력해주세요.")

    while True:
        target_folders_input = input("학습에 사용할 폴더 이름들을 쉼표로 구분하여 입력하세요 (예: 1115,1211): ").strip()
        if target_folders_input:
            target_folders = [folder.strip() for folder in target_folders_input.split(',')]
            break
        else:
            print("적어도 하나의 폴더 이름을 입력해주세요.")

    return use_dropout, target_folders


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Recognition Model Training")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    args = parser.parse_args()

    use_dropout, target_folders = get_user_inputs()
    train_face_recognition_model(args, use_dropout, target_folders)