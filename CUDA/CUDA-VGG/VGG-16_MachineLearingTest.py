import json
import os
import random
import time
import gc  # 가비지 컬렉션 모듈 추가
import logging  # 로깅 모듈 추가
import sys  # 표준 출력 재정의를 위한 모듈 추가

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
BASE_DIR = 'C:\\DARAM-ai-Archive'
KNOWN_FACES_DIR = os.path.join(BASE_DIR, 'knows_faces')

# 로깅 설정
logging.basicConfig(filename='training_log.txt', level=logging.INFO, format='%(asctime)s %(message)s')

# 표준 출력과 에러를 로그 파일로 리다이렉트
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open('training_log.txt', 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # 버퍼를 즉시 비워서 실시간으로 기록되도록 함

    def flush(self):
        pass

sys.stdout = Logger()
sys.stderr = Logger()

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
            print(f"이미지 로드 중 오류 발생 {img_path}: {e}")
            # 빈 이미지 생성 (필요에 따라 조정 가능)
            image = Image.new('RGB', (224, 224))
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

def load_images_for_recognition(folder, sample_size=None, target_folders=None):
    image_paths = []
    labels = []
    image_counts = {}
    class_names = []
    subdirs = [subdir for subdir in sorted(os.listdir(folder)) if
               os.path.isdir(os.path.join(folder, subdir)) and subdir != 'Other']

    if target_folders is not None:
        subdirs = [subdir for subdir in subdirs if subdir in target_folders]
        valid_subdirs = subdirs
        class_names = subdirs
        label_mapping = {class_name: idx for idx, class_name in enumerate(class_names)}
    else:
        for subdir in subdirs:
            subdir_path = os.path.join(folder, subdir)
            num_images = len(os.listdir(subdir_path))
            image_counts[subdir] = num_images

        if len(image_counts) > 0:
            average_num_images = sum(image_counts.values()) / len(image_counts)
        else:
            average_num_images = 0

        valid_subdirs = []
        for subdir in subdirs:
            if image_counts[subdir] >= average_num_images:
                valid_subdirs.append(subdir)
                class_names.append(subdir)
            else:
                print(f"폴더 '{subdir}'는 이미지 수가 적어 학습에서 제외됩니다.")

        label_mapping = {class_name: idx for idx, class_name in enumerate(class_names)}

    for class_name in valid_subdirs:
        label = label_mapping[class_name]
        subdir_path = os.path.join(folder, class_name)
        file_list = os.listdir(subdir_path)
        if sample_size is not None and len(file_list) > sample_size:
            file_list = random.sample(file_list, sample_size)
        for filename in file_list:
            img_path = os.path.join(subdir_path, filename)
            image_paths.append(img_path)
            labels.append(label)
    return image_paths, labels, label_mapping

def augment_dataset(image_paths, labels, target_size):
    current_size = len(image_paths)
    if current_size >= target_size:
        return image_paths, labels
    num_new_images = target_size - current_size
    new_image_paths = []
    new_labels = []
    for _ in range(num_new_images):
        idx = random.randint(0, current_size - 1)
        img_path = image_paths[idx]
        label = labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            # 랜덤 변환 적용
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])
            image_aug = transform(image)
            # 증강된 이미지를 임시로 메모리에 저장
            new_image_paths.append(image_aug)
            new_labels.append(label)
        except Exception as e:
            print(f"이미지 증강 중 오류 발생 {img_path}: {e}")
    # 기존 이미지 경로와 레이블에 증강된 이미지 추가
    image_paths.extend(new_image_paths)
    labels.extend(new_labels)
    return image_paths, labels

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
    mem_available = mem.available / (1024 ** 3)
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
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_face_recognition_model(use_dropout, target_folders=None):
    image_paths, labels, label_mapping = load_images_for_recognition(KNOWN_FACES_DIR,
                                                                     target_folders=target_folders)
    if len(image_paths) == 0:
        print("학습에 사용할 이미지가 없습니다.")
        return

    # 데이터 증강을 통해 이미지 수를 1900개로 증가
    target_size = 7000
    image_paths, labels = augment_dataset(image_paths, labels, target_size)

    num_classes = len(label_mapping)

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

    # 데이터 변환 정의
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 데이터셋 및 데이터로더 생성
    train_dataset = FaceDataset(X_train, y_train, transform=transform_train)
    test_dataset = FaceDataset(X_test, y_test, transform=transform_test)

    # 배치 크기 조정 (더 작은 배치 크기 사용)
    batch_size = 16

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 모델 초기화
    model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')

    # 필요한 레이어만 학습하도록 설정
    for param in model.parameters():
        param.requires_grad = False

    # 드롭아웃 사용 여부에 따라 모델 정의
    if use_dropout:
        model.classifier[6] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.classifier[6].in_features, num_classes)
        )
    else:
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    model = model.to(device)

    # 손실 함수 및 옵티마이저 정의 (가중치 감쇠 적용)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
    early_stopping = EarlyStopping(patience=5, min_delta=0.01)

    # 메트릭 초기화
    metrics = {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}
    num_epochs = 50

    # 타이머 초기화
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (images, labels) in enumerate(train_loader):
            # 150초마다 학습 일시 중지 및 메모리 관리
            current_time = time.time()
            if current_time - start_time >= 99999:
                print("학습을 일시 중지하고 메모리 청소를 수행합니다...")
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(5)
                print("학습을 재개합니다.")
                start_time = time.time()  # 타이머 리셋

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

        # 로그 기록
        log_message = f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, ' \
                      f'Train Accuracy: {train_accuracy:.2f}, Validation Accuracy: {val_accuracy:.2f}'
        logging.info(log_message)
        print(log_message)

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("조기 종료")
            break

    # 모델 저장 및 결과 출력
    if os.path.exists('face_recognition_model.pth'):
        os.remove('face_recognition_model.pth')
    torch.save(model.state_dict(), 'face_recognition_model.pth')
    print("모델 저장 완료: face_recognition_model.pth")
    save_metrics_to_excel(metrics, 'face_recognition_metrics.xlsx')
    plot_metrics(metrics, 'Face Recognition Model Metrics', 'face_recognition_metrics.png')

    # 레이블 매핑 저장
    inverse_label_mapping = {v: k for k, v in label_mapping.items()}
    with open('label_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(inverse_label_mapping, f, ensure_ascii=False)

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
    final_accuracy = 100 * correct / total
    print("테스트 셋에서의 정확도: ", final_accuracy)
    logging.info(f"Final Test Accuracy: {final_accuracy:.2f}")

    # 학습한 레이블 출력
    print("\n학습한 레이블(클래스):")
    for label_idx, class_name in inverse_label_mapping.items():
        print(f"레이블 {label_idx}: {class_name}")

if __name__ == "__main__":
    for filename in ['face_recognition_metrics.xlsx', 'face_recognition_model.pth',
                     'face_recognition_metrics.png']:
        if os.path.exists(filename):
            os.remove(filename)
    # 로그 파일은 삭제하지 않음
    use_dropout_input = input("드롭아웃을 사용하시겠습니까? (yes/no): ").strip().lower()
    use_dropout = use_dropout_input == 'yes'

    target_folders_input = input("학습에 사용할 폴더 이름들을 쉼표로 구분하여 입력하세요 (예: 1404,1405). 'null'을 입력하면 모든 폴더를 사용합니다: ").strip()
    if target_folders_input.lower() == 'null' or not target_folders_input:
        target_folders = None
    else:
        target_folders = [folder.strip() for folder in target_folders_input.split(',')]
    print("얼굴 인식 모델 학습 시작...")
    logging.info("Training started")
    train_face_recognition_model(use_dropout, target_folders=target_folders)
    print("얼굴 인식 모델 학습 완료")
    logging.info("Training completed")