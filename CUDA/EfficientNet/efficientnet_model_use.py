import json
from collections import Counter, deque

import cv2
import requests
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def load_model_and_labels(model_path, label_path):
    with open(label_path, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    num_features = model.classifier[1].in_features
    saved_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    saved_num_classes = saved_state_dict['classifier.1.1.weight'].size(0)
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, saved_num_classes)
    )
    model.load_state_dict(saved_state_dict)
    model.eval()
    return model, {int(k): v for k, v in label_mapping.items()}

def preprocess_face(frame, face_coords, transform):
    x, y, w, h = face_coords
    face = frame[y:y + h, x:x + w]
    if face is None or face.size == 0:
        return None
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = Image.fromarray(face).resize((224, 224))
    face = transform(face)
    return face.unsqueeze(0)

def combine_results(results):
    combined = Counter(results)
    return combined.most_common(1)[0][0]

def send_checkin(student_id):
    url = "https://amond-server.kro.kr/dev-server/daram/checkin"
    data = {"studentId": student_id}
    try:
        response = requests.patch(url, json=data)
        if response.status_code == 200:
            print(f"Check-in 성공: {student_id}")
        else:
            print(f"Check-in 실패: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Check-in 요청 중 오류 발생: {e}")

def real_time_face_recognition(model, label_mapping):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    mtcnn = MTCNN(keep_all=True, device=device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 기본 카메라 사용
    recent_results = deque(maxlen=60)
    processed_students = set()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다. 카메라 연결을 확인하세요.")
                continue

            boxes, _ = mtcnn.detect(frame)
            results = []

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    face_coords = (x1, y1, x2 - x1, y2 - y1)
                    face_tensor = preprocess_face(frame, face_coords, transform)

                    if face_tensor is None:
                        continue

                    face_tensor = face_tensor.to(device)
                    with torch.no_grad():
                        output = model(face_tensor)
                        _, predicted = torch.max(output, 1)
                        results.append(predicted.item())

                    label = label_mapping[predicted.item()]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if results:
                final_result = combine_results(results)
                recent_results.append(final_result)
                if len(recent_results) == 30 and recent_results.count(final_result) == 30:
                    student_id = int(label_mapping[final_result])
                    if student_id not in processed_students:
                        send_checkin(student_id)
                        processed_students.add(student_id)
                        recent_results.clear()
                print(f"최종 결과: {label_mapping[final_result]}" if final_result in label_mapping else f"최종 결과: Unknown ({final_result})")

            # API 요청된 인물들 상태 출력용 윈도우 생성
            processed_queue = "Processed: " + ", ".join([label_mapping.get(sid, f"Unknown ({sid})") for sid in processed_students])
            queue_frame = 255 * np.ones((200, 800, 3), dtype=np.uint8)
            cv2.putText(queue_frame, processed_queue, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.imshow("Processed Queue", queue_frame)

            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = "face_recognition_model.pth"
    label_path = "label_mapping.json"
    model, label_mapping = load_model_and_labels(model_path, label_path)
    real_time_face_recognition(model, label_mapping)
