import os
import cv2
import numpy as np
import joblib
import dlib
from tensorflow.keras.models import load_model

# 얼굴 및 랜드마크 모델
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # 사전 학습된 모델 필요

# 볼 좌표 인덱스 (dlib 68 landmarks 기준)
CHEEK_IDXS = {
    "left_cheek": [1, 2, 3, 4, 31],
    "right_cheek": [15, 14, 13, 12, 35],
}

# 볼색 평균 HSV 추출 함수
def cheek_color_avg(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detections = detector(gray, 0)
    cheek_colors = []

    for d in detections:
        shape = predictor(gray, d)

        for name, indices in CHEEK_IDXS.items():
            pts = np.array([[shape.part(j).x, shape.part(j).y] for j in indices], np.int32)
            center_x, center_y = np.mean(pts, axis=0).astype(np.int32)
            half_size = 5
            x1, y1 = max(0, center_x - half_size), max(0, center_y - half_size)
            x2, y2 = min(img.shape[1], center_x + half_size), min(img.shape[0], center_y + half_size)

            cheek_region = img[y1:y2, x1:x2]
            if cheek_region.size == 0:
                continue
            hsv_cheek_region = cv2.cvtColor(cheek_region, cv2.COLOR_BGR2HSV)
            mean_color = np.mean(hsv_cheek_region, axis=(0, 1)).astype(np.uint8)
            cheek_colors.append(mean_color)

        if cheek_colors:
            avg_color = np.mean(cheek_colors, axis=0).astype(np.uint8)
            return avg_color

    return None

# 이미지 1장 전처리 + 모델 예측
def predict_personal_color(image_path, model_path="personal_color_model.h5", scaler_path="scaler.pkl"):
    image = cv2.imread(image_path)
    if image is None:
        print("이미지를 불러올 수 없습니다.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print("얼굴이 감지되지 않았습니다.")
        return

    for i, (x, y, w, h) in enumerate(faces):
        cropped_face = image[y:y+h, x:x+w]
        avg_hsv = cheek_color_avg(cropped_face)

        if avg_hsv is None:
            print("볼 색상을 추출하지 못했습니다.")
            return

        # 스케일러 로드 및 변환
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform([avg_hsv])

        # 모델 로드 및 예측
        model = load_model(model_path)
        pred = model.predict(X_scaled)
        label_map = {0: "spring", 1: "summer", 2: "autumn", 3: "winter"}
        predicted_label = label_map[np.argmax(pred)]

        print(f"예측된 퍼스널 컬러: {predicted_label}")
        return predicted_label
    


predict_personal_color("test_img.jpg")

