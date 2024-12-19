# -*- coding: utf-8 -*-
"""
Oluşturulma Tarihi: 18 Kasım 2024

Kullanılan Kütüphaneler:
- OpenCV
- Mediapipe
- Scikit-Learn

Bu kod işaret dili tanıma işlemi gerçekleştirmektedir. Mediapipe ile ellerin ve pozların tespit edilmesi,
özelliklerin çıkarılması ve makine öğrenimi modelleriyle sınıflandırma yapılması amaçlanmıştır.

Yazar: mertb
"""

# Giriş logları ve uyarılar için ayarlar
import os

os.environ["GLOG_minloglevel"] = "3"  # Mediapipe loglarını sessize alır

import warnings

warnings.filterwarnings("ignore")  # Uyarı mesajlarını bastırır

# Gerekli kütüphaneler
import cv2
import mediapipe as mp
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Mediapipe bileşenlerini başlat
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose


# Çizim fonksiyonu: Tespit edilen el landmarklarını görüntü üzerine çizer
def draw_landmarks(image, results):
    if hand_result.multi_hand_landmarks:  # Eğer el tespit edilmişse
        for hand_landmarks in hand_result.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)


# El ve poz tespiti yapan fonksiyon
def detect_hands_and_pose(img):
    """
    Bir görüntüde elleri ve pozları tespit eder.

    Parametre:
        img: Giriş görüntüsü

    Döndürür:
        hand_result: Tespit edilen el sonuçları
        pose_result: Tespit edilen poz sonuçları
    """

    with mp_hands.Hands(static_image_mode=True) as hands, mp_pose.Pose(static_image_mode=True) as pose:
        # Görüntüyü RGB formatına dönüştür
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hand_result = hands.process(image_rgb)
        pose_result = pose.process(image_rgb)

        return hand_result, pose_result


# Örnek görüntü işlemesi
image = cv2.imread('C:/Users\mertb\PycharmProjects\DjangoProject\SignLanguage\sign_language_3_class\hello1.jpg')
hand_result, pose_result = detect_hands_and_pose(image)

# İlk görüntüyü göster
cv2.imshow('Ilk Resim', image)
cv2.waitKey(0)

# Landmarkları görüntü üzerine çiz
draw_landmarks(image, hand_result)

# İşlenmiş görüntüyü göster
cv2.imshow('Ikinci Resim', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Özellik çıkarım fonksiyonu
def extract_features(hand_result):
    """
    El landmarklarından özellik çıkarır.

    Parametre:
        hand_result: Mediapipe el tespiti sonuçları

    Döndürür:
        features: Çıkarılan özelliklerin listesi
    """
    features = []
    if hand_result.multi_hand_landmarks:  # Eğer el tespit edilmişse
        for hand_landmarks in hand_result.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                features.extend([landmark.x, landmark.y, landmark.z])

    # Eksik özellikleri sıfır ile doldur
    max_landmarks = 21  # Bir eldeki maksimum landmark sayısı
    expected_features = max_landmarks * 3 * 2  # 2 el için maksimum özellik sayısı
    if len(features) < expected_features:
        features.extend([0] * (expected_features - len(features)))

    return features


# Veri kümesi yolu
DATASET_PATH = 'C:/Users\mertb\PycharmProjects\DjangoProject\SignLanguage\sign_language_3_class'


# Dosya adlarından sayıları kaldıran fonksiyon
def remove_numbers(name):
    """Dosya adlarındaki sayıları temizler."""
    name = name.split('.')[0]
    return re.sub(r'\d', '', name)


# Verileri yükleme fonksiyonu
def load_data(dataset_path):
    """
    Görüntüleri yükler ve özellik çıkarır.

    Parametre:
        dataset_path: Veri kümesinin dosya yolu

    Döndürür:
        x: Özellikler
        y: Etiketler
    """
    x = []
    y = []
    classes = os.listdir(dataset_path)  # Veri kümesindeki tüm dosyaları listele
    for image_name in classes:
        image_path = os.path.join(dataset_path, image_name)
        image = cv2.imread(image_path)
        hand_results, _ = detect_hands_and_pose(image)
        features = extract_features(hand_results)
        if features:  # Eğer özellik çıkarılabilmişse
            x.append(features)
            y.append(remove_numbers(image_name))

    return np.array(x), np.array(y)


# Veriyi yükle
x, y = load_data(DATASET_PATH)

# Veri boyutlarını yazdır
print(f"Veri boyutu: {x.shape}")
print(f"Etiket boyutu: {y.shape}")

# Veri setini eğitim ve test olarak ayır
np.random.seed(42)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Rastgele Orman modeli
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_train, y_train)
rf_prediction = rf_model.predict(x_test)
rf_accuracy = accuracy_score(y_test, rf_prediction)

# Destek Vektör Makineleri modeli
svc_model = SVC(random_state=42)
svc_model.fit(x_train, y_train)
svc_prediction = svc_model.predict(x_test)
svc_accuracy = accuracy_score(y_test, svc_prediction)

# Modellerin sonuçlarını karşılaştır
results = {
    "Random Forest": rf_accuracy,
    "Support Vector Classifier": svc_accuracy
}
best_model_name = max(results, key=results.get)
best_model_accuracy = results[best_model_name]

# En iyi modelin tahmin sonuçları
best_model_prediction = rf_prediction if best_model_name == "Random Forest" else svc_prediction
conf_matrix = confusion_matrix(y_test, best_model_prediction)

# Karışıklık matrisi
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title(f'Karışıklık Matrisi - {best_model_name}')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.show()

# Sınıflandırma raporu
class_report = classification_report(y_test, best_model_prediction, zero_division=0)
print(f"{best_model_name} için sınıflandırma raporu: \n{class_report}")

# Doğruluk karşılaştırması
accuracy_comparison = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
print(accuracy_comparison)


# Tahmin fonksiyonu
def predict_sign(image):
    """
    Bir görüntüyü alıp işaret dili sınıfını tahmin eder.

    Parametre:
        image: Giriş görüntüsü yolu

    Döndürür:
        prediction: Tahmin edilen sınıf
    """
    #img = cv2.imread(image)
    hand_results, _ = detect_hands_and_pose(image)
    features = extract_features(hand_results)
    if features:
        prediction = rf_model.predict([features])
        return prediction[0]
    else:
        return "El tespit edilemedi"
