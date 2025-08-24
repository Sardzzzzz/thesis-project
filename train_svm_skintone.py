#Skin Tone SVM Training with ResNet + RGB Color Features + Augmentation
import os
import cv2
import torch
import torchvision
import numpy as np
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from joblib import dump
import random

DATASET_DIR = "dataset_skin"  #separate folder for skin tone dataset
IMAGE_SIZE = (224, 224)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load pretrained ResNet18
resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()
resnet.to(device)

#Transform for images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def augment_image(img):
    #Random brightness, contrast, and small rotation
    if random.random() < 0.5:
        factor = random.uniform(0.8, 1.2)
        img = cv2.convertScaleAbs(img, alpha=factor, beta=0)
    if random.random() < 0.5:
        angle = random.uniform(-10, 10)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        img = cv2.warpAffine(img, M, (w, h))
    return img

def extract_features(img):
    #Convert image to Pytorch tensor and extract ResNet features
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = resnet(img_t).cpu().numpy().flatten()
    #Append the mean RGB as additional features for skin tone
    mean_rgb = img.mean(axis=(0, 1))  # shape (3,)
    return np.hstack([feat, mean_rgb])

#For loading of dataset
X = []
y_skin = []

for folder in os.listdir(DATASET_DIR):
    folder_path = os.path.join(DATASET_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    label = folder.lower()  #dark, light, mid-dark, mid-light

    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #Original image
        features = extract_features(img)
        X.append(features)
        y_skin.append(label)

        #Augmented image (optional, can add 1–2 per original)
        for _ in range(2):  # 2 augmentations per original
            aug_img = augment_image(img.copy())
            features_aug = extract_features(aug_img)
            X.append(features_aug)
            y_skin.append(label)

X = np.array(X)
y_skin = np.array(y_skin)

if len(X) == 0:
    raise ValueError("No images found. Check your dataset_skin folder.")

#Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_skin, test_size=0.2, stratify=y_skin, random_state=42
)

#SVM PIPELINE
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=300)),
    ("svm", SVC(C=10, gamma="scale", kernel="rbf", class_weight="balanced"))
])

#Train
model = pipeline.fit(X_train, y_train)

#Report
y_pred = model.predict(X_test)
print("Skin Tone Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#Save model as .joblib
dump(model, "svm_skin.joblib")
print("Skin tone model saved as 'svm_skin.joblib'.")
