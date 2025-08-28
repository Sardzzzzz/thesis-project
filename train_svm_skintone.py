# Skin Tone SVM Training (Just separated)

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from joblib import dump
from skimage.feature import local_binary_pattern, hog
from tqdm import tqdm

#Configs
DATASET_DIR = "dataset_skin"   #folder structure: dataset_skin/dark, dataset_skin/light
IMAGE_SIZE = (128, 128)
LBP_RADIUS = 2
LBP_POINTS = 8 * LBP_RADIUS

X = []
skin_labels = []

#Feature extraction
def extract_features(image):
    #LBP
    lbp = local_binary_pattern(image, LBP_POINTS, LBP_RADIUS, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3),
                               range=(0, LBP_POINTS + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    #HOG
    hog_feat = hog(image,
                   orientations=9,
                   pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2),
                   block_norm='L2-Hys',
                   transform_sqrt=True,
                   feature_vector=True)
    if np.linalg.norm(hog_feat) > 0:
        hog_feat = hog_feat / np.linalg.norm(hog_feat)

    return np.hstack([lbp_hist, hog_feat]).astype("float32")

#Augment images
def augment_image(img):
    augmented = [img]

    #Preprocess flips
    augmented.append(cv2.flip(img, 1))

    #Preprocess rotations
    for angle in [-5, 5]:
        M = cv2.getRotationMatrix2D((IMAGE_SIZE[0]//2, IMAGE_SIZE[1]//2), angle, 1)
        rotated = cv2.warpAffine(img, M, IMAGE_SIZE)
        augmented.append(rotated)

    #For brightness adjustment
    for alpha in [0.9, 1.1]:
        bright = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
        augmented.append(bright)

    return augmented

#Load dataset_skin
for folder in tqdm(os.listdir(DATASET_DIR)):
    path = os.path.join(DATASET_DIR, folder)
    if not os.path.isdir(path):
        continue

    label = folder.lower()  #"dark" or "light"

    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, IMAGE_SIZE)

        for img_variant in augment_image(img):
            X.append(extract_features(img_variant))
            skin_labels.append(label)

X = np.array(X)
y_skin = np.array(skin_labels)

if len(X) == 0:
    raise ValueError("No images found in dataset_skin!")

#Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_skin, test_size=0.2, random_state=42, shuffle=True, stratify=y_skin
)

#SVM Pipeline
skin_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=100)),
    ('svm', SVC(C=1, gamma='scale', kernel='rbf',
                class_weight='balanced', probability=True, random_state=42))
])
skin_pipeline.fit(X_train, y_train)

#Evaluate results
print("\nSkin Tone Model:")
y_pred_skin = skin_pipeline.predict(X_test)
print(classification_report(y_test, y_pred_skin))
print("Accuracy:", accuracy_score(y_test, y_pred_skin))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_skin))

#Save to svm_skin.joblib
dump(skin_pipeline, "svm_skin.joblib")
print("\nSkin tone model saved successfully!")
