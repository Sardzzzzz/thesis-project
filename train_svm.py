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

#Configurations
DATASET_DIR = "dataset"
IMAGE_SIZE = (128, 128)
LBP_RADIUS = 2
LBP_POINTS = 8 * LBP_RADIUS

X = []
gender_labels = []
age_labels = []


def extract_features(image):
    #LBP
    lbp = local_binary_pattern(image, LBP_POINTS, LBP_RADIUS, method="uniform")
    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, LBP_POINTS + 3),
        range=(0, LBP_POINTS + 2),
    )
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    #HOG
    hog_feat = hog(
        image,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    )
    if np.linalg.norm(hog_feat) > 0:
        hog_feat = hog_feat / np.linalg.norm(hog_feat)

    return np.hstack([lbp_hist, hog_feat]).astype("float32")


#Augmentation part
def augment_image(img):
    augmented = [img]

    #Flip
    augmented.append(cv2.flip(img, 1))

    #Small rotations
    for angle in [-5, 5]:
        M = cv2.getRotationMatrix2D((IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2), angle, 1)
        rotated = cv2.warpAffine(img, M, IMAGE_SIZE)
        augmented.append(rotated)

    #Light brightness adjustments
    for alpha in [0.9, 1.1]:
        bright = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
        augmented.append(bright)

    return augmented


#Load dataset folder
for folder in tqdm(os.listdir(DATASET_DIR)):
    path = os.path.join(DATASET_DIR, folder)
    if not os.path.isdir(path):
        continue

    try:
        age, gender = folder.lower().replace("-", "_").split("_")
    except Exception:
        print(f"Skipping invalid folder name: {folder}")
        continue

    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, IMAGE_SIZE)

        for img_variant in augment_image(img):
            X.append(extract_features(img_variant))
            gender_labels.append(gender)
            age_labels.append(age)

X = np.array(X)
y_gender = np.array(gender_labels)
y_age = np.array(age_labels)

if len(X) == 0:
    raise ValueError("No images found in dataset!")


#Split train/test
y_combined = np.array([f"{a}_{g}" for a, g in zip(y_age, y_gender)])
X_train, X_test, y_gender_train, y_gender_test, y_age_train, y_age_test = train_test_split(
    X,
    y_gender,
    y_age,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y_combined,
)

#Gender model
print("Training gender model")
gender_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=100)),
    (
        "svm",
        SVC(
            C=1,
            gamma="scale",
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=42,
        ),
    ),
])
gender_pipeline.fit(X_train, y_gender_train)

#Age model
print("Training age model")
age_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=100)),
    (
        "svm",
        SVC(
            C=1,
            gamma="scale",
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=42,
        ),
    ),
])
age_pipeline.fit(X_train, y_age_train)

#Evaluate results
print("\nGender Model:")
y_pred_gender = gender_pipeline.predict(X_test)
print(classification_report(y_gender_test, y_pred_gender))
print("Accuracy:", accuracy_score(y_gender_test, y_pred_gender))
print("Confusion Matrix:\n", confusion_matrix(y_gender_test, y_pred_gender))

print("\nAge Model:")
y_pred_age = age_pipeline.predict(X_test)
print(classification_report(y_age_test, y_pred_age))
print("Accuracy:", accuracy_score(y_age_test, y_pred_age))
print("Confusion Matrix:\n", confusion_matrix(y_age_test, y_pred_age))

#Save the models
dump(gender_pipeline, "svm_gender.joblib")
dump(age_pipeline, "svm_age.joblib")
print("\nModels saved successfully!")
