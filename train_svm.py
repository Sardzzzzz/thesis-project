import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump
from skimage.feature import local_binary_pattern, hog

DATASET_DIR = "dataset"
IMAGE_SIZE = (128, 128)
LBP_RADIUS = 2
LBP_POINTS = 8 * LBP_RADIUS

X = []
gender_labels = []
age_labels = []

def extract_features(image):
    #LBP Histogram
    lbp = local_binary_pattern(image, LBP_POINTS, LBP_RADIUS, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), range=(0, LBP_POINTS + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    #HOG
    hog_feat = hog(
        image,
        orientations=9,
        pixels_per_cell=(16, 16),  #coarser > fewer features
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        transform_sqrt=True,
        feature_vector=True
    )
    if np.linalg.norm(hog_feat) > 0:
        hog_feat = hog_feat / np.linalg.norm(hog_feat)

    return np.hstack([lbp_hist, hog_feat]).astype("float32")

#Load dataset
for folder in os.listdir(DATASET_DIR):
    path = os.path.join(DATASET_DIR, folder)
    if not os.path.isdir(path):
        continue

    try:
        age, gender = folder.lower().replace('-', '_').split('_')
    except Exception:
        print(f"Skipping invalid folder name: {folder}")
        continue

    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, IMAGE_SIZE)

        #Original + horizontal flip
        for img_variant in [img, cv2.flip(img, 1)]:
            X.append(extract_features(img_variant))
            gender_labels.append(gender)
            age_labels.append(age)

#Convert to arrays
X = np.array(X)
y_gender = np.array(gender_labels)
y_age = np.array(age_labels)

if len(X) == 0:
    raise ValueError("No images found in dataset!")

#Split train/test dataset.
y_combined = np.array([f"{a}_{g}" for a, g in zip(y_age, y_gender)])

X_train, X_test, y_gender_train, y_gender_test, y_age_train, y_age_test = train_test_split(
    X, y_gender, y_age, test_size=0.2, random_state=42, shuffle=True, stratify=y_combined
)

#Section for gender model
gender_model = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=200)),  #fixed component cap to reduce memory
    ('svm', SVC(kernel='rbf', C=5, gamma='scale', class_weight='balanced', probability=True))
])

#Section for age model
age_model = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=200)),
    ('svm', SVC(kernel='rbf', C=5, gamma='scale', class_weight='balanced', probability=True))
])

#Train the models
print("[INFO] Training gender model...")
gender_model.fit(X_train, y_gender_train)

print("[INFO] Training age model...")
age_model.fit(X_train, y_age_train)

#Print the reports
print("\nGender Classification Report:")
y_pred_gender = gender_model.predict(X_test)
print(classification_report(y_gender_test, y_pred_gender))
print("Accuracy:", accuracy_score(y_gender_test, y_pred_gender))

print("\nAge Classification Report:")
y_pred_age = age_model.predict(X_test)
print(classification_report(y_age_test, y_pred_age))
print("Accuracy:", accuracy_score(y_age_test, y_pred_age))


#Save the models
dump(gender_model, "svm_gender.joblib")
dump(age_model, "svm_age.joblib")
print("\nModels saved successfully!")
