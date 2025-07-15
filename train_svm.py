import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from joblib import dump

#Path to the your dataset folder
DATASET_DIR = "dataset"
IMAGE_SIZE = (128, 128)

X = []
gender_labels = []
age_labels = []

#Read and then process each image
for folder in os.listdir(DATASET_DIR):
    path = os.path.join(DATASET_DIR, folder)
    if not os.path.isdir(path):
        continue
    try:
        #foldername splits (supports dash / underscores.)
        gender, age = folder.lower().replace('-', '_').split('_')
    except:
        #debugging purpose
        print(f"Skipping invalid folder name: {folder}")
        continue

    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, IMAGE_SIZE)
        X.append(img.flatten())
        gender_labels.append(gender)
        age_labels.append(age)

#Converts it to NumPy arrays
X = np.array(X)
y_gender = np.array(gender_labels)
y_age = np.array(age_labels)

#Check if dataset is empty, for debugging
if len(X) == 0:
    raise ValueError("No images found. Make sure your dataset/ contains subfolders with valid images. :)")

#Splits the data
X_train, X_test, y_gender_train, y_gender_test = train_test_split(X, y_gender, test_size=0.2, random_state=42)
_, _, y_age_train, y_age_test = train_test_split(X, y_age, test_size=0.2, random_state=42)

#Dynamically determine PCA size to avoid errors
pca_components = min(100, X_train.shape[0], X_train.shape[1])
print(f"[INFO] Using {pca_components} PCA components on {X_train.shape[0]} training samples.")
#Separate the gender and age model
#Gender model
gender_model = make_pipeline(
    PCA(n_components=pca_components),
    StandardScaler(),
    SVC(kernel='linear', probability=True)
)
gender_model.fit(X_train, y_gender_train)

#Age model
age_model = make_pipeline(
    PCA(n_components=pca_components),
    StandardScaler(),
    SVC(kernel='linear', probability=True)
)
age_model.fit(X_train, y_age_train)

#Reports / stats
print("\nGender Classification Report:")
print(classification_report(y_gender_test, gender_model.predict(X_test)))
print("\nAge Classification Report:")
print(classification_report(y_age_test, age_model.predict(X_test)))

#Save the models to be prepared for testing
dump(gender_model, "svm_gender.joblib")
dump(age_model, "svm_age.joblib")
print("\nModels saved as 'svm_gender.joblib' and 'svm_age.joblib'")
