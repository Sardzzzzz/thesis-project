#Added opt-in, black window when not agreeing with the consent, close camera after 10 seconds of no face detection (
#might change this since I think there will be a lot of faces detected.), added window for advertisement and camera, working dynamically advertisements ( will change ).

"""
NEW AD CATEGORIES
inside ads/ 
teen_male_dark/
teen_male_light/
teen_male_mid-dark/
teen_male_mid-light/
teen_female_dark/ 
adult_female_mid-light/
ETC.
idle/
"""
import cv2
import torch
import torchvision
from torchvision.transforms import functional as F
from joblib import load
import numpy as np
import os
import random
import threading
from PIL import Image, ImageTk, ImageSequence
import tkinter as tk
import time
import imagehash  #Added for perceptual hashing
from skimage.feature import local_binary_pattern, hog

#Load the models trained by train_svm.py
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

svm_gender = load("svm_gender.joblib")
svm_age = load("svm_age.joblib")
svm_skin = load("svm_skin.joblib")  #Skin SVM

#Load ResNet for skin feature extraction
resnet_skin = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
resnet_skin = torch.nn.Sequential(*list(resnet_skin.children())[:-1])
resnet_skin.eval()
resnet_skin.to(device)

#Constants
FACE_CLASS_IDS = [1]
SCORE_THRESHOLD = 0.7
FACE_SIZE = (128, 128)
LBP_RADIUS = 2
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = 'uniform'
AD_REFRESH_INTERVAL = 2000
FRAME_CONFIRMATION_COUNT = 3
OPT_IN_TIMEOUT = 10
saved_faces_dir = 'saved_faces'
os.makedirs(saved_faces_dir, exist_ok=True)

#Shared State
current_ad_category = ["idle"]
ad_lock = threading.Lock()
recent_predictions = []
opt_in_given = False
last_face_time = 0
saved_face_hashes = set()

#Initialize the saved faces only once as much as possible.
for category in os.listdir(saved_faces_dir):
    category_dir = os.path.join(saved_faces_dir, category)
    if os.path.isdir(category_dir):
        for fname in os.listdir(category_dir):
            if fname.endswith('.jpg'):
                saved_face_hashes.add(fname.split('.')[0])

#Hash function part, updated with perceptual hash
def hash_face(face_array):
    pil_img = Image.fromarray(face_array)
    return str(imagehash.average_hash(pil_img))

#Idle frame
idle_image = np.zeros((480, 640, 3), dtype=np.uint8)

#Ad window which used Tkinter
def show_ad_window():
    ad_win = tk.Tk()
    ad_win.title("Advertisements")
    ad_label = tk.Label(ad_win)
    ad_label.pack()

    idle_gif_path = os.path.join("ads", "idle", "idle.gif")
    idle_gif_frames = []
    idle_gif_index = 0

    if os.path.exists(idle_gif_path):
        idle_gif = Image.open(idle_gif_path)
        idle_gif_frames = [ImageTk.PhotoImage(f.copy().resize((400, 400))) for f in ImageSequence.Iterator(idle_gif)]

    last_displayed_category = ["idle"]  #Track last category shown

    def update_ad():
        nonlocal idle_gif_index
        with ad_lock:
            category = current_ad_category[0]

        #Only update if category changed or idle gif frame changed
        if category != last_displayed_category[0] or category == "idle":
            folder_path = os.path.join("ads", category)
            ad_path = None

            if category == "idle" and idle_gif_frames:
                ad_label.config(image=idle_gif_frames[idle_gif_index], text="")
                ad_label.image = idle_gif_frames[idle_gif_index]
                idle_gif_index = (idle_gif_index + 1) % len(idle_gif_frames)
            elif os.path.exists(folder_path):
                images = [f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                if images:
                    selected_image = random.choice(images)
                    ad_path = os.path.join(folder_path, selected_image)
                if ad_path and os.path.exists(ad_path):
                    image = Image.open(ad_path).resize((400, 400))
                    photo = ImageTk.PhotoImage(image)
                    ad_label.config(image=photo, text="")
                    ad_label.image = photo
                else:
                    ad_label.config(text="No ad available", image='', font=("Arial", 24))
            last_displayed_category[0] = category

        ad_win.after(AD_REFRESH_INTERVAL, update_ad)

    update_ad()
    ad_win.mainloop()

#This starts the ad thread dynamics
ad_thread = threading.Thread(target=show_ad_window, daemon=True)
ad_thread.start()

#Consent/opt-in prompt
def get_user_consent():
    global opt_in_given
    consent_win = tk.Tk()
    consent_win.title("Consent for opt-in!")
    label = tk.Label(consent_win, text="By clicking Agree, you allow access to the camera for age/gender detection. Targeted Advertisement :)", wraplength=300)
    label.pack(pady=10)

    def agree():
        global opt_in_given
        opt_in_given = True
        consent_win.destroy()

    tk.Button(consent_win, text="Agree", command=agree).pack(pady=5)
    consent_win.mainloop()

#Feature extraction for SVM
def extract_features(face_gray):
    #LBP
    lbp = local_binary_pattern(face_gray, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), range=(0, LBP_POINTS + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    
    #HOG
    hog_feat = hog(face_gray, orientations=9, pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True,
                   feature_vector=True)
    if np.linalg.norm(hog_feat) > 0:
        hog_feat = hog_feat / np.linalg.norm(hog_feat)
    
    return np.hstack([lbp_hist, hog_feat]).astype("float32").reshape(1, -1)

#This is the main loop of the program
cap = None
while True:
    if not opt_in_given:
        current_ad_category[0] = "idle"
        cv2.imshow("SmartTarget", idle_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        get_user_consent()
        last_face_time = time.time()
        cap = cv2.VideoCapture(0)
        continue

    ret, frame = cap.read()
    if not ret:
        continue

    small_frame = cv2.resize(frame, (640, 480))
    image_tensor = F.to_tensor(small_frame).to(device)

    with torch.no_grad():
        outputs = model([image_tensor])[0]

    #Select only the largest face (closest)
    best_box = None
    best_area = 0
    for box, label_id, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
        if label_id in FACE_CLASS_IDS and score > SCORE_THRESHOLD:
            x1, y1, x2, y2 = box.int().tolist()
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_box = (x1, y1, x2, y2)

    if best_box:
        last_face_time = time.time()
        x1, y1, x2, y2 = best_box
        face = small_frame[y1:y2, x1:x2]

        if face.shape[0] > 0 and face.shape[1] > 0:
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_gray, FACE_SIZE)
            features = extract_features(face_resized)
            gender_pred = svm_gender.predict(features)[0]
            age_pred = svm_age.predict(features)[0]

            #Skin tone prediction
            face_rgb_resized = cv2.resize(face, (224, 224))
            face_tensor = torchvision.transforms.functional.to_tensor(face_rgb_resized).unsqueeze(0).to(device)
            with torch.no_grad():
                resnet_feat = resnet_skin(face_tensor).cpu().numpy().flatten()
            mean_rgb = face_rgb_resized.mean(axis=(0, 1))
            skin_features = np.hstack([resnet_feat, mean_rgb]).reshape(1, -1)
            skin_pred = svm_skin.predict(skin_features)[0]

            label = f"{age_pred.lower()}_{gender_pred.lower()}"  #teen/adult + male/female
            label_text = f"{gender_pred}, {age_pred}, {skin_pred}"  #display with skin

            face_hash = hash_face(face_resized)
            category_folder = os.path.join(saved_faces_dir, label)
            os.makedirs(category_folder, exist_ok=True)

            #Check if already saved (using perceptual hashing)
            already_exists = False
            for existing_hash in saved_face_hashes:
                if imagehash.hex_to_hash(face_hash) - imagehash.hex_to_hash(existing_hash) <= 5:
                    already_exists = True
                    break

            #Only save if not already saved
            if not already_exists:
                save_path = os.path.join(category_folder, f"{face_hash}.jpg")
                cv2.imwrite(save_path, face)
                saved_face_hashes.add(face_hash)

            recent_predictions.append(label)
            if len(recent_predictions) > FRAME_CONFIRMATION_COUNT:
                recent_predictions.pop(0)

            #Update ad immediately when consecutive predictions match
            if len(recent_predictions) == FRAME_CONFIRMATION_COUNT and all(x == recent_predictions[0] for x in recent_predictions):
                with ad_lock:
                    current_ad_category[0] = recent_predictions[0]

            cv2.rectangle(small_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(small_frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        if time.time() - last_face_time > OPT_IN_TIMEOUT:
            opt_in_given = False
            cap.release()
            cap = None
            continue

    cv2.imshow("SmartTarget", small_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if cap:
    cap.release()
cv2.destroyAllWindows()
