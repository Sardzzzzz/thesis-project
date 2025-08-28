#Improved live classification, FASTER-RCNN focuses on the person (this includes the body), added haar cascades to focus more on the face when it comes to bounding boxes.
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
import imagehash
from skimage.feature import local_binary_pattern, hog

#Load the models trained by train_svm.py
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

svm_gender = load("svm_gender.joblib")
svm_age = load("svm_age.joblib")
svm_skin = load("svm_skin.joblib")

#Haar cascade to focus on the face, rather than the body which reduces classification accuracy
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#Constants
FACE_CLASS_IDS = [1]
SCORE_THRESHOLD = 0.7
FACE_SIZE = (128, 128)
LBP_RADIUS = 2
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = 'uniform'
AD_REFRESH_INTERVAL = 2000
#Change if needed
FRAME_CONFIRMATION_COUNT = 5
OPT_IN_TIMEOUT = 3
saved_faces_dir = 'saved_faces'
os.makedirs(saved_faces_dir, exist_ok=True)

#Grace period for losing face, helps with the smoothness of bounding box
FACE_LOST_GRACE = 1.0  # seconds

#Shared state
current_ad_category = ["idle"]
ad_lock = threading.Lock()
recent_predictions = []
opt_in_given = False
saved_face_hashes = set()
opted_in_hash = None
tracked_box = None  #stores (center_x, center_y, width, height)
last_valid_face_time = time.time()
last_face_seen_time = time.time()  #for grace period

#Initialize the saved faces only once as much as possible
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
    
    last_displayed_category = ["idle"] #Track last category show

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
threading.Thread(target=show_ad_window, daemon=True).start()

#Consent/opt-in prompt
def get_user_consent():
    global opt_in_given
    consent_win = tk.Tk()
    consent_win.title("Consent for opt-in!")
    tk.Label(consent_win, text="By clicking Agree, you allow access to the camera for age/gender/skin detection. Targeted Advertisement :)", wraplength=300).pack(pady=10)

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

cap = None

while True:
    if not opt_in_given:
        current_ad_category[0] = "idle"
        cv2.imshow("SmartTarget", idle_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        get_user_consent()
        cap = cv2.VideoCapture(0)
        tracked_box = None
        last_valid_face_time = time.time()
        last_face_seen_time = time.time()
        opted_in_hash = None
        continue

    ret, frame = cap.read()
    if not ret:
        continue

    small_frame = cv2.resize(frame, (640, 480))
    image_tensor = F.to_tensor(small_frame).to(device)

    with torch.no_grad():
        outputs = model([image_tensor])[0]

    detected_persons = []
    for box, label_id, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
        if label_id in FACE_CLASS_IDS and score > SCORE_THRESHOLD:
            detected_persons.append(box.int().tolist())

    face_found = None
    for person_box in detected_persons:
        px1, py1, px2, py2 = person_box
        gray_crop = cv2.cvtColor(small_frame[py1:py2, px1:px2], cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_crop, scaleFactor=1.05, minNeighbors=5, minSize=(50,50))

        for (fx, fy, fw, fh) in faces:
            pad_w, pad_h = int(fw*0.3), int(fh*0.3)
            fx1 = max(px1 + fx - pad_w, 0)
            fy1 = max(py1 + fy - pad_h, 0)
            fx2 = min(px1 + fx + fw + pad_w, small_frame.shape[1])
            fy2 = min(py1 + fy + fh + pad_h, small_frame.shape[0])
            face_crop = small_frame[fy1:fy2, fx1:fx2]
            face_gray = cv2.resize(cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY), FACE_SIZE)
            candidate_hash = hash_face(face_gray)

            #Track only the original opted-in person
            if opted_in_hash is None:
                opted_in_hash = candidate_hash
                face_found = (fx1, fy1, fx2, fy2)
                last_valid_face_time = time.time()
                break
            else:
                hash_diff = imagehash.hex_to_hash(candidate_hash) - imagehash.hex_to_hash(opted_in_hash)
                if hash_diff <= 10:
                    face_found = (fx1, fy1, fx2, fy2)
                    last_valid_face_time = time.time()
                    break
        if face_found:
            break

    #Smooth tracking with grace period
    if face_found:
        fx1, fy1, fx2, fy2 = face_found
        f_cx = (fx1 + fx2) / 2
        f_cy = (fy1 + fy2) / 2
        f_w = fx2 - fx1
        f_h = fy2 - fy1

        if tracked_box is None:
            tracked_box = (f_cx, f_cy, f_w, f_h)
        else:
            cx, cy, w, h = tracked_box
            alpha = 0.7
            cx = alpha * cx + (1 - alpha) * f_cx
            cy = alpha * cy + (1 - alpha) * f_cy
            w = alpha * w + (1 - alpha) * f_w
            h = alpha * h + (1 - alpha) * f_h
            tracked_box = (cx, cy, w, h)

        last_face_seen_time = time.time()  #reset grace timer
    else:
        #Keep box for grace period before removing
        if tracked_box is not None and (time.time() - last_face_seen_time > FACE_LOST_GRACE):
            tracked_box = None
            current_ad_category[0] = "idle"

    #Compute display_box normally
    display_box = None
    if tracked_box is not None:
        cx, cy, w, h = tracked_box
        x1 = int(cx - w/2)
        y1 = int(cy - h/2)
        x2 = int(cx + w/2)
        y2 = int(cy + h/2)
        display_box = (x1, y1, x2, y2)
    else:
        display_box = None

    #Timeout
    if time.time() - last_valid_face_time > OPT_IN_TIMEOUT:
        opt_in_given = False
        tracked_box = None
        if cap:
            cap.release()
            cap = None
        continue

     #Proceed with classification on tracked face only
    if display_box:
        x1, y1, x2, y2 = display_box
        face = small_frame[y1:y2, x1:x2]
        if face.shape[0] > 0 and face.shape[1] > 0:
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_gray, FACE_SIZE)
            features = extract_features(face_resized)

            gender_pred = svm_gender.predict(features)[0]
            age_pred = svm_age.predict(features)[0]
            skin_pred = svm_skin.predict(features)[0]

            label = f"{age_pred.lower()}_{gender_pred.lower()}_{skin_pred.lower()}"
            label_text = f"{gender_pred}, {age_pred}, {skin_pred}"

            face_hash = hash_face(face_resized)
            category_folder = os.path.join(saved_faces_dir, label)
            os.makedirs(category_folder, exist_ok=True)

            if all(imagehash.hex_to_hash(face_hash) - imagehash.hex_to_hash(eh) > 5 for eh in saved_face_hashes):
                cv2.imwrite(os.path.join(category_folder, f"{face_hash}.jpg"), face)
                saved_face_hashes.add(face_hash)

            recent_predictions.append(label)
             #Update ad immediately when consecutive predictions match
            if len(recent_predictions) > FRAME_CONFIRMATION_COUNT:
                recent_predictions.pop(0)

            if len(recent_predictions) == FRAME_CONFIRMATION_COUNT and all(x == recent_predictions[0] for x in recent_predictions):
                with ad_lock:
                    current_ad_category[0] = recent_predictions[0]

            cv2.rectangle(small_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(small_frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("SmartTarget", small_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if cap:
    cap.release()
cv2.destroyAllWindows()
