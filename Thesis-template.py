#Added opt-in, black window when not agreeing with the consent, close camera after 10 seconds of no face detection (
#might change this since I think there will be a lot of faces detected.), added window for advertisement and camera, working dynamically advertisements ( will change ).

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

#Load the models trained by train_svm.py
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

svm_gender = load("svm_gender.joblib")
svm_age = load("svm_age.joblib")

#Constants
FACE_CLASS_IDS = [1]
SCORE_THRESHOLD = 0.7
FACE_SIZE = (128, 128)
AD_REFRESH_INTERVAL = 2000
FRAME_CONFIRMATION_COUNT = 3
OPT_IN_TIMEOUT = 10

#Shared State
current_ad_category = ["idle"]
ad_lock = threading.Lock()
recent_predictions = []
opt_in_given = False
last_face_time = 0

#Idle frame
idle_frame_path = os.path.join("ads", "idle")
idle_image = cv2.imread(idle_frame_path)
if idle_image is not None:
    idle_image = cv2.resize(idle_image, (640, 480))
else:
    idle_image = np.zeros((480, 640, 3), dtype=np.uint8)

#Ad window which used Tkinter
def show_ad_window():
    ad_win = tk.Tk()
    ad_win.title("Advertisements")
    ad_label = tk.Label(ad_win)
    ad_label.pack()

    idle_gif_path = os.path.join("ads", "idle", "idle.gif")
    idle_gif = None
    idle_gif_frames = []
    idle_gif_index = 0

    if os.path.exists(idle_gif_path):
        idle_gif = Image.open(idle_gif_path)
        idle_gif_frames = [ImageTk.PhotoImage(f.copy().resize((400, 400))) for f in ImageSequence.Iterator(idle_gif)]

    def update_ad():
        nonlocal idle_gif_index
        with ad_lock:
            category = current_ad_category[0]

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

    agree_btn = tk.Button(consent_win, text="Agree", command=agree)
    agree_btn.pack(pady=5)
    consent_win.mainloop()

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

    best_box = None
    best_area = 0

    for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
        if label in FACE_CLASS_IDS and score > SCORE_THRESHOLD:
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
            face_resized = cv2.resize(face_gray, FACE_SIZE).flatten().reshape(1, -1)

            gender_pred = svm_gender.predict(face_resized)[0]
            age_pred = svm_age.predict(face_resized)[0]
            label = f"{gender_pred.lower()}_{age_pred.lower()}"

            recent_predictions.append(label)
            if len(recent_predictions) > FRAME_CONFIRMATION_COUNT:
                recent_predictions.pop(0)

            if recent_predictions.count(recent_predictions[0]) == FRAME_CONFIRMATION_COUNT:
                with ad_lock:
                    if current_ad_category[0] != recent_predictions[0]:
                        current_ad_category[0] = recent_predictions[0]

            label_text = f"{gender_pred}, {age_pred}"
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
