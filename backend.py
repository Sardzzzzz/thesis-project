import cv2
import torch
import torchvision
from torchvision.transforms import functional as F
from joblib import load
import numpy as np
import threading
import time
import os
from flask import Flask, Response, request, jsonify, send_file
from flask_cors import CORS

# --- Flask Setup ---
app = Flask(__name__)
CORS(app)

# --- Model Loading ---
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

svm_gender = load("svm_gender.joblib")
svm_age = load("svm_age.joblib")

# --- Constants ---
FACE_CLASS_IDS = [1]
SCORE_THRESHOLD = 0.7
FACE_SIZE = (128, 128)
FRAME_CONFIRMATION_COUNT = 3
OPT_IN_TIMEOUT = 10

# --- Shared State ---
current_ad_category = ["idle"]
recent_predictions = []
last_face_time = time.time()
consent = False
camera = cv2.VideoCapture(0)

# --- Face Detection Thread ---
def face_detection_loop():
    global last_face_time, consent
    while True:
        if not consent:
            time.sleep(1)
            continue

        ret, frame = camera.read()
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
                    if current_ad_category[0] != recent_predictions[0]:
                        current_ad_category[0] = recent_predictions[0]
        else:
            # No face detected
            if time.time() - last_face_time > OPT_IN_TIMEOUT:
                consent = False
                current_ad_category[0] = "idle"

# Start background detection
threading.Thread(target=face_detection_loop, daemon=True).start()

# --- Flask Routes ---

@app.route('/consent', methods=['GET', 'POST'])
def consent_route():
    global consent
    if request.method == 'POST':
        data = request.get_json()
        consent = bool(data.get('consent', False))
        if not consent:
            current_ad_category[0] = "idle"
        return jsonify({"success": True, "consent": consent})
    return jsonify({"consent": consent})

@app.route('/ad-category', methods=['GET'])
def ad_category_route():
    return jsonify({"category": current_ad_category[0]})

@app.route('/video_feed')
def video_feed():
    def gen_frames():
        while True:
            if not consent:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                ret, frame = camera.read()
                if not ret:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ad-image')
def ad_image():
    # Get the absolute path to the ads directory
    ads_base = os.path.abspath(os.path.join(os.path.dirname(__file__), 'ads'))
    category = current_ad_category[0]
    ads_dir = os.path.join(ads_base, category)
    if not os.path.exists(ads_dir):
        # fallback to idle ad
        ads_dir = os.path.join(ads_base, 'idle')
    images = [f for f in os.listdir(ads_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    if not images:
        # fallback to idle ad if no images in category
        ads_dir = os.path.join(ads_base, 'idle')
        images = [f for f in os.listdir(ads_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        if not images:
            return '', 404
    img_path = os.path.join(ads_dir, images[0])  # always serve the first image (not random)
    return send_file(img_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(port=5000, debug=True, use_reloader=False)
