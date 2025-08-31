import os
import time
import random
import threading
from flask import Flask, Response, jsonify, send_file, request
from flask_cors import CORS
import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F
from joblib import load
from PIL import Image
import imagehash
from skimage.feature import local_binary_pattern, hog
import smtplib
from email.mime.text import MIMEText
import base64
from PIL import Image
import io
import psycopg2
from dotenv import load_dotenv
from datetime import date
load_dotenv()

# ---------------- Config / Toggles ----------------
USE_RESNET_FOR_AGE_GENDER = False
FACE_CLASS_IDS = [1]
SCORE_THRESHOLD = 0.7
FACE_SIZE = (128, 128)
LBP_RADIUS = 2
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = 'uniform'
FRAME_CONFIRMATION_COUNT = 5
HASH_DISTANCE_TOL = 5
OPT_IN_TIMEOUT = 3.0
FACE_LOST_GRACE = 1.0

# ---------------- Paths ----------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = PROJECT_DIR
ADS_DIR = os.path.join(MODEL_DIR, "public/ads")
SAVED_FACES_DIR = os.path.join(MODEL_DIR, "saved_faces")
os.makedirs(SAVED_FACES_DIR, exist_ok=True)

# ---------------- Flask ----------------
app = Flask(__name__)
url = os.getenv("DATABASE_URL")
connection = psycopg2.connect(url)
CORS(app)

# ---------------- Shared State ----------------
current_attributes = {"age": "Unknown", "gender": "Unknown", "skin": "Unknown"}
current_ad_category = ["idle"]
ad_lock = threading.Lock()
recent_predictions = []
saved_face_hashes = set()
saved_locked_categories = set()  # Track saved faces per locked category
opted_in_hash = None
tracked_box = None
last_valid_face_time = time.time()
last_face_seen_time = time.time()
locked_category = None
user_email = None
email_lock = threading.Lock()
GOOGLE_FORM_LINK = "https://docs.google.com/forms/d/e/1FAIpQLSdYgqFi0wrjeXkCe2rYGsTKG8LqXshUX3dQjjM5MDNLVTrq6A/viewform"
EMAIL_SEND_DELAY = 2.0

# ---------------- Email Credentials ----------------
DEL_EMAIL = "induadvertisementsystem@gmail.com"  # <- replace with your email
PASSWORD = "oeqrhjmkfpzrvvbp"  # <- replace with your app password

camera = cv2.VideoCapture(0)
latest_frame = None
latest_frame_lock = threading.Lock()

# ---------------- Models ----------------
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

svm_gender = None
svm_age = None
svm_skin = None

def try_load_joblib(name):
    path = os.path.join(MODEL_DIR, name)
    if os.path.exists(path):
        try:
            return load(path)
        except Exception as e:
            print(f"Error loading {path}:", e)
            return None
    else:
        print(f"File not found: {path}")
        return None

svm_gender = try_load_joblib("svm_gender.joblib")
svm_age    = try_load_joblib("svm_age.joblib")
svm_skin   = try_load_joblib("svm_skin.joblib")
print("Loaded SVMs:", svm_gender is not None, svm_age is not None, svm_skin is not None)

svm_gender_resnet = None
svm_age_resnet = None
try:
    path_g = os.path.join(MODEL_DIR, "svm_gender_resnet.joblib")
    path_a = os.path.join(MODEL_DIR, "svm_age_resnet.joblib")
    if os.path.exists(path_g) and os.path.exists(path_a):
        svm_gender_resnet = load(path_g)
        svm_age_resnet = load(path_a)
        USE_RESNET_FOR_AGE_GENDER = True
except Exception as e:
    print("ResNet SVMs not loaded:", e)
    USE_RESNET_FOR_AGE_GENDER = False

resnet_skin = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
resnet_skin = torch.nn.Sequential(*list(resnet_skin.children())[:-1])
resnet_skin.eval()
resnet_skin.to(device)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ---------------- Init saved faces ----------------
for category in os.listdir(SAVED_FACES_DIR):
    category_dir = os.path.join(SAVED_FACES_DIR, category)
    if os.path.isdir(category_dir):
        for fname in os.listdir(category_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                saved_face_hashes.add(fname.split('.')[0])

# ---------------- Preload ad images for instant delivery ----------------
preloaded_ads = {}
for cat in os.listdir(ADS_DIR):
    cat_dir = os.path.join(ADS_DIR, cat)
    if os.path.isdir(cat_dir):
        files = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        preloaded_ads[cat.lower()] = files
# Ensure idle exists
if 'idle' not in preloaded_ads:
    preloaded_ads['idle'] = [os.path.join(ADS_DIR, 'idle', 'indu.png')]

# ---------------- Utils ----------------
def hash_face(face_array):
    try:
        pil_img = Image.fromarray(face_array)
        return str(imagehash.average_hash(pil_img))
    except Exception as e:
        print("hash_face error:", e)
        return str(random.getrandbits(64))

def extract_lbp_hog(face_gray):
    lbp = local_binary_pattern(face_gray, P=16, R=2, method="uniform")
    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, 16 + 3),
        range=(0, 16 + 2)
    )
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    hog_feat = hog(
        face_gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    )
    hog_feat /= (np.linalg.norm(hog_feat) + 1e-6)

    return np.hstack([lbp_hist, hog_feat]).astype("float32").reshape(1, -1)

def resnet_embed(face_bgr_224):
    face_rgb = cv2.cvtColor(face_bgr_224, cv2.COLOR_BGR2RGB)
    tensor = F.to_tensor(face_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = resnet_skin(tensor).cpu().numpy().reshape(-1)
    return feat

def classify_face(face_bgr):
    age_pred, gender_pred, skin_pred = "Unknown", "Unknown", "Unknown"
    try:
        if USE_RESNET_FOR_AGE_GENDER and svm_gender_resnet and svm_age_resnet:
            face_224 = cv2.resize(face_bgr, (224, 224))
            emb = resnet_embed(face_224).reshape(1, -1)
            try:
                gender_pred = svm_gender_resnet.predict(emb)[0]
                age_pred = svm_age_resnet.predict(emb)[0]
            except Exception:
                gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
                gray_resized = cv2.resize(gray, FACE_SIZE)
                feats = extract_lbp_hog(gray_resized)
                if svm_gender: gender_pred = svm_gender.predict(feats)[0]
                if svm_age: age_pred = svm_age.predict(feats)[0]
        else:
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            gray_resized = cv2.resize(gray, FACE_SIZE)
            feats = extract_lbp_hog(gray_resized)
            if svm_gender: gender_pred = svm_gender.predict(feats)[0]
            if svm_age: age_pred = svm_age.predict(feats)[0]

        if svm_skin:
            face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            feats_skin = extract_lbp_hog(cv2.resize(face_gray, FACE_SIZE))
            skin_pred = svm_skin.predict(feats_skin)[0]
    except Exception as e:
        print("Classification error:", e)
    return age_pred, gender_pred, skin_pred

# ---------------- Detection Thread ----------------
def face_detection_loop():
    global opted_in_hash, recent_predictions, current_ad_category, tracked_box
    global last_valid_face_time, last_face_seen_time, camera, latest_frame, locked_category
    alpha = 0.7

    while True:
        ret, frame = camera.read()
        if not ret:
            time.sleep(0.01)
            continue

        small_frame = cv2.resize(frame, (640, 480))
        image_tensor = F.to_tensor(small_frame).to(device)

        with torch.no_grad():
            outputs = model([image_tensor])[0]

        detected_persons = []
        for box, label_id, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
            if int(label_id) in FACE_CLASS_IDS and float(score) > SCORE_THRESHOLD:
                x1, y1, x2, y2 = box.int().tolist()
                x1 = max(0, min(x1, small_frame.shape[1]-1))
                x2 = max(0, min(x2, small_frame.shape[1]-1))
                y1 = max(0, min(y1, small_frame.shape[0]-1))
                y2 = max(0, min(y2, small_frame.shape[0]-1))
                if x2 > x1 and y2 > y1:
                    detected_persons.append((x1, y1, x2, y2))

        face_found = None
        for (px1, py1, px2, py2) in detected_persons:
            person_crop = small_frame[py1:py2, px1:px2]
            if person_crop.size == 0:
                continue
            gray_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_crop, scaleFactor=1.05, minNeighbors=5, minSize=(50, 50))
            if len(faces) == 0:
                continue

            fx, fy, fw, fh = faces[0]
            pad_w, pad_h = int(fw * 0.3), int(fh * 0.3)
            fx1 = max(px1 + fx - pad_w, 0)
            fy1 = max(py1 + fy - pad_h, 0)
            fx2 = min(px1 + fx + fw + pad_w, small_frame.shape[1])
            fy2 = min(py1 + fy + fh + pad_h, small_frame.shape[0])

            face_crop = small_frame[fy1:fy2, fx1:fx2]
            try:
                face_gray_for_hash = cv2.resize(cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY), FACE_SIZE)
            except Exception:
                continue

            candidate_hash = hash_face(face_gray_for_hash)

            if opted_in_hash is None:
                opted_in_hash = candidate_hash
                face_found = (fx1, fy1, fx2, fy2)
                last_valid_face_time = time.time()
                break
            else:
                try:
                    hash_diff = imagehash.hex_to_hash(candidate_hash) - imagehash.hex_to_hash(opted_in_hash)
                    if hash_diff <= HASH_DISTANCE_TOL:
                        face_found = (fx1, fy1, fx2, fy2)
                        last_valid_face_time = time.time()
                        break
                except Exception:
                    pass

        if face_found:
            fx1, fy1, fx2, fy2 = face_found
            f_cx = (fx1 + fx2) / 2.0
            f_cy = (fy1 + fy2) / 2.0
            f_w = float(fx2 - fx1)
            f_h = float(fy2 - fy1)

            if tracked_box is None:
                tracked_box = (f_cx, f_cy, f_w, f_h)
            else:
                cx, cy, w, h = tracked_box
                cx = alpha * cx + (1 - alpha) * f_cx
                cy = alpha * cy + (1 - alpha) * f_cy
                w = alpha * w + (1 - alpha) * f_w
                h = alpha * h + (1 - alpha) * f_h
                tracked_box = (cx, cy, w, h)

            last_face_seen_time = time.time()

        if tracked_box is not None:
            cx, cy, w, h = tracked_box
            x1 = int(max(0, cx - w / 2.0))
            y1 = int(max(0, cy - h / 2.0))
            x2 = int(min(small_frame.shape[1] - 1, cx + w / 2.0))
            y2 = int(min(small_frame.shape[0] - 1, cy + h / 2.0))

            if x2 > x1 and y2 > y1:
                face = small_frame[y1:y2, x1:x2]
                if face.size != 0:
                    age_pred, gender_pred, skin_pred = classify_face(face)
                    if not skin_pred or skin_pred.strip().lower() in ["", "unknown", "none"]:
                        skin_pred = "Unclassified"

                    current_attributes["age"] = age_pred
                    current_attributes["gender"] = gender_pred
                    current_attributes["skin"] = skin_pred

                    label = f"{age_pred.lower()}_{gender_pred.lower()}_{skin_pred.lower()}"

                    # Track recent predictions
                    recent_predictions.append(label)
                    if len(recent_predictions) > FRAME_CONFIRMATION_COUNT:
                        recent_predictions.pop(0)

                    # Lock category and save face only once
                    if len(recent_predictions) == FRAME_CONFIRMATION_COUNT and all(x == recent_predictions[0] for x in recent_predictions):
                        majority = recent_predictions[0]
                        with ad_lock:
                            if locked_category is None:
                                locked_category = majority
                                current_ad_category[0] = majority
                                

                                # Save face only once per locked category
                                if locked_category not in saved_locked_categories:
                                    try:
                                        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                                        face_resized = cv2.resize(face_gray, FACE_SIZE)
                                        face_phash = hash_face(face_resized)
                                        category_folder = os.path.join(SAVED_FACES_DIR, locked_category)
                                        os.makedirs(category_folder, exist_ok=True)
                                        save_path = os.path.join(category_folder, f"{face_phash}.jpg")
                                        cv2.imwrite(save_path, face)
                                        saved_face_hashes.add(face_phash)
                                        saved_locked_categories.add(locked_category)
                                        print(f"Saved face for locked category: {locked_category}")
                                    except Exception as e:
                                        print("Save face error:", e)

        with latest_frame_lock:
            latest_frame = small_frame.copy()

        time.sleep(0.005)

threading.Thread(target=face_detection_loop, daemon=True).start()

# ---------------- Email Utils ----------------
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

def send_google_form_email_with_all_images(to_email, category):
    """
    Sends email with all images in the category folder, showing inline preview with name and price,
    and attaches full-size images.
    """
    try:
        category_dir = None
        for folder in os.listdir(ADS_DIR):
            if folder.lower() == category.lower():
                category_dir = os.path.join(ADS_DIR, folder)
                break
        if category_dir is None or not os.path.exists(category_dir):
            category_dir = os.path.join(ADS_DIR, "idle")

        # Get all images
        img_files = [f for f in os.listdir(category_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

        # Create email
        msg = MIMEMultipart()
        msg['Subject'] = f"Confirm Your Ad Preferences - {category}"
        msg['From'] = DEL_EMAIL
        msg['To'] = to_email

        # Inline previews with name and price
        img_html = ""
        for fname in img_files:
            path = os.path.join(category_dir, fname)
            name, price = "Unknown", "Unknown"
            try:
                # Regex parse filename: name_price.ext
                import re
                match = re.match(r"(.+)_([\d.]+)$", fname.rsplit('.', 1)[0])
                if match:
                    name = match.group(1).replace("_", " ")
                    price = f"${float(match.group(2)):.2f}"
                else:
                    name = fname.rsplit('.', 1)[0].replace("_", " ")
                    price = "Unknown"
            except Exception as e:
                print("Filename parse error:", e)

            try:
                with Image.open(path) as img:
                    img.thumbnail((200, 200))
                    buf = io.BytesIO()
                    img.convert("RGB").save(buf, format="JPEG")
                    encoded = base64.b64encode(buf.getvalue()).decode()
                    img_html += f"""
                    <div style="margin:5px; text-align:center;">
                        <img src="data:image/jpeg;base64,{encoded}" />
                        <br><b>{name}</b>
                        <br>{price}
                    </div>
                    """
            except Exception as e:
                print("Inline image preview error:", e)

        # HTML body
        body = f"""
        <html>
            <body>
                <p>Hello!</p>
                <p>Thank you for confirming your ad preference: <b>{category}</b>.</p>
                <p>Please take a moment to fill out our form by clicking the link below:</p>
                <p><a href="{GOOGLE_FORM_LINK}" target="_blank">Click here to open the form</a></p>
                <p>Here are all images related to your category:</p>
                <div style="display:flex; flex-wrap: wrap;">{img_html}</div>
                <p>For full resolution images, please check the attachments.</p>
                <p>We really appreciate your feedback! 😊</p>
                <p>Best regards,<br>Indu Targeted Advertisement Research Group</p>
            </body>
        </html>
        """
        msg.attach(MIMEText(body, 'html'))

        # Attach full-size images
        for fname in img_files:
            path = os.path.join(category_dir, fname)
            with open(path, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename="{fname}"')
                msg.attach(part)

        # Send email
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(DEL_EMAIL, PASSWORD)
            server.send_message(msg)

        print(f"Email successfully sent to {to_email} with all images and inline previews.")

    except Exception as e:
        print(f"Email send error: {e}")

# ---------------- Routes ----------------
@app.route('/attributes', methods=['GET'])
def get_attributes():
    return jsonify(current_attributes)

@app.route('/ad-category', methods=['GET'])
def ad_category_route():
    with ad_lock:
        return jsonify({"category": locked_category if locked_category else current_ad_category[0]})

@app.route('/reset', methods=['POST'])
def reset_route():
    global opted_in_hash, tracked_box, recent_predictions, current_attributes, locked_category, saved_locked_categories
    opted_in_hash = None
    tracked_box = None
    recent_predictions = []
    current_attributes = {"age": "Unknown", "gender": "Unknown", "skin": "Unknown"}
    with ad_lock:
        current_ad_category[0] = "idle"
        locked_category = None
        saved_locked_categories = set()
    return jsonify({"ok": True})

@app.route("/submit-email", methods=["POST"])
def submit_email():
    global user_email
    data = request.json
    email = data.get("email")
    if not email:
        return jsonify({"status": "error", "message": "Email not provided"}), 400
    with email_lock:
        user_email = email
    return jsonify({"status": "pending_confirmation", "email": email})

@app.route("/confirm-email", methods=["POST"])
def confirm_email():
    global user_email
    with email_lock:
        if not user_email:
            return jsonify({"status": "error", "message": "No email to send"}), 400
        
        category = locked_category if locked_category else current_ad_category[0]
        recipient = user_email

    def send_email_thread(to_email, category):
        try:
            print(f"[EMAIL] Sending email to {to_email} for category '{category}'...")
            send_google_form_email_with_all_images(to_email, category)
            print(f"[EMAIL] Successfully sent to {to_email}")
        except Exception as e:
            print(f"[EMAIL ERROR] Failed to send email to {to_email}: {e}")

    # Start the email in a separate daemon thread
    threading.Thread(target=send_email_thread, args=(recipient, category), daemon=True).start()

    # Immediately return to API caller without waiting for email
    return jsonify({"status": "email_queued", "email": recipient, "category": category})

# Record Agree and Disagree terms and services
#Records visit for that certain demographic, WHEN PRESSED YES  = RECORD
@app.route("/confirm-visit", methods=["POST"])
def confirm_visit():
    try:
        data = request.get_json()
        category = data.get("category")

        if not category:
            return jsonify({"error": "No category provided"}), 400

        parts = category.split("_")
        if len(parts) != 3:
            return jsonify({"error": "Invalid category format"}), 400

        age, gender, skin = parts

        with connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO demographics (age_range, skin_color, gender)
                    VALUES (%s, %s, %s)
                    """,
                    (age, skin, gender)
                )

        return jsonify({"message": "Visit recorded successfully."}), 200

    except Exception as e:
        print(f"Error inserting visit: {e}")
        return jsonify({"error": "Failed to record visit"}), 500
    
@app.post("/api/consent")
def insert_consent():
    data = request.get_json()
    choice = data.get("choice")  # Expecting "agree" or "disagree"

    if choice not in ["agree", "disagree"]:
        return jsonify({"error": "Invalid choice, must be 'agree' or 'disagree'"}), 400

    with connection:
        with connection.cursor() as cursor:
            if choice == "agree":
                cursor.execute("""
                    INSERT INTO consent_trends (date, agree, disagree)
                    VALUES (CURRENT_DATE, 1, 0)
                    ON CONFLICT (date) DO UPDATE
                    SET agree = consent_trends.agree + 1;
                """)
            else:
                cursor.execute("""
                    INSERT INTO consent_trends (date, agree, disagree)
                    VALUES (CURRENT_DATE, 0, 1)
                    ON CONFLICT (date) DO UPDATE
                    SET disagree = consent_trends.disagree + 1;
                """)

    return jsonify({"message": f"Recorded {choice}"})

# Fetch Consent
@app.route('/api/consent-latest', methods=['GET'])
def get_consent_trends():
    with connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT SUM(agree) AS total_agree, SUM(disagree) AS total_disagree
                FROM consent_trends;
            """)
            result = cursor.fetchone()
            total_agree = result[0] or 0
            total_disagree = result[1] or 0

    # Format response for Recharts PieChart
    data = [
        {"name": "Agree", "value": total_agree},
        {"name": "Disagree", "value": total_disagree}
    ]

    return {"data": data, "message": "Consent trends fetched successfully"}, 200

@app.route("/api/consent-trend-daily", methods=["GET"])
def get_daily_consent_trends():
    with connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT date, agree, disagree
                FROM consent_trends
                ORDER BY date ASC;
            """)
            rows = cursor.fetchall()

    data = [{"date": str(row[0]), "agree": row[1], "disagree": row[2]} for row in rows]
    return {"data": data, "message": "Daily consent trends fetched successfully"}, 200

# Insert Feedback
@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    data = request.get_json()
    email = data.get("email")
    message = data.get("message")

    # Validate input
    if not email or not message or not message.strip():
        return jsonify({"error": "Email and feedback are required"}), 400

    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO feedback (email, message, date)
                    VALUES (%s, %s, %s)
                    RETURNING id
                    """,
                    (email, message, date.today())
                )
                feedback_id = cursor.fetchone()[0]

        return jsonify({"id": feedback_id, "message": "Feedback submitted!"}), 201

    except Exception as e:
        print("Error submitting feedback:", e)  # Log error to console
        return jsonify({"error": "Internal server error"}), 500

@app.route("/status", methods=['GET'])
def status():
    return jsonify({"email": user_email, "category": current_ad_category[0]})

@app.route("/reset-user", methods=["POST"])
def reset_user():
    global user_email
    with email_lock:
        user_email = None
    return jsonify({"status": "reset"})

@app.route('/video_feed')
def video_feed():
    def gen_frames():
        global latest_frame
        while True:
            with latest_frame_lock:
                frame = latest_frame.copy() if latest_frame is not None else None
            if frame is None:
                idle = np.zeros((480, 640, 3), dtype=np.uint8)
                _, buffer = cv2.imencode('.jpg', idle)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.05)
                continue
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.03)
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

#--POSTGRESQL QUERIES--

INSERT_DEMOGRAPHIC_RECORD = ("""
    INSERT INTO demographics (age_range, skin_color, gender)
    VALUES (%s, %s, %s)
    RETURNING id;
    """
)
INSERT_SALE_RECORD = """
INSERT INTO sales (product, price, demographic, used, date)
VALUES (%s, %s, %s, %s, %s)
RETURNING id;
"""


FETCH_SALES = ("SELECT * FROM sales;")

FETCH_DEMOGRAPHICS = (""" 
    SELECT age_range || '-' || skin_color || '-' || gender AS demographic,
    COUNT(*) AS total
    FROM demographics
    GROUP BY age_range, skin_color, gender;
""")

FETCH_FEEDBACK = "SELECT email, message, date FROM feedback;"
#--API ENDPOINT FOR MAIN SYSTEM--
#INSERT DEMOGRAPHIC endpoint
@app.post('/api/demographic')
def createDemographic():
    data = request.get_json()
    age_range = data["age_range"]
    skin_color = data["skin_color"]
    gender = data["gender"]

    with connection:
        with connection.cursor() as cursor:
            cursor.execute(INSERT_DEMOGRAPHIC_RECORD,(age_range, skin_color, gender))
            demographicID = cursor.fetchone()[0]

    return {"id": demographicID, "message": "Demographic data recorded"}, 201

#--API ENDPOINTS FOR ADMIN-- 
@app.route('/', methods=['GET'])
def home():
    return "Home page"

#Fetch DEMOGRAHPICS endpoint
@app.get('/api/get-demographic')
def getDemographics():
    with connection:
        with connection.cursor() as cursor:
            cursor.execute(FETCH_DEMOGRAPHICS)
            rows = cursor.fetchall()

    demographics = [
        {
            "demographic": row[0],
            "total": row[1]
        }
        for row in rows
    ]
    return jsonify(demographics)

#Record Sales Endpoint
@app.post('/api/sales')
def createSale():
    data = request.get_json()
    product = data["product"]
    price = data["price"]
    demographic = data["demographic"]
    used = data["used"]
    date = data["date"]
    
    with connection:
        with connection.cursor() as cursor:
            cursor.execute(INSERT_SALE_RECORD, (product, price, demographic, used, date))
            saleID = cursor.fetchone()[0]

    return jsonify({"id": saleID, "message": f"Product sold on {date}"}), 201


#Fetch Sales records Endpoint
@app.get('/api/get-sales')
def getSales():
    with connection:
        with connection.cursor() as cursor:
            cursor.execute(FETCH_SALES)
            rows = cursor.fetchall()
    sales = [
        {   
            "id": row[0],
            "product": row[1],
            "price": float(row[2]),
            "demographic": row[3],
            "used": row[4],
            "date": row[5].isoformat()
        }
        for row in rows
    ]
    return jsonify(sales)

#Fetch FEEDBACK endpoint
@app.get('/api/get-feedback')
def getFeedback():
    with connection:
        with connection.cursor() as cursor:
            cursor.execute(FETCH_FEEDBACK)
            rows = cursor.fetchall()
    feedback = [
        {
            "email": row[0],
            "message": row[1],
            "date": row[2].isoformat()
        }
        for row in rows
    ]
    return jsonify(feedback)

# ---------------- Preload Ads ----------------
preloaded_ads = {}

def preload_ads():
    global preloaded_ads
    categories = os.listdir(ADS_DIR)
    for cat in categories:
        cat_dir = os.path.join(ADS_DIR, cat)
        if os.path.isdir(cat_dir):
            preloaded_ads[cat.lower()] = []
            for fname in os.listdir(cat_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    path = os.path.join(cat_dir, fname)
                    try:
                        with open(path, 'rb') as f:
                            preloaded_ads[cat.lower()].append((fname, f.read()))
                    except:
                        continue
    # Fallback idle ad
    idle_path = os.path.join(ADS_DIR, "idle/indu.png")
    with open(idle_path, 'rb') as f:
        preloaded_ads["idle"] = [("indu.png", f.read())]

preload_ads()

@app.route('/ad-image')
def ad_image():
    with ad_lock:
        category = locked_category if locked_category else current_ad_category[0]
    cat = category.lower()
    if cat not in preloaded_ads or len(preloaded_ads[cat]) == 0:
        cat = "idle"
    fname, img_bytes = random.choice(preloaded_ads[cat])
    mime = 'image/webp' if fname.lower().endswith('.webp') else 'image/jpeg'
    return Response(img_bytes, mimetype=mime)

# ---------------- Main ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
