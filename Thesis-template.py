import cv2
import torch
import torchvision
from torchvision.transforms import functional as F
from joblib import load
import numpy as np

# Load models
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

svm_gender = load("svm_gender.joblib")
svm_age = load("svm_age.joblib")

FACE_CLASS_IDS = [1] 
#Adjust if ever inaccurate
SCORE_THRESHOLD = 0.7
FACE_SIZE = (128, 128)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (640, 480))
    image_tensor = F.to_tensor(small_frame).to(device)

    with torch.no_grad():
        outputs = model([image_tensor])[0]

    for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
        if label in FACE_CLASS_IDS and score > SCORE_THRESHOLD:
            x1, y1, x2, y2 = box.int().tolist()
            face = small_frame[y1:y2, x1:x2]

            if face.shape[0] > 0 and face.shape[1] > 0:
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, FACE_SIZE).flatten().reshape(1, -1)

                gender_pred = svm_gender.predict(face_resized)[0]
                age_pred = svm_age.predict(face_resized)[0]

                label_text = f"{gender_pred}, {age_pred}"
                cv2.rectangle(small_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(small_frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Faster R-CNN + SVM Face Analyzer", small_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
