import cv2
import torch
import torchvision
from torchvision.transforms import functional as F

# Load pretrained Faster R-CNN
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Define a function to filter for face-like classes
FACE_CLASS_IDS = [1]  # COCO label 1 = 'person', a proxy for face here
SCORE_THRESHOLD = 0.7  # confidence threshold

# Start video capture
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for better FPS (optional)
    small_frame = cv2.resize(frame, (640, 480))

    # Convert frame to tensor
    image_tensor = F.to_tensor(small_frame).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model([image_tensor])[0]

    for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
        if label in FACE_CLASS_IDS and score > SCORE_THRESHOLD:
            x1, y1, x2, y2 = box.int().tolist()
            cv2.rectangle(small_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("SmartTarget - Faster R-CNN Face Detection", small_frame)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
