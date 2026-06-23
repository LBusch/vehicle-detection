import torch
import cv2
import os
from models import load_pretrained_fasterrcnn
from torchvision.transforms import v2

IMG_SEQUENCE_DIR = "data/DETRAC-Images/MVI_40981"
MODEL_PATH = os.path.join("frcnn_default", "best_model.pt")
VIDEO_NAME = "video_inference.avi"
SCORE_THRESHOLD = 0.4
        
def draw_predictions(frame, predictions):
    '''Draws predicted bounding boxes and the corresponding scores'''
    boxes = predictions["boxes"]
    scores = predictions["scores"]
    keep_idx = scores >= SCORE_THRESHOLD
    boxes = boxes[keep_idx]
    scores = scores[keep_idx]

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box.int().tolist()
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        cv2.putText(frame, f"{score:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

test_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

model = load_pretrained_fasterrcnn(num_classes=2, weights_path=MODEL_PATH)
images = sorted([img for img in os.listdir(IMG_SEQUENCE_DIR) if img.endswith('.jpg')])
frame = cv2.imread(os.path.join(IMG_SEQUENCE_DIR, images[0]))
height, width, _ = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(VIDEO_NAME, fourcc, 30.0, (width, height))

# go through each frame in the image sequence and pass it to the model
for img_name in images:
    img_path = os.path.join(IMG_SEQUENCE_DIR, img_name)
    frame = cv2.imread(img_path)
    predictions = model([test_transforms(frame)])
    draw_predictions(frame, predictions[0])
    video_writer.write(frame)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_writer.release()
cv2.destroyAllWindows()