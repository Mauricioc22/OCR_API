from ultralytics import YOLO
from paddleocr import PaddleOCR

# Initialize models
model = YOLO("best.pt")
ocr = PaddleOCR(use_angle_cls=True, lang='en')