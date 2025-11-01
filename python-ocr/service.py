from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import numpy as np
import re
import imutils

app = FastAPI()

# Inicializar modelos solo una vez
model = YOLO("best.pt")
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def extract_plate_number(text):
    blacklist_words = ['VOLKSWAGEN', 'TOYOTA', 'NISSAN', 'CHEVROLET', 'FORD', 'HONDA', 'HYUNDAI', 
                     'KIA', 'MAZDA', 'SUZUKI', 'MITSUBISHI', 'SUBARU', 'BMW', 'MERCEDES', 'AUDI',
                     'MEXICO', 'MEX', 'ESTADO', 'MUNICIPIO', 'PLACA', 'PLATE']
    plate_patterns = [
        r'[A-Z]{3}[0-9]{3}',
        r'[0-9]{3}[A-Z]{3}',
        r'[A-Z]{2}[0-9]{4}',
        r'[0-9]{2}[A-Z]{4}',
        r'[A-Z]{1}[0-9]{2}[A-Z]{3}',
        r'[0-9]{3}[A-Z]{2}[0-9]{1}'
    ]
    for pattern in plate_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            is_valid = True
            for word in blacklist_words:
                if word in match:
                    is_valid = False
                    break
            if is_valid and 5 <= len(match) <= 8:
                return match
    clean_text = re.sub(r'[^A-Z0-9]', '', text)
    for word in blacklist_words:
        clean_text = clean_text.replace(word, '')
    if 5 <= len(clean_text) <= 8:
        return clean_text
    if len(clean_text) > 8:
        for i in range(len(clean_text) - 5):
            candidate = clean_text[i:i+6]
            if re.search(r'[A-Z]', candidate) and re.search(r'[0-9]', candidate):
                return candidate
        return clean_text[:7]
    return clean_text

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        npimg = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        results = model(image)
        for result in results:
            index_plates = (result.boxes.cls == 0).nonzero(as_tuple=True)[0]
            for idx in index_plates:
                conf = result.boxes.conf[idx].item()
                if conf > 0.7:
                    xyxy = result.boxes.xyxy[idx].squeeze().tolist()
                    x1, y1 = int(xyxy[0]), int(xyxy[1])
                    x2, y2 = int(xyxy[2]), int(xyxy[3])
                    plate_image = image[y1-15:y2+15, x1-15:x2+15]
                    result_ocr = ocr.predict(cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB))
                    boxes = result_ocr[0]['rec_boxes']
                    texts = result_ocr[0]['rec_texts']
                    left_to_right = sorted(zip(boxes, texts), key=lambda x: min(x[0][::2]))
                    full_text = ''.join([t for _, t in left_to_right]).upper()
                    full_text = re.sub(r'[-\s]', '', full_text)
                    output_text = extract_plate_number(full_text)
                    return JSONResponse(content={"plate": output_text})
        return JSONResponse(content={"plate": None})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
