from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from core.processing import load_image, extract_plate_number, parse_ine_data
from models import model, ocr
import cv2
import re

router = APIRouter()

def process_plate_detection(image):
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
                return plate_image, round(conf * 100, 2)  # Return confidence as percentage truncated to 2 decimals
    return None, None

def process_ocr(plate_image):
    result_ocr = ocr.predict(cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB))
    boxes = result_ocr[0]['rec_boxes']
    texts = result_ocr[0]['rec_texts']
    left_to_right = sorted(zip(boxes, texts), key=lambda x: min(x[0][::2]))
    full_text = ''.join([t for _, t in left_to_right]).upper()
    full_text = re.sub(r'[-\s]', '', full_text)
    return full_text

def process_text_lines(box_text_pairs):
    box_text_pairs.sort(key=lambda x: min(x[0][1], x[0][3]))
    lines = []
    line_threshold = 15
    for box, text in box_text_pairs:
        y = min(box[1], box[3])
        if not lines or abs(y - lines[-1][0]) > line_threshold:
            lines.append([y, [(box, text)]])
        else:
            lines[-1][1].append((box, text))

    extracted_lines = []
    for _, line in lines:
        line.sort(key=lambda x: min(x[0][0], x[0][2]))
        extracted_lines.extend([t for _, t in line])

    return extracted_lines

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        npimg = load_image(contents)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        plate_image, confidence = process_plate_detection(image)
        if plate_image is not None:
            full_text = process_ocr(plate_image)
            output_text = extract_plate_number(full_text)
            return JSONResponse(content={"plate": output_text, "confidence": confidence})

        return JSONResponse(content={"plate": None, "confidence": None})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/extract_text")
async def extract_text(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        npimg = load_image(contents)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # OCR
        result_ocr = ocr.predict(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        boxes = result_ocr[0]['rec_boxes']
        texts = result_ocr[0]['rec_texts']
        box_text_pairs = list(zip(boxes, texts))

        extracted_lines = process_text_lines(box_text_pairs)
        ine_data = parse_ine_data(extracted_lines)

        return JSONResponse(content=ine_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))