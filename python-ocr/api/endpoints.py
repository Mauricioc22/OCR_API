from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from core.processing import load_image, extract_plate_number, parse_ine_data
from models import model, ocr
import cv2
import re

router = APIRouter()

def process_plate_detection(image):
    """
    Detects license plates in an image using YOLO model.
    
    Args:
        image: OpenCV image array
        
    Returns:
        tuple: (plate_image, confidence) if plate found, (None, None) otherwise
    """

    results = model(image)
    
    for result in results:
        # Get indices where class is 0 (license plate class)
        index_plates = (result.boxes.cls == 0).nonzero(as_tuple=True)[0]
        
        # Process each detected plate
        for idx in index_plates:
            # Get confidence score for this detection
            conf = result.boxes.conf[idx].item()
            
            # Only process detections with confidence > 70%
            if conf > 0.7:
                # Extract bounding box coordinates
                xyxy = result.boxes.xyxy[idx].squeeze().tolist()
                x1, y1 = int(xyxy[0]), int(xyxy[1])
                x2, y2 = int(xyxy[2]), int(xyxy[3])
                
                # Crop plate region with padding for better OCR
                plate_image = image[y1-15:y2+15, x1-15:x2+15]
                return plate_image, round(conf * 100, 2)  # Return confidence as percentage
                
    return None, None

def process_ocr(plate_image):
    """
    Performs OCR on a license plate image and extracts text.
    
    Args:
        plate_image: Cropped image containing the license plate
        
    Returns:
        str: Cleaned text extracted from the plate
    """
    # Convert BGR to RGB for OCR model
    result_ocr = ocr.predict(cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB))
    
    # Extract bounding boxes and text predictions
    boxes = result_ocr[0]['rec_boxes']
    texts = result_ocr[0]['rec_texts']
    
    # Sort text boxes from left to right based on x-coordinate
    left_to_right = sorted(zip(boxes, texts), key=lambda x: min(x[0][::2]))
    
    # Concatenate all text and convert to uppercase
    full_text = ''.join([t for _, t in left_to_right]).upper()
    
    # Remove hyphens and spaces for cleaner output
    full_text = re.sub(r'[-\s]', '', full_text)
    return full_text

def process_text_lines(box_text_pairs):
    """
    Organizes OCR text boxes into logical lines based on vertical position.
    
    Args:
        box_text_pairs: List of tuples containing (bounding_box, text)
        
    Returns:
        list: Ordered list of text strings organized by lines
    """
    # Sort pairs by vertical position (y-coordinate)
    box_text_pairs.sort(key=lambda x: min(x[0][1], x[0][3]))
    
    lines = []
    line_threshold = 15  # Vertical distance threshold for line grouping
    
    # Group text boxes into lines based on vertical proximity
    for box, text in box_text_pairs:
        y = min(box[1], box[3])  # Get minimum y-coordinate of the box
        
        # Check if this belongs to an existing line or starts a new one
        if not lines or abs(y - lines[-1][0]) > line_threshold:
            lines.append([y, [(box, text)]])  # Start new line
        else:
            lines[-1][1].append((box, text))   # Add to current line

    extracted_lines = []
    
    # Sort text within each line from left to right and extract text
    for _, line in lines:
        line.sort(key=lambda x: min(x[0][0], x[0][2]))  # Sort by x-coordinate
        extracted_lines.extend([t for _, t in line])    # Extract text only

    return extracted_lines

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    License plate detection and text extraction endpoint.
    
    Args:
        file: Uploaded image file containing a license plate
        
    Returns:
        JSON response with extracted plate number and confidence score
    """
    try:
        # Read uploaded file content
        contents = await file.read()
        npimg = load_image(contents)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        # Validate image loading
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Detect license plate in the image
        plate_image, confidence = process_plate_detection(image)
        
        if plate_image is not None:
            # Extract text from detected plate
            full_text = process_ocr(plate_image)
            # Clean and validate plate number
            output_text = extract_plate_number(full_text)
            return JSONResponse(content={"plate": output_text, "confidence": confidence})

        # No plate detected
        return JSONResponse(content={"plate": None, "confidence": None})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/extract_text")
async def extract_text(file: UploadFile = File(...)):
    """
    General text extraction endpoint for identity documents (INE, passports, etc.).
    
    Args:
        file: Uploaded image file containing an identity document
        
    Returns:
        JSON response with extracted personal information
    """
    try:
        # Read uploaded file content
        contents = await file.read()
        npimg = load_image(contents)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        # Validate image loading
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Convert image to grayscale to improve contrast for OCR
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Perform OCR on the entire document
        result_ocr = ocr.predict(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB))
        boxes = result_ocr[0]['rec_boxes']
        texts = result_ocr[0]['rec_texts']
        box_text_pairs = list(zip(boxes, texts))

        # Organize text into logical lines
        extracted_lines = process_text_lines(box_text_pairs)
        
        # Parse structured data from organized text lines
        ine_data = parse_ine_data(extracted_lines)

        return JSONResponse(content=ine_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))