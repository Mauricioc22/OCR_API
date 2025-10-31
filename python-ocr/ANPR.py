# Importar librerías necesarias
from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import imutils
import re

# Cargar imagen de entrada
image = cv2.imread("./Inputs/image_001.jpg")

# Inicializar modelos
model = YOLO("best.pt") # Modelo YOLO entrenado para detectar placas vehiculares
ocr = PaddleOCR(use_angle_cls=True, lang='en') # OCR con corrección de inclinación

# Ejecutar YOLO sobre la imagen
results = model(image)
#print(results[0].boxes)

for result in results:
    # Filtrar solo las detecciones de clase "placa" (cls == 0)
    index_plates = (result.boxes.cls == 0).nonzero(as_tuple=True)[0]
    #print(index_plates)

    for idx in index_plates:
        # Obtener confianza de la caja
        conf = result.boxes.conf[idx].item()
        if conf > 0.7:
            # Obtener las coordenadas de la caja
            xyxy = result.boxes.xyxy[idx].squeeze().tolist()
            x1, y1 = int(xyxy[0]), int(xyxy[1])
            x2, y2 = int(xyxy[2]), int(xyxy[3])
            
            # Recortar imagen de la placa con padding
            plate_image = image[y1-15:y2+15, x1-15:x2+15]

            # Ejecutar OCR con PaddleOCR
            result_ocr = ocr.predict(cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB))
            #print(result_ocr)

            # Ordenar los textos detectados de izquierda a derecha
            boxes = result_ocr[0]['rec_boxes']
            texts = result_ocr[0]['rec_texts']
            left_to_right = sorted(zip(boxes, texts), key=lambda x: min(x[0][::2]))
            print(f"left_to_right:", left_to_right)
            
            # Concatenar todo el texto detectado
            full_text = ''.join([t for _, t in left_to_right]).upper()
            # Limpiar caracteres especiales comunes como guiones
            full_text = re.sub(r'[-\s]', '', full_text)
            print(f"Texto completo detectado: {full_text}")            # Función para extraer el número de placa
            def extract_plate_number(text):
                # Palabras comunes que aparecen en placas pero no son el número
                blacklist_words = ['VOLKSWAGEN', 'TOYOTA', 'NISSAN', 'CHEVROLET', 'FORD', 'HONDA', 'HYUNDAI', 
                                 'KIA', 'MAZDA', 'SUZUKI', 'MITSUBISHI', 'SUBARU', 'BMW', 'MERCEDES', 'AUDI',
                                 'MEXICO', 'MEX', 'ESTADO', 'MUNICIPIO', 'PLACA', 'PLATE']
                
                # Patrones comunes de placas mexicanas
                # Formato: 3 letras + 3 números (ABC123) o 3 números + 3 letras (123ABC)
                plate_patterns = [
                    r'[A-Z]{3}[0-9]{3}',  # ABC123
                    r'[0-9]{3}[A-Z]{3}',  # 123ABC
                    r'[A-Z]{2}[0-9]{4}',  # AB1234
                    r'[0-9]{2}[A-Z]{4}',  # 12ABCD
                    r'[A-Z]{1}[0-9]{2}[A-Z]{3}',  # A12BCD
                    r'[0-9]{3}[A-Z]{2}[0-9]{1}'   # 123AB1
                ]
                
                # Buscar patrones de placa en el texto
                for pattern in plate_patterns:
                    matches = re.findall(pattern, text)
                    for match in matches:
                        # Verificar que no contenga palabras de la blacklist
                        is_valid = True
                        for word in blacklist_words:
                            if word in match:
                                is_valid = False
                                break
                        if is_valid and len(match) >= 5 and len(match) <= 8:
                            return match
                
                # Si no encuentra un patrón específico, intenta extraer la secuencia más probable
                # Filtrar solo letras y números
                clean_text = re.sub(r'[^A-Z0-9]', '', text)
                
                # Remover palabras de la blacklist
                for word in blacklist_words:
                    clean_text = clean_text.replace(word, '')
                
                # Si queda una secuencia de 5-8 caracteres, es probable que sea la placa
                if 5 <= len(clean_text) <= 8:
                    return clean_text
                
                # Como última opción, tomar los primeros 6-7 caracteres válidos
                if len(clean_text) > 8:
                    # Buscar secuencias que parezcan placas (mezcla de letras y números)
                    for i in range(len(clean_text) - 5):
                        candidate = clean_text[i:i+6]
                        if re.search(r'[A-Z]', candidate) and re.search(r'[0-9]', candidate):
                            return candidate
                    
                    return clean_text[:7]  # Tomar los primeros 7 caracteres
                
                return clean_text
            
            # Extraer el número de placa
            output_text = extract_plate_number(full_text)
            print(f"Número de placa extraído: {output_text}")
            
            # Visualización
            cv2.imshow("plate_image", plate_image)
            # Dibujar resultados sobre la imagen
            cv2.rectangle(image, (x1 - 10, y1 - 35), (x2 + 10, y2-(y2 -y1)), (0, 255, 0), -1)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, output_text, (x1-7, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            
# Mostrar imagen final
cv2.imshow("Image", imutils.resize(image, width=720))
cv2.waitKey(0)
cv2.destroyAllWindows()