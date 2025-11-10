import cv2
import numpy as np
import re

def load_image(contents):
    """
    Carga y decodifica una imagen desde un archivo.
    """
    npimg = np.frombuffer(contents, np.uint8)
    return npimg

def extract_plate_number(text):
    """
    Extrae un número de placa válido de un texto dado.
    """
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
            if 5 <= len(match) <= 8:
                return match
    return None

def parse_ine_data(lines):
    """
    Procesa las líneas de texto detectadas para extraer información específica.
    """
    text_all = " ".join(lines)

    data = {
        "nombre": None,
        "domicilio": None,
        "fecha_nacimiento": None,
        "curp": None,
        "sexo": None
    }

    # Nombre: 3 líneas después de "NOMBRE"
    try:
        idx = lines.index("NOMBRE")
        data["nombre"] = " ".join(lines[idx+1:idx+4])
    except:
        pass

    # Domicilio: después de "DOMICILIO" hasta nueva sección
    try:
        idx = lines.index("DOMICILIO")
        domicilio_parts = []
        for i in range(idx+1, len(lines)):
            if "CLAVE" in lines[i] or "CURP" in lines[i] or "SEXO" in lines[i]:
                break
            domicilio_parts.append(lines[i])
        data["domicilio"] = " ".join(domicilio_parts)
    except:
        pass

    # CURP
    curp = re.search(r"\b[A-Z]{4}\d{6}[A-Z]{6}\d{2}\b", text_all)
    if curp:
        data["curp"] = curp.group(0)

    # Fecha de nacimiento (dd/mm/aaaa)
    fecha = re.search(r"\b\d{2}/\d{2}/\d{4}\b", text_all)
    if fecha:
        data["fecha_nacimiento"] = fecha.group(0)

    # Sexo
    if "SEXO H" in text_all:
        data["sexo"] = "H"
    elif "SEXO M" in text_all:
        data["sexo"] = "M"

    return data