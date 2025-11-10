import cv2
import numpy as np
import re

def load_image(contents):
    """
    Loads and decodes an image from a file.
    """
    npimg = np.frombuffer(contents, np.uint8)
    return npimg

def extract_plate_number(text):
    """
    Extracts a valid license plate number from given text.
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
    Parses detected text lines to extract specific information.
    """
    
    text_all = " ".join(lines)

    data = {
        "tipo_identificacion": None,
        "nombre": None,
        "domicilio": None,
        "fecha_nacimiento": None,
        "curp": None,
        "sexo": None
    }

    #Tipo_identificacion: 
    id_types = {
        "INSTITUTO NACIONAL ELECTORAL": "INE",
        "CREDENCIAL PARA VOTAR": "INE",
        "PASAPORTE": "Pasaporte",
        "LICENCIA DE CONDUCIR": "Licencia de Conducir"
    }
    for key, value in id_types.items():
        if key in text_all:
            data["tipo_identificacion"] = value
            break

    # Nombre: todo el texto después de "NOMBRE" hasta "DOMICILIO"
    try:
        idx_nombre = lines.index("NOMBRE")
        idx_domicilio = lines.index("DOMICILIO") if "DOMICILIO" in lines else len(lines)
        nombre_lines = []
        for line in lines[idx_nombre+1:idx_domicilio]:
            if line in ["SEXO H", "SEXO M"]:
                continue
            nombre_lines.append(line)
        data["nombre"] = " ".join(nombre_lines)
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