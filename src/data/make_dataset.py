import os
import pandas as pd
import json
import numpy as np
from pathlib import Path
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.logger import setup_logger

logger = setup_logger(name="DataPreparation", level=20) # 20 is logging.INFO

# Configuración de rutas (asumiendo que se ejecuta desde la raíz del proyecto)
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

YOLO_DIR = PROCESSED_DIR / "yolo"
COCO_DIR = PROCESSED_DIR / "coco"

# En el dataset GWHD 2021 todas las imágenes son de 1024x1024
IMG_WIDTH = 1024
IMG_HEIGHT = 1024

def setup_dirs():
    """Crea la estructura de carpetas necesaria para YOLO y COCO."""
    for split in ['train', 'val', 'test']:
        (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    COCO_DIR.mkdir(parents=True, exist_ok=True)

def parse_boxes(boxes_string):
    """Parsea el string de cajas del CSV original a una lista de coordenadas float."""
    if pd.isna(boxes_string) or boxes_string.strip() == "no_box" or boxes_string.strip() == "":
        return []
    
    boxes = []
    # boxes_string: "x_min y_min x_max y_max;x_min y_min..."
    for box_str in boxes_string.split(';'):
        if not box_str.strip():
            continue
        try:
            coords = list(map(float, box_str.strip().split(' ')))
            if len(coords) == 4:
                boxes.append(coords)
        except Exception as e:
            pass
    return boxes

def convert_to_yolo(split_name, df):
    """Convierte anotaciones a YOLO y hace symlinks a las imágenes."""
    for idx, row in df.iterrows():
        img_name = row['image_name']
        if not img_name.endswith('.png'):
            img_name += '.png'
            
        boxes = parse_boxes(row['BoxesString'])
        
        src_img_path = RAW_DIR / "images" / img_name
        dst_img_path = YOLO_DIR / "images" / split_name / img_name
        
        # En lugar de copiar los 10GB de imágenes, creamos enlaces simbólicos
        if src_img_path.exists():
            if not dst_img_path.exists():
                os.symlink(src_img_path.absolute(), dst_img_path)
        
        # Escribir el archivo .txt de YOLO
        txt_name = img_name.replace(".png", ".txt").replace(".jpg", ".txt")
        label_path = YOLO_DIR / "labels" / split_name / txt_name
        
        with open(label_path, 'w') as f:
            for box in boxes:
                x_min, y_min, x_max, y_max = box
                w = x_max - x_min
                h = y_max - y_min
                
                # Calcular centro
                x_c = x_min + w / 2
                y_c = y_min + h / 2
                
                # Normalizar a valores entre 0 y 1
                x_c /= IMG_WIDTH
                y_c /= IMG_HEIGHT
                w /= IMG_WIDTH
                h /= IMG_HEIGHT
                
                # Asegurar límites correctos
                x_c, y_c, w, h = np.clip([x_c, y_c, w, h], 0, 1)
                
                # Clase 0 para YOLO (wheat_head)
                f.write(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")


def convert_to_coco(split_name, df):
    """Convierte las anotaciones a formato COCO (JSON)."""
    coco = {
        "info": {"description": f"Global Wheat Head {split_name}"},
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "wheat_head"}]
    }
    
    ann_id = 1
    for img_id, (idx, row) in enumerate(df.iterrows(), 1):
        img_name = row['image_name']
        if not img_name.endswith('.png'):
            img_name += '.png'
            
        boxes = parse_boxes(row['BoxesString'])
        
        coco["images"].append({
            "id": img_id,
            "file_name": img_name,
            "width": IMG_WIDTH,
            "height": IMG_HEIGHT
        })
        
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            w = x_max - x_min
            h = y_max - y_min
            
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [x_min, y_min, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            ann_id += 1
            
    # Guardar el JSON
    json_path = COCO_DIR / f"instances_{split_name}.json"
    with open(json_path, 'w') as f:
        json.dump(coco, f, separators=(',', ':'))

def main():
    logger.info("Inicializando estructura de directorios en data/processed...")
    setup_dirs()
    
    # Mapeo de archivos originales a sus nombres de 'split' (división)
    splits = {
        "competition_train.csv": "train",
        "competition_val.csv": "val",
        "competition_test.csv": "test"
    }
    
    for split_file, split_name in splits.items():
        csv_path = RAW_DIR / split_file
        if csv_path.exists():
            logger.info(f"--- Procesando subset: {split_name.upper()} ---")
            df = pd.read_csv(csv_path)
            
            logger.info(f" -> Generando formato YOLO en {YOLO_DIR}/...")
            convert_to_yolo(split_name, df)
            
            logger.info(f" -> Generando formato COCO en {COCO_DIR}/...")
            convert_to_coco(split_name, df)
        else:
            logger.warning(f"Archivo {split_file} no encontrado, saltando {split_name}.")
            
    # Crear el archivo dataset.yaml para Ultralytics (YOLOv8/v9/v10)
    yolo_yaml_path = PROCESSED_DIR / "yolo" / "dataset.yaml"
    with open(yolo_yaml_path, 'w') as f:
        abs_yolo_path = YOLO_DIR.absolute()
        f.write(f"path: {abs_yolo_path}  # Ruta absoluta al dataset\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write("\nnames:\n")
        f.write("  0: wheat_head\n")

    logger.info("¡Conversión completada con éxito!")

if __name__ == "__main__":
    main()
