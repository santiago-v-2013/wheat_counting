import os
import json
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path

# Asumimos que aumentations.py está en la misma carpeta
try:
    from .augmentations import get_train_transforms, get_val_transforms
except ImportError:
    from augmentations import get_train_transforms, get_val_transforms

class WheatCOCODataset(Dataset):
    """
    Custom PyTorch Dataset para cargar imágenes y anotaciones en formato COCO.
    Aplica Data Augmentation "On-the-fly" usando Albumentations.
    """
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): Directorio base de los datos procesados (ej: 'data/processed').
            split (str): 'train', 'val' o 'test'.
            transform (callable, optional): Pipeline de Albumentations.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_dir = self.root_dir / "yolo" / "images" / split
        self.coco_json_path = self.root_dir / "coco" / f"instances_{split}.json"
        self.transform = transform
        
        # Cargar el archivo COCO JSON en memoria
        with open(self.coco_json_path, 'r') as f:
            self.coco_data = json.load(f)
            
        # Mapear imágenes por ID para acceso rápido
        self.images_info = {img['id']: img for img in self.coco_data['images']}
        self.image_ids = list(self.images_info.keys())
        
        # Agrupar anotaciones por image_id
        self.annotations = {img_id: [] for img_id in self.image_ids}
        for ann in self.coco_data['annotations']:
            self.annotations[ann['image_id']].append(ann)
            
    def __len__(self):
        return len(self.image_ids)
        
    def __getitem__(self, idx):
        # 1. Obtener info de la imagen
        img_id = self.image_ids[idx]
        img_info = self.images_info[img_id]
        
        # 2. Cargar imagen (OpenCV carga en BGR, pasamos a RGB para Albumentations/PyTorch)
        img_path = self.img_dir / img_info['file_name']
        image = cv2.imread(str(img_path))
        
        if image is None:
            raise FileNotFoundError(f"Imagen no encontrada: {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 3. Extraer bounding boxes [x_min, y_min, width, height] y etiquetas
        anns = self.annotations[img_id]
        bboxes = []
        class_labels = []
        
        for ann in anns:
            bboxes.append(ann['bbox'])
            class_labels.append(ann['category_id']) # 1 = wheat_head
            
        # 4. Aplicar "On-the-fly" Data Augmentation (Albumentations)
        if self.transform is not None:
            # Albumentations espera las boxes y los labels de clase
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            image = transformed['image']
            bboxes = transformed['bboxes']
            class_labels = transformed['class_labels']
            
        # 5. Formatear salida para PyTorch
        # Convertir imagen de HWC a CHW y normalizar a rango [0, 1]
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # Formatear target para modelos tipo Torchvision (FasterRCNN, etc.)
        # Torchvision espera las cajas en formato [x_min, y_min, x_max, y_max]
        if len(bboxes) > 0:
            boxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
            # COCO: [x, y, w, h] -> [x1, y1, x2, y2]
            boxes_tensor[:, 2] = boxes_tensor[:, 0] + boxes_tensor[:, 2]
            boxes_tensor[:, 3] = boxes_tensor[:, 1] + boxes_tensor[:, 3]
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            
        labels_tensor = torch.tensor(class_labels, dtype=torch.int64)
        
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([img_id])
        }
        
        return image, target

# --- Ejemplo de Uso Rápido ---
if __name__ == "__main__":
    # Creamos un pipeline de validación para probar que lee bien (formato COCO)
    val_transform = get_val_transforms(
        config_path="../../configs/augmentation.yaml", 
        bbox_format="coco"
    )
    
    # Suponiendo que ejecutamos esto desde src/data/
    try:
        dataset = WheatCOCODataset(
            root_dir="../../data/processed", 
            split="train", 
            transform=val_transform
        )
        print(f"Total de imágenes cargadas: {len(dataset)}")
        
        if len(dataset) > 0:
            img, target = dataset[0]
            print(f"Shape de la imagen: {img.shape}")
            print(f"Cantidad de Bounding Boxes: {target['boxes'].shape[0]}")
            print("Target keys:", target.keys())
    except FileNotFoundError as e:
        print("Aún no has generado los datos procesados. Por favor corre src/data/make_dataset.py desde la raíz del proyecto.")
