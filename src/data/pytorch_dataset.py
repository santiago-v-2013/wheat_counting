import torch
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
import os

class WheatCocoDataset(CocoDetection):
    """
    Dataset wrapper que adapta el CocoDetection de torchvision para
    retornar tensores compatibles con Faster R-CNN nativo de PyTorch.
    """
    def __init__(self, img_folder, ann_file):
        # Inicializa la clase padre de PyTorch que lee el archivo .json
        super().__init__(img_folder, ann_file)
        
    def __getitem__(self, idx):
        # 1. Extraer imagen (PIL) y lista de diccionarios (targets COCO)
        img, target = super().__getitem__(idx)
        
        # 2. Transformar imagen a Tensor [C, H, W] y normalizar [0, 1]
        img_tensor = T.ToTensor()(img)
        
        # 3. Procesar las cajas delimitadoras a formato [x_min, y_min, x_max, y_max]
        boxes = []
        labels = []
        for obj in target:
            x_min, y_min, w, h = obj['bbox']
            # Faster R-CNN espera coordenadas absolutas (no normalizadas)
            boxes.append([x_min, y_min, x_min + w, y_min + h])
            # La clase debe ser > 0, ya que 0 está reservado para el Background
            labels.append(obj['category_id']) 
            
        # 4. Empaquetar todo como tensores de PyTorch
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            # Caso donde la imagen no tiene trigo (solo fondo)
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
            
        target_dict = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }
        
        return img_tensor, target_dict

def collate_fn(batch):
    """
    Función necesaria para el DataLoader porque cada imagen puede tener 
    un número variable de cajas. Zip agrupa imágenes con imágenes y targets con targets.
    """
    return tuple(zip(*batch))
