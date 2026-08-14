import torch
from torchvision.datasets import CocoDetection

class WheatHFDataset(CocoDetection):
    """
    Dataset wrapper para modelos de HuggingFace (DETR, YOLOS, Conditional-DETR).
    Retorna la imagen cruda (PIL) y las anotaciones crudas.
    El procesamiento final a tensores se hace dinámicamente en el collate_fn 
    usando el AutoImageProcessor.
    """
    def __init__(self, img_folder, ann_file):
        super().__init__(img_folder, ann_file)
        
    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        # HuggingFace espera imágenes siempre en RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Obtenemos el image_id real de COCO, si está disponible, si no, usamos el idx
        image_id = self.ids[idx]
        
        # Filtramos para asegurar que todas las anotaciones tengan categoría
        annotations = []
        for obj in target:
            annotations.append({
                "bbox": obj['bbox'],
                "category_id": obj['category_id'],
                "area": obj.get('area', obj['bbox'][2] * obj['bbox'][3]),
                "iscrowd": obj.get('iscrowd', 0)
            })
            
        target_dict = {
            "image_id": image_id,
            "annotations": annotations
        }
        
        return img, target_dict

def hf_collate_fn(batch, processor):
    """
    Recibe un batch de (imagen, target_dict) y utiliza el ImageProcessor de 
    HuggingFace para normalizarlas, escalarlas y re-formatear las cajas.
    """
    images = [item[0] for item in batch]
    annotations = [item[1] for item in batch]
    
    # El processor hace todo el trabajo pesado: resize, normalize, bbox format (xyxy/cxcywh)
    batch_processed = processor(images=images, annotations=annotations, return_tensors="pt")
    
    # Inyectamos datos crudos para facilitar la validación (mAP)
    batch_processed['raw_annotations'] = annotations
    # PIL img.size es (W, H), pero post_process_object_detection espera (H, W)
    batch_processed['target_sizes'] = torch.tensor([img.size[::-1] for img in images])
    
    # batch_processed contiene 'pixel_values', 'pixel_mask', y 'labels'
    return batch_processed
