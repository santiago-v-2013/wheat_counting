import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

def get_torchvision_detector(model_name, num_classes):
    """
    Carga dinámicamente un modelo de detección de torchvision.
    Soporta familias:
    - Faster R-CNN ('fasterrcnn_resnet50_fpn', etc)
    - RetinaNet ('retinanet_resnet50_fpn', etc)
    - FCOS ('fcos_resnet50_fpn', etc)
    """
    if not hasattr(torchvision.models.detection, model_name):
        raise ValueError(f"El modelo {model_name} no existe en torchvision.models.detection")
        
    model_builder = getattr(torchvision.models.detection, model_name)
    
    # Cargar el modelo base con pesos preentrenados (COCO)
    model = model_builder(weights="DEFAULT")
    
    # Adaptación dinámica de la capa final (Cirugía de Arquitectura)
    if "fasterrcnn" in model_name:
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
    elif "retinanet" in model_name:
        from torchvision.models.detection.retinanet import RetinaNetClassificationHead
        in_channels = model.head.classification_head.conv[0][0].in_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(in_channels, num_anchors, num_classes)
        
    elif "fcos" in model_name:
        from torchvision.models.detection.fcos import FCOSClassificationHead
        in_channels = model.head.classification_head.conv[0].in_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = FCOSClassificationHead(in_channels, num_anchors, num_classes)
        
    else:
        raise NotImplementedError(f"La cirugía de adaptación para {model_name} aún no está programada.")
        
    return model

def get_hf_detector(model_name, num_classes):
    """
    Carga dinámicamente cualquier modelo de detección de objetos de HuggingFace 
    (ej. facebook/detr-resnet-50, microsoft/conditional-detr-resnet-50, etc).
    """
    from transformers import AutoModelForObjectDetection, AutoImageProcessor
    
    processor = AutoImageProcessor.from_pretrained(model_name)
    
    # ignore_mismatched_sizes=True permite descargar pesos preentrenados (de COCO 91 clases)
    # y reescribir la capa de clasificación final para nuestras 2 clases.
    model = AutoModelForObjectDetection.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    
    return model, processor
