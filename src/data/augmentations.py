import albumentations as A
import yaml
from pathlib import Path

def load_augmentation_config(config_path="configs/augmentation.yaml"):
    """Lee el archivo YAML de configuración de transformaciones."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_train_transforms(config_path="configs/augmentation.yaml", bbox_format="coco"):
    """
    Construye el pipeline de aumentación dinámico de Albumentations 
    basado en los parámetros del archivo YAML.
    """
    config = load_augmentation_config(config_path)
    transf_cfg = config['transformations']
    
    transforms = []
    
    # 1. Simulación de Movimiento / Blur
    if transf_cfg.get('motion_blur', {}).get('apply', False):
        cfg = transf_cfg['motion_blur']
        transforms.append(A.MotionBlur(blur_limit=cfg['blur_limit'], p=cfg['p']))
        
    if transf_cfg.get('gaussian_blur', {}).get('apply', False):
        cfg = transf_cfg['gaussian_blur']
        transforms.append(A.GaussianBlur(blur_limit=cfg['blur_limit'], p=cfg['p']))
        
    # 2. Oclusiones (CoarseDropout / Parches Negros)
    if transf_cfg.get('coarse_dropout', {}).get('apply', False):
        cfg = transf_cfg['coarse_dropout']
        transforms.append(
            A.CoarseDropout(
                max_holes=cfg['max_holes'],
                max_height=cfg['max_height'],
                max_width=cfg['max_width'],
                min_holes=cfg['min_holes'],
                min_height=cfg['min_height'],
                min_width=cfg['min_width'],
                fill_value=cfg['fill_value'],
                p=cfg['p']
            )
        )
        
    # 3. Clima e Iluminación
    if transf_cfg.get('random_brightness_contrast', {}).get('apply', False):
        cfg = transf_cfg['random_brightness_contrast']
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=cfg['brightness_limit'],
                contrast_limit=cfg['contrast_limit'],
                p=cfg['p']
            )
        )
        
    if transf_cfg.get('random_shadow', {}).get('apply', False):
        cfg = transf_cfg['random_shadow']
        transforms.append(
            A.RandomShadow(
                num_shadows_lower=cfg['num_shadows_lower'],
                num_shadows_upper=cfg['num_shadows_upper'],
                shadow_dimension=cfg['shadow_dimension'],
                p=cfg['p']
            )
        )
        
    # 4. Geométricas Espaciales
    if transf_cfg.get('horizontal_flip', {}).get('apply', False):
        transforms.append(A.HorizontalFlip(p=transf_cfg['horizontal_flip']['p']))
        
    if transf_cfg.get('vertical_flip', {}).get('apply', False):
        transforms.append(A.VerticalFlip(p=transf_cfg['vertical_flip']['p']))

    # Crear el pipeline maestro.
    # label_fields se usa para pasar la clase de la caja junto a las coordenadas.
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format=bbox_format, label_fields=['class_labels'])
    )

def get_val_transforms(config_path="configs/augmentation.yaml", bbox_format="coco"):
    """
    Pipeline para validación/test.
    Las imágenes de validación NO deben llevar aumentaciones pesadas.
    """
    return A.Compose(
        [], 
        bbox_params=A.BboxParams(format=bbox_format, label_fields=['class_labels'])
    )
