import os
import sys
import yaml
import time
import shutil
import pandas as pd
import torch
import argparse
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.logger import setup_logger
from src.utils.plotter import plot_training_results
from src.models.detector import get_torchvision_detector
from src.data.pytorch_dataset import WheatCocoDataset, collate_fn

def main():
    parser = argparse.ArgumentParser(description="Entrenar modelo PyTorch Puro (Faster R-CNN)")
    parser.add_argument('--config', type=str, default='configs/train_faster_rcnn.yaml', help='Ruta configuración de entrenamiento')
    parser.add_argument('--model', type=str, default=None, help='Sobrescribe el modelo especificado en el YAML')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    if args.model:
        cfg['model'] = args.model
        
    with open(cfg['dataset_cfg'], 'r') as f:
        data_cfg = yaml.safe_load(f)

    # --- 1. MLOps: Preparación de Directorios y Logger ---
    timestamp = time.strftime("%Y%m%d_%H%M")
    model_name = cfg.get('model', 'faster_rcnn')
    run_name = f"{model_name}_{timestamp}"
    run_dir = os.path.join("models", run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Guardar copia de la configuración en la carpeta del modelo (Trazabilidad)
    shutil.copy(args.config, os.path.join(run_dir, "config.yaml"))
    
    os.makedirs("logs", exist_ok=True)
    logger = setup_logger(name="PyTorch_Pipeline", log_file=os.path.join("logs", f"{run_name}.log"))
    
    logger.info("=== INICIANDO ENTRENAMIENTO PyTorch Puro (Faster R-CNN) ===")
    
    # --- 2. Preparación de Datos (DataLoaders) ---
    root_dir = data_cfg['dataset']['root_dir'] # data/processed
    train_img_dir = os.path.join(root_dir, "yolo", "images", "train")
    train_ann_file = os.path.join(root_dir, "coco", "instances_train.json")
    
    val_img_dir = os.path.join(root_dir, "yolo", "images", "val")
    val_ann_file = os.path.join(root_dir, "coco", "instances_val.json")
    
    train_dataset = WheatCocoDataset(train_img_dir, train_ann_file)
    val_dataset = WheatCocoDataset(val_img_dir, val_ann_file)
    
    logger.info(f"Dataset Entrenamiento: {len(train_dataset)} | Validación: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.get('batch', 4),
        shuffle=data_cfg['dataloader']['shuffle_train'],
        num_workers=cfg.get('workers', 0),
        collate_fn=collate_fn,
        pin_memory=data_cfg['dataloader']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.get('batch', 4),
        shuffle=False,
        num_workers=cfg.get('workers', 0),
        collate_fn=collate_fn,
        pin_memory=data_cfg['dataloader']['pin_memory']
    )
    
    # --- 3. Instanciar Modelo ---
    device = torch.device(cfg.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    logger.info(f"Dispositivo de entrenamiento: {device}")
    
    # Clases = Fondo (0) + Trigo (1) = 2
    num_classes = data_cfg['dataset']['num_classes']
    model_name = cfg.get('model', 'fasterrcnn_resnet50_fpn')
    model = get_torchvision_detector(model_name=model_name, num_classes=num_classes)
    model.to(device)
    
    # --- 4. Optimizador ---
    params = [p for p in model.parameters() if p.requires_grad]
    
    lr = cfg.get('lr0', 0.005)
    momentum = cfg.get('momentum', 0.9)
    weight_decay = cfg.get('weight_decay', 0.0005)
    
    if cfg.get('optimizer', 'sgd').lower() == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        
    # 1. LR Scheduler (ReduceLROnPlateau)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
    epochs = cfg.get('epochs', 1)
    patience = cfg.get('patience', 15)
    
    # Importar métricas de COCO compatibles con YOLO
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    
    # Estructura para guardar el historial imitando las columnas de YOLO
    history = []
    
    best_map = 0.0
    epochs_without_improvement = 0
    
    # --- 5. Bucle de Entrenamiento (Training Loop) ---
    logger.info(f"Comenzando el bucle de entrenamiento por {epochs} época(s) con patience de {patience}...")
    
    for epoch in range(epochs):
        # ==================================================
        # FASE DE ENTRENAMIENTO
        # ==================================================
        model.train() 
        epoch_total_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_box_loss = 0.0
        epoch_obj_loss = 0.0
        epoch_rpn_loss = 0.0
        epoch_ctr_loss = 0.0
        
        for i, (images, targets) in enumerate(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            
            # Extraer las pérdidas dinámicamente según la arquitectura
            # Faster R-CNN usa: loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg
            # RetinaNet usa: classification, bbox_regression
            # FCOS usa: classification, bbox_regression, bbox_ctrness
            
            loss_classifier = loss_dict.get('classification', loss_dict.get('loss_classifier', torch.tensor(0.0))).item()
            loss_box_reg = loss_dict.get('bbox_regression', loss_dict.get('loss_box_reg', torch.tensor(0.0))).item()
            loss_objectness = loss_dict.get('loss_objectness', torch.tensor(0.0)).item()
            loss_rpn_box_reg = loss_dict.get('loss_rpn_box_reg', torch.tensor(0.0)).item()
            loss_ctrness = loss_dict.get('bbox_ctrness', torch.tensor(0.0)).item()
            
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            
            # 3. Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_total_loss += losses.item()
            epoch_cls_loss += loss_classifier
            epoch_box_loss += loss_box_reg
            epoch_obj_loss += loss_objectness
            epoch_rpn_loss += loss_rpn_box_reg
            epoch_ctr_loss += loss_ctrness
            
            if (i+1) % 10 == 0:
                cls_loss = loss_dict.get('loss_classifier', torch.tensor(0.0)).item()
                box_loss = loss_dict.get('loss_box_reg', torch.tensor(0.0)).item()
                logger.info(f"Época [{epoch+1}/{epochs}] Lote [{i+1}/{len(train_loader)}] Loss(Cls/Box): {cls_loss:.4f}/{box_loss:.4f}")
                
        avg_train_loss = epoch_total_loss / len(train_loader)
        
        logger.info("Realizando validación (Calculando mAP)...")
        # ==================================================
        # FASE DE VALIDACIÓN (EVALUACIÓN mAP)
        # ==================================================
        model.eval()
        metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = list(image.to(device) for image in images)
                outputs = model(images)
                
                preds = []
                for out in outputs:
                    preds.append({
                        "boxes": out["boxes"].cpu(),
                        "scores": out["scores"].cpu(),
                        "labels": out["labels"].cpu()
                    })
                    
                target_metrics = []
                for t in targets:
                    target_metrics.append({
                        "boxes": t["boxes"].cpu(),
                        "labels": t["labels"].cpu()
                    })
                    
                metric.update(preds, target_metrics)
                
        # Computar métricas finales
        metric_results = metric.compute()
        map50 = metric_results['map_50'].item()
        map50_95 = metric_results['map'].item()
        recall = metric_results['mar_100'].item()
        
        # 1. Aplicar paso del LR Scheduler basado en mAP50-95
        scheduler.step(map50_95)
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Tasa de aprendizaje actual: {current_lr}")
        
        avg_cls = epoch_cls_loss / len(train_loader)
        avg_box = epoch_box_loss / len(train_loader)
        
        logger.info(f"--- Fin Época {epoch+1} | mAP50: {map50:.4f} | mAP50-95: {map50_95:.4f} | Recall: {recall:.4f} | Loss(Cls/Box): {avg_cls:.4f}/{avg_box:.4f} ---")
        
        # Guardar historial con métricas nativas de Faster R-CNN
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'loss_classifier': epoch_cls_loss / len(train_loader),
            'loss_box_reg': epoch_box_loss / len(train_loader),
            'loss_objectness': epoch_obj_loss / len(train_loader),
            'loss_rpn_box_reg': epoch_rpn_loss / len(train_loader),
            'loss_ctrness': epoch_ctr_loss / len(train_loader) if 'epoch_ctr_loss' in locals() else 0.0,
            'map50': map50,
            'map50_95': map50_95,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Early Stopping y Guardado de Pesos
        last_path = os.path.join(run_dir, "last.pth")
        torch.save(model.state_dict(), last_path)
        
        if map50_95 > best_map:
            best_map = map50_95
            epochs_without_improvement = 0
            best_path = os.path.join(run_dir, "best.pth")
            torch.save(model.state_dict(), best_path)
            logger.info(f"🌟 Nuevo mejor modelo guardado con mAP50-95: {best_map:.4f}")
        else:
            epochs_without_improvement += 1
            logger.info(f"⚠️ Sin mejoras por {epochs_without_improvement} época(s).")
            
        if epochs_without_improvement >= patience:
            logger.info(f"🛑 Early Stopping desencadenado en la época {epoch+1}. El modelo dejó de aprender.")
            break
    
    # --- 6. Finalización y Ploteo MLOps ---
    # Guardar CSV
    csv_path = os.path.join(run_dir, "results.csv")
    df_history = pd.DataFrame(history)
    df_history.to_csv(csv_path, index=False)
    
    # Generar el Plot usando nuestra utilidad externa genérica!
    plot_path = os.path.join(run_dir, "custom_training_summary.png")
    plot_training_results(csv_path, plot_path, model_name="Faster R-CNN (PyTorch)")
    
    logger.info(f"¡Entrenamiento Puro finalizado! Pesos guardados en {run_dir}")

if __name__ == '__main__':
    main()
