import os
import sys
import yaml
import time
import shutil
import pandas as pd
import torch
import argparse
from functools import partial
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.logger import setup_logger
from src.utils.plotter import plot_training_results
from src.models.detector import get_hf_detector
from src.data.hf_dataset import WheatHFDataset, hf_collate_fn

def main():
    parser = argparse.ArgumentParser(description="Entrenar modelo HuggingFace (DETR, YOLOS)")
    parser.add_argument('--config', type=str, default='configs/train_hf.yaml', help='Ruta configuración HF')
    parser.add_argument('--model', type=str, default=None, help='Sobrescribe el modelo especificado en el YAML')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    if args.model:
        cfg['model'] = args.model
        
    with open(cfg['dataset_cfg'], 'r') as f:
        data_cfg = yaml.safe_load(f)

    # --- 1. MLOps: Preparación de Directorios ---
    timestamp = time.strftime("%Y%m%d_%H%M")
    # Extraer nombre limpio (ej: facebook/detr-resnet-50 -> detr-resnet-50)
    model_name_clean = cfg['model'].split('/')[-1]
    run_name = f"{model_name_clean}_{timestamp}"
    run_dir = os.path.join("models", run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    shutil.copy(args.config, os.path.join(run_dir, "config.yaml"))
    
    os.makedirs("logs", exist_ok=True)
    logger = setup_logger(name="HF_Pipeline", log_file=os.path.join("logs", f"{run_name}.log"))
    logger.info(f"=== INICIANDO ENTRENAMIENTO HuggingFace: {cfg['model']} ===")
    
    # --- 2. Instanciar Modelo y Procesador ---
    device = torch.device(cfg.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    logger.info(f"Dispositivo: {device}")
    
    num_classes = data_cfg['dataset']['num_classes']
    model, processor = get_hf_detector(model_name=cfg['model'], num_classes=num_classes)
    model.to(device)
    
    # --- 3. Preparación de Datos (DataLoaders) ---
    root_dir = data_cfg['dataset']['root_dir']
    train_dataset = WheatHFDataset(
        img_folder=os.path.join(root_dir, "yolo", "images", "train"),
        ann_file=os.path.join(root_dir, "coco", "instances_train.json")
    )
    val_dataset = WheatHFDataset(
        img_folder=os.path.join(root_dir, "yolo", "images", "val"),
        ann_file=os.path.join(root_dir, "coco", "instances_val.json")
    )
    
    logger.info(f"Dataset Entrenamiento: {len(train_dataset)} | Validación: {len(val_dataset)}")
    
    collate_with_processor = partial(hf_collate_fn, processor=processor)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.get('batch', 2),
        shuffle=data_cfg['dataloader']['shuffle_train'],
        num_workers=cfg.get('workers', 0),
        collate_fn=collate_with_processor,
        pin_memory=data_cfg['dataloader']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.get('batch', 2),
        shuffle=False,
        num_workers=cfg.get('workers', 0),
        collate_fn=collate_with_processor,
        pin_memory=data_cfg['dataloader']['pin_memory']
    )
    
    # --- 4. Optimizador ---
    lr = cfg.get('lr0', 1e-4)
    weight_decay = cfg.get('weight_decay', 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 1. LR Scheduler (ReduceLROnPlateau)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    epochs = cfg.get('epochs', 1)
    patience = cfg.get('patience', 15)
    
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    history = []
    
    best_map = 0.0
    epochs_without_improvement = 0
    
    # --- 5. Bucle de Entrenamiento (Training Loop) ---
    logger.info(f"Comenzando el bucle de entrenamiento por {epochs} época(s) con patience de {patience}...")
    
    for epoch in range(epochs):
        # --- FASE ENTRENAMIENTO ---
        model.train() 
        epoch_total_loss = 0.0
        epoch_loss_ce = 0.0
        epoch_loss_bbox = 0.0
        epoch_loss_giou = 0.0
        
        for i, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch.get("pixel_mask", None)
            if pixel_mask is not None:
                pixel_mask = pixel_mask.to(device)
                
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
            
            # Forward pass HF
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            
            loss = outputs.loss
            loss_dict = outputs.loss_dict
            
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients (Recomendado para Transformers)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            
            epoch_total_loss += loss.item()
            epoch_loss_ce += loss_dict.get('loss_ce', torch.tensor(0)).item()
            epoch_loss_bbox += loss_dict.get('loss_bbox', torch.tensor(0)).item()
            epoch_loss_giou += loss_dict.get('loss_giou', torch.tensor(0)).item()
            
            if (i+1) % 10 == 0:
                ce_loss = loss_dict.get('loss_ce', torch.tensor(0)).item()
                box_loss = loss_dict.get('loss_bbox', torch.tensor(0)).item()
                giou_loss = loss_dict.get('loss_giou', torch.tensor(0)).item()
                logger.info(f"Época [{epoch+1}/{epochs}] Lote [{i+1}/{len(train_loader)}] Loss(CE/Box/GIoU): {ce_loss:.4f}/{box_loss:.4f}/{giou_loss:.4f}")
                
        avg_train_loss = epoch_total_loss / len(train_loader)
        
        logger.info("Realizando validación (Calculando mAP)...")
        # --- FASE VALIDACIÓN ---
        model.eval()
        metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
        
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                pixel_mask = batch.get("pixel_mask", None)
                if pixel_mask is not None:
                    pixel_mask = pixel_mask.to(device)
                
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
                
                # Decodificar predicciones a XYXY absoluto
                target_sizes = batch['target_sizes'].to(device)
                results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.0)
                
                # Formatear Predicciones para torchmetrics
                preds = []
                for res in results:
                    preds.append({
                        "boxes": res["boxes"].cpu(),
                        "scores": res["scores"].cpu(),
                        "labels": res["labels"].cpu()
                    })
                    
                # Extraer Anotaciones Crudas Originales para torchmetrics
                target_metrics = []
                for raw_ann in batch['raw_annotations']:
                    boxes = []
                    labels_list = []
                    for obj in raw_ann['annotations']:
                        # COCO raw is [x, y, w, h], we need [x1, y1, x2, y2]
                        x, y, w, h = obj['bbox']
                        boxes.append([x, y, x + w, y + h])
                        labels_list.append(obj['category_id'])
                        
                    if len(boxes) > 0:
                        target_metrics.append({
                            "boxes": torch.tensor(boxes, dtype=torch.float32),
                            "labels": torch.tensor(labels_list, dtype=torch.int64)
                        })
                    else:
                        target_metrics.append({
                            "boxes": torch.empty((0, 4), dtype=torch.float32),
                            "labels": torch.empty((0,), dtype=torch.int64)
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
        
        avg_ce = epoch_loss_ce / len(train_loader)
        avg_box = epoch_loss_bbox / len(train_loader)
        avg_giou = epoch_loss_giou / len(train_loader)
        
        logger.info(f"--- Fin Época {epoch+1} | mAP50: {map50:.4f} | mAP50-95: {map50_95:.4f} | Recall: {recall:.4f} | Loss(CE/Box/GIoU): {avg_ce:.4f}/{avg_box:.4f}/{avg_giou:.4f} ---")
        
        # Guardar historial con métricas nativas de HF (DETR usa Cross Entropy, Bbox, y GIoU)
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'loss_ce': epoch_loss_ce / len(train_loader),
            'loss_bbox': epoch_loss_bbox / len(train_loader),
            'loss_giou': epoch_loss_giou / len(train_loader),
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
    csv_path = os.path.join(run_dir, "results.csv")
    df_history = pd.DataFrame(history)
    df_history.to_csv(csv_path, index=False)
    
    plot_path = os.path.join(run_dir, "custom_training_summary.png")
    plot_training_results(csv_path, plot_path, model_name=f"HF {model_name_clean}")
    
    logger.info(f"¡Entrenamiento HuggingFace finalizado! Pesos guardados en {run_dir}")

if __name__ == '__main__':
    main()
