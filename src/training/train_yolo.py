import yaml
from ultralytics import YOLO
import os
import time
import shutil
from pathlib import Path
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.logger import setup_logger

def train_model(cfg, logger, run_name):
    """
    Ejecuta el entrenamiento de un modelo YOLO usando los parámetros del YAML.
    """
    raw_model_name = cfg.get('model_name', 'yolov8n.pt')
    
    # Crear directorio para almacenar los pesos preentrenados y evitar que queden en la raíz
    pretrained_dir = os.path.abspath(os.path.join("models", "pretrained"))
    os.makedirs(pretrained_dir, exist_ok=True)
    
    # MLOps: Configurar Ultralytics para que descargue siempre en la carpeta pretrained
    from ultralytics import settings
    settings.update({'weights_dir': pretrained_dir})
    
    model_path = os.path.join(pretrained_dir, raw_model_name)
    
    logger.info(f"Iniciando entrenamiento para la arquitectura: {raw_model_name}...")
    
    try:
        original_cwd = os.getcwd()
        try:
            os.chdir(pretrained_dir)
            # Instanciar el modelo aquí asegura que se descargue en pretrained_dir
            model = YOLO(raw_model_name)
            
            def on_train_epoch_start(trainer):
                trainer.my_batch_count = 0
                
            def on_train_batch_end(trainer):
                epoch = trainer.epoch + 1
                epochs = trainer.epochs
                
                if not hasattr(trainer, 'my_batch_count'):
                    trainer.my_batch_count = 0
                trainer.my_batch_count += 1
                i = trainer.my_batch_count
                total_batches = len(trainer.train_loader)
                
                # Logear el desglose exacto de las pérdidas cada 10 lotes
                if i % 10 == 0:
                    try:
                        box_loss, cls_loss, dfl_loss = trainer.loss_items
                        logger.info(f"Época [{epoch}/{epochs}] Lote [{i}/{total_batches}] Loss(Box/Cls/DFL): {box_loss.item():.4f}/{cls_loss.item():.4f}/{dfl_loss.item():.4f}")
                    except Exception:
                        pass
                
            def on_fit_epoch_end(trainer):
                epoch = trainer.epoch + 1
                metrics = trainer.validator.metrics
                if metrics:
                    r = metrics.box.mr  # Mean Recall
                    map50 = metrics.box.map50
                    map50_95 = metrics.box.map
                    logger.info(f"--- Fin Época {epoch} | mAP50: {map50:.4f} | mAP50-95: {map50_95:.4f} | Recall: {r:.4f} ---")

            model.add_callback("on_train_epoch_start", on_train_epoch_start)
            model.add_callback("on_train_batch_end", on_train_batch_end)
            model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
            
            # MLOps: Resolver rutas absolutas
            abs_data_path = str(Path(original_cwd) / cfg['data'])
            abs_project_path = str(Path(original_cwd) / 'models')
            
            # Parámetros de aumentación online nativos de YOLO
            aug_params = cfg.get('augmentation_params', {})
            
            # MLOps: Inyección de hiperparámetros y APAGADO de los plots de YOLO
            results = model.train(
                data=abs_data_path,
                epochs=cfg['epochs'],
                batch=cfg['batch'],
                imgsz=cfg['imgsz'],
                project=abs_project_path,  # Ruta absoluta
                name=run_name,             # La subcarpeta será model_name + timestamp
                device=str(cfg['device']),
                workers=cfg.get('workers', 0), 
                optimizer=cfg.get('optimizer', 'auto'),
                lr0=cfg.get('lr0', 0.01),
                patience=cfg.get('patience', 15),
                plots=False,         # MLOps: ¡APAGAMOS LA GENERACIÓN DE IMÁGENES DE YOLO!
                exist_ok=True,
                deterministic=True,  
                seed=42,             
                **aug_params         
            )
        finally:
            os.chdir(original_cwd)
            
        # MLOps: Nuestro propio creador de gráficos a partir del results.csv usando la función compartida
        logger.info("Generando gráfico de entrenamiento personalizado a partir de results.csv...")
        run_dir = os.path.join("models", run_name)
        csv_path = os.path.join(run_dir, "results.csv")
        plot_path = os.path.join(run_dir, "custom_training_summary.png")
        
        # Importar el generador de gráficos genérico
        from src.utils.plotter import plot_training_results
        success = plot_training_results(csv_path, plot_path, model_name=raw_model_name)
        
        if success:
            logger.info(f"Gráfico personalizado guardado en: {plot_path}")
        else:
            logger.warning("No se encontró results.csv o hubo un error al generar el gráfico.")
            
        logger.info(f"✓ Entrenamiento completado con éxito: {raw_model_name}")
        return True
    except Exception as e:
        logger.error(f"❌ Falló el entrenamiento de {raw_model_name}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Entrenar modelo YOLO con hiperparámetros configurables.")
    parser.add_argument('--config', type=str, default='configs/train_yolo.yaml', help='Ruta al archivo YAML de configuración')
    parser.add_argument('--model', type=str, default=None, help='Sobrescribe el modelo especificado en el YAML')
    args = parser.parse_args()

    config_path = args.config
    
    if not os.path.exists(config_path):
        print(f"Error: No se encontró el archivo de configuración {config_path}")
        sys.exit(1)
        
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    if args.model:
        cfg['model_name'] = args.model
        
    # --- MLOps: Gestión del Nombre del Modelo y Timestamp ---
    model_name = cfg.get('model_name', 'yolov8n.pt')
    clean_model_name = model_name.replace('.pt', '')
    timestamp = time.strftime("%Y%m%d_%H%M")
    
    run_name = f"{clean_model_name}_{timestamp}"
    run_dir = os.path.join("models", run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # --- MLOps: Archivo de Log Centralizado ---
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", f"{run_name}.log")
    logger = setup_logger(name="YOLO_Pipeline", log_file=log_path)
    
    # --- MLOps: Respaldo de Configuraciones (Snapshot) ---
    # Copiamos EXACTAMENTE qué parámetros usamos a la carpeta de salida.
    snapshot_path = os.path.join(run_dir, "experiment_config.yaml")
    shutil.copy(config_path, snapshot_path)
    
    logger.info("=== INICIANDO ENTRENAMIENTO YOLO MLOPS ===")
    logger.info(f"Modelo Objetivo: {model_name}")
    logger.info(f"Directorio de resultados (Pesos, Gráficas, CSV): {os.path.abspath(run_dir)}")
    logger.info(f"Archivo de log: {os.path.abspath(log_path)}")
    logger.info(f"Dispositivo(s): {cfg['device']}")
    logger.info("==================================================")
    
    # Ejecución del modelo
    train_model(cfg, logger, run_name)
            
    logger.info(f"¡Entrenamiento finalizado! Puedes encontrar el mejor peso en {os.path.abspath(run_dir)}/weights/best.pt")

if __name__ == '__main__':
    main()
