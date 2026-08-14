#!/bin/bash

# Este script ejecuta entrenamientos completos iterando sobre familias de modelos
# Ideal para clústeres Ubuntu. Los logs de salida se guardan en full_training.log

echo "=========================================================="
echo "    INICIANDO BARRIDO DE MODELOS EN CLÚSTER (UBUNTU)      "
echo "=========================================================="
echo "Asegúrate de ejecutar esto dentro de tmux o usando nohup si vas a cerrar la terminal."
echo "Ejemplo: nohup bash pipelines/run_full_training.sh > full_training.log 2>&1 &"
echo ""

# Activar entorno (modifica si tu entorno en el cluster tiene otro nombre/ruta)
eval "$(conda shell.bash hook)"
conda activate wheat_env

# ==========================================
# 1. Familia YOLO (Ultralytics)
# ==========================================
YOLO_MODELS=("yolov8n.pt" "yolov8s.pt" "yolov8m.pt")

echo "----------------------------------------------------------"
echo " Lote 1: Modelos YOLO (Anchor-Free)"
echo "----------------------------------------------------------"
for model in "${YOLO_MODELS[@]}"; do
    echo ">> Iniciando entrenamiento para YOLO: $model"
    python src/training/train_yolo.py --model "$model"
    if [ $? -ne 0 ]; then
        echo "❌ Error en el entrenamiento de $model. Abortando script..."
        exit 1
    fi
    echo "✓ Entrenamiento completado: $model"
    echo ""
done

# ==========================================
# 2. Familia PyTorch (Faster R-CNN / RetinaNet)
# ==========================================
# Nombres nativos de torchvision.models.detection
PT_MODELS=("fasterrcnn_resnet50_fpn" "retinanet_resnet50_fpn")

echo "----------------------------------------------------------"
echo " Lote 2: Modelos PyTorch (Anchor-Based)"
echo "----------------------------------------------------------"
for model in "${PT_MODELS[@]}"; do
    echo ">> Iniciando entrenamiento para PyTorch: $model"
    python src/training/train_custom.py --model "$model"
    if [ $? -ne 0 ]; then
        echo "❌ Error en el entrenamiento de $model. Abortando script..."
        exit 1
    fi
    echo "✓ Entrenamiento completado: $model"
    echo ""
done

# ==========================================
# 3. Familia HuggingFace (Transformers DETR)
# ==========================================
# Nombres de repositorios en HuggingFace hub
HF_MODELS=("facebook/detr-resnet-50" "hustvl/yolos-small")

echo "----------------------------------------------------------"
echo " Lote 3: Modelos HuggingFace (Transformers)"
echo "----------------------------------------------------------"
for model in "${HF_MODELS[@]}"; do
    echo ">> Iniciando entrenamiento para HuggingFace: $model"
    python src/training/train_hf.py --model "$model"
    if [ $? -ne 0 ]; then
        echo "❌ Error en el entrenamiento de $model. Abortando script..."
        exit 1
    fi
    echo "✓ Entrenamiento completado: $model"
    echo ""
done

echo "=========================================================="
echo " 🎉 BARRIDO COMPLETO DE TODOS LOS MODELOS TERMINADO 🎉"
echo "=========================================================="
