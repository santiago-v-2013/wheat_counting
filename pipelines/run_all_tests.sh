#!/bin/bash

# Este script ejecuta secuencialmente las pruebas de entrenamiento de 1 época
# para cada uno de los tres pipelines principales del proyecto (YOLO, Faster R-CNN, DETR).
# Esto asegura que todos los flujos MLOps están funcionando correctamente sin crashear.

echo "=========================================================="
echo "      INICIANDO PRUEBAS SECUENCIALES DE 1 ÉPOCA           "
echo "=========================================================="
echo ""

# 1. Pipeline YOLOv8 (Anchor-Free, Ultralytics)
echo "----------------------------------------------------------"
echo "[1/3] Ejecutando Pipeline YOLOv8..."
echo "----------------------------------------------------------"
conda run -n wheat_env python src/training/train_yolo.py
if [ $? -ne 0 ]; then
    echo "❌ Error en el pipeline YOLOv8. Abortando secuencial..."
    exit 1
fi
echo "✓ Pipeline YOLOv8 completado."
echo ""

# 2. Pipeline PyTorch Puro (Anchor-Based, Faster R-CNN)
echo "----------------------------------------------------------"
echo "[2/3] Ejecutando Pipeline PyTorch (Faster R-CNN)..."
echo "----------------------------------------------------------"
conda run -n wheat_env python src/training/train_custom.py
if [ $? -ne 0 ]; then
    echo "❌ Error en el pipeline PyTorch. Abortando secuencial..."
    exit 1
fi
echo "✓ Pipeline PyTorch completado."
echo ""

# 3. Pipeline HuggingFace (Transformer-Based, DETR)
echo "----------------------------------------------------------"
echo "[3/3] Ejecutando Pipeline HuggingFace (DETR)..."
echo "----------------------------------------------------------"
conda run -n wheat_env python src/training/train_hf.py
if [ $? -ne 0 ]; then
    echo "❌ Error en el pipeline HuggingFace. Abortando secuencial..."
    exit 1
fi
echo "✓ Pipeline HuggingFace completado."
echo ""

echo "=========================================================="
echo " 🎉 TODAS LAS PRUEBAS SE COMPLETARON CON ÉXITO 🎉"
echo "=========================================================="
echo "Revisa la carpeta 'models/' para ver los tres resultados y gráficos generados."
