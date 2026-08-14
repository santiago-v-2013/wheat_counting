# Wheat Heads Detection and Tracking

Proyecto de visión por computadora para el conteo de cabezas de trigo en imágenes y su seguimiento en video, orientado a aplicaciones en robótica.

## Estructura del Proyecto

* **data/**:
  * **raw/**: Datos e imágenes originales (CSV, PNG).
  * **processed/**: Enlaces simbólicos y formatos listos para entrenar (YOLO y COCO). (Generado automáticamente).
* **notebooks/**: Jupyter notebooks para exploración de datos y calibración visual de *Data Augmentation*.
* **src/**: Código fuente principal.
  * **data/**: `make_dataset.py`, y adaptadores PyTorch (`pytorch_dataset.py`, `hf_dataset.py`).
  * **models/**: Arquitecturas de detección (`detector.py` modular para soportar Torchvision y HuggingFace).
  * **tracking/**: Algoritmos para seguimiento en video de las espigas.
  * **training/**: Scripts MLOps de entrenamiento (`train_yolo.py`, `train_custom.py`, `train_hf.py`).
  * **utils/**: Utilidades generales (ej. `plotter.py` para gráficas dinámicas de métricas).
* **configs/**: Archivos de configuración modulares:
  * `train_yolo.yaml`: Configuración para Ultralytics.
  * `train_faster_rcnn.yaml`: Configuración para modelos PyTorch nativos (Faster R-CNN, RetinaNet, FCOS).
  * `train_hf.yaml`: Configuración para Transformers (HuggingFace DETR, YOLOS).
  * `dataset_coco.yaml`: Configuración universal de DataLoaders.
  * `augmentation.yaml`: Probabilidades de Data Augmentation agnósticas al framework.
* **pipelines/**: Scripts Bash (`run_all_tests.sh`) para pruebas CI/CD y despliegue secuencial.
* **models/**: Experimentos, modelos entrenados y bitácoras (CSV). (Ignorado por git)
* **logs/**: Logs de consola. (Ignorado por git)
* **tests/**: Pruebas unitarias.

## Instalación

1. Crear un entorno virtual con Conda:
```bash
conda create --name wheat_env python=3.10
conda activate wheat_env
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

---

## 🛠️ Pipeline de Datos y Data Augmentation

El proyecto implementa Data Augmentation dinámico (Online). Las transformaciones se aplican en memoria (CPU) durante la carga de lotes antes de transferirse a la GPU, evitando la duplicación de imágenes en disco y reduciendo el sobreajuste.

### 1. Preparación del Dataset

Las imágenes crudas y los archivos CSV deben ubicarse en `data/raw/`. Para generar los formatos requeridos para el entrenamiento mediante enlaces simbólicos, ejecuta:

```bash
python src/data/make_dataset.py
```

**Salida:** Creará la carpeta `data/processed/` con dos estructuras:
- `yolo/`: Etiquetas normalizadas al centro en archivos `.txt`.
- `coco/`: Etiquetas en formato absoluto en archivos JSON.

### 2. Configuración de Aumentaciones

Los parámetros de aumento de datos se definen de manera centralizada en:

- 📄 `configs/augmentation.yaml`

Este archivo estandariza las transformaciones para que todos los modelos utilicen las mismas condiciones de entrenamiento.

### 3. PyTorch Custom Dataset

Si deseas entrenar un modelo puramente en PyTorch, el proyecto ya cuenta con una clase estándar `Dataset` que integra la lectura del formato COCO con el motor de transformaciones de Albumentations.

- 📄 `src/data/dataset.py`

**Para probar que el dataset funciona correctamente en consola:**
```bash
python src/data/dataset.py
```

## 🧪 Pruebas y Exploración de Datos

El repositorio incluye cuadernos de Jupyter para la validación visual del pipeline antes del entrenamiento:

1. **Validación del Dataset Procesado**
   `notebooks/01_visualizacion.ipynb`: Permite verificar la alineación espacial de las etiquetas (YOLO y COCO) generadas por `make_dataset.py` sobre las imágenes correspondientes.

2. **Evaluación de Data Augmentation**
   `notebooks/02_augmentation.ipynb`: Facilita el ajuste y visualización de los parámetros de Albumentations definidos en el YAML. Muestra comparativas entre las imágenes originales y las aumentadas para asistir en la selección de parámetros óptimos.
