# Wheat Heads Detection and Tracking

Proyecto de visión por computadora para el conteo de cabezas de trigo en imágenes y su seguimiento en video, orientado a aplicaciones en robótica.

## Estructura del Proyecto

* **data/**: Contiene los datos crudos, intermedios y procesados. (Ignorado por git)
* **notebooks/**: Jupyter notebooks para exploración de datos y prototipado.
* **src/**: Código fuente.
  * **data/**: Procesamiento de datos y técnicas de data augmentation (oclusión, blur).
  * **models/**: Arquitecturas de detección de objetos.
  * **tracking/**: Algoritmos para seguimiento en video.
  * **training/**: Scripts para entrenar y evaluar modelos.
  * **utils/**: Utilidades, métricas y visualización.
* **configs/**: Archivos `.yaml` de configuración.
* **weights/**: Pesos de los modelos entrenados. (Ignorado por git)
* **logs/**: Logs de entrenamiento (TensorBoard). (Ignorado por git)
* **tests/**: Pruebas unitarias.

## Instalación

1. Crear un entorno virtual:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```
