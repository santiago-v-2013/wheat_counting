import logging
import os
import sys

def setup_logger(name="WheatProject", log_file=None, level=logging.INFO):
    """
    Configura y devuelve un logger centralizado para el proyecto.
    
    Args:
        name (str): Nombre del logger.
        log_file (str, optional): Ruta absoluta o relativa donde guardar el log. 
                                  Si es None, solo mostrará por consola.
        level (int): Nivel de logging (e.g., logging.INFO, logging.DEBUG).
        
    Returns:
        logging.Logger: Instancia del logger configurado.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Evitar propagar a loggers root y evitar manejadores duplicados
    logger.propagate = False
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # Manejador de consola
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # Manejador de archivo (si se proporciona ruta)
    if log_file is not None:
        # Asegurarse de que el directorio del archivo de log exista
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger
