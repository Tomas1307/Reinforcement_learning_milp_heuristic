import os
import logging
import datetime
import sys

def setup_logger(log_name="scatter_search_debug", log_dir="logs"):
    """
    Configura un logger que escribe tanto en la consola como en un archivo.
    
    Args:
        log_name: Nombre para el archivo de log
        log_dir: Directorio donde se guardarán los logs
        
    Returns:
        Un objeto logger configurado
    """
    # Crear directorio si no existe
    os.makedirs(log_dir, exist_ok=True)
    
    # Obtener fecha y hora actual para el nombre del archivo
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{log_name}_{timestamp}.log")
    
    # Configurar logger
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    
    # Verificar si ya tiene handlers para evitar duplicados
    if not logger.handlers:
        # Handler para escribir en archivo
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Handler para escribir en consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formato de log
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Agregar handlers al logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    logger.info(f"Logger configurado. Archivo: {log_file}")
    return logger