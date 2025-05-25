import os
from datetime import datetime

def ensure_directory_exists(path: str):
    """
    Asegura que el directorio especificado exista. Si no existe, lo crea.

    Args:
        path (str): La ruta del directorio a verificar/crear.
    """
    os.makedirs(path, exist_ok=True)
    print(f"Directorio asegurado: {path}")

def get_timestamp_filepath(base_dir: str, prefix: str = "", suffix: str = ".json") -> str:
    """
    Genera una ruta de archivo única basada en la fecha y hora actual.

    Args:
        base_dir (str): El directorio base donde se creará el archivo.
        prefix (str, optional): Un prefijo para el nombre del archivo. Por defecto es "".
        suffix (str, optional): El sufijo (extensión) del archivo. Por defecto es ".json".

    Returns:
        str: La ruta completa del archivo con el nombre de archivo generado.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}{timestamp}{suffix}"
    filepath = os.path.join(base_dir, filename)
    ensure_directory_exists(base_dir) # Asegurarse de que el directorio base exista
    return filepath

def convert_time_to_index(time_value: float, times_list: list, dt: float) -> int:
    """
    Convierte un valor de tiempo real a su índice de paso de tiempo más cercano.

    Args:
        time_value (float): El valor de tiempo a convertir (e.g., 8.0 para 8 AM).
        times_list (list): La lista de todos los valores de tiempo discretos del entorno.
        dt (float): El tamaño del paso de tiempo (e.g., 0.25 para 15 minutos).

    Returns:
        int: El índice entero del paso de tiempo.
    """
    # Manejar el caso de un solo timestep para evitar divisiones por cero
    if not times_list or dt == 0:
        return 0 # O levantar un error, dependiendo del contexto

    # Esto asume que times_list comienza en 0 o un valor cercano
    # y los tiempos son uniformes.
    # Se busca el índice más cercano para evitar problemas de coma flotante.
    return int(round((time_value - times_list[0]) / dt))

# Puedes añadir más funciones de utilidad aquí según las necesites
# Por ejemplo:
# def calculate_weighted_average(data_points: list, weights: list) -> float:
#     """Calcula el promedio ponderado."""
#     if not data_points or len(data_points) != len(weights):
#         return 0.0
#     return sum(dp * w for dp, w in zip(data_points, weights)) / sum(weights)