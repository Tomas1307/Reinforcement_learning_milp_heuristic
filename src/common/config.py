import json
import yaml
import os

def load_data(json_path):
    """
    Carga el archivo JSON y retorna los datos del sistema.
    REPLICACIÓN EXACTA del main que funciona.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    energy_prices = sorted(data["energy_prices"], key=lambda x: x["time"])
    times = [ep["time"] for ep in energy_prices]
    prices = [ep["price"] for ep in energy_prices]

    arrivals = sorted(data["arrivals"], key=lambda x: x["id"])
    parking_config = data["parking_config"]
    chargers = parking_config["chargers"]
    station_limit = parking_config["transformer_limit"]
    n_spots = parking_config["n_spots"]

    if len(times) > 1:
        dt = times[1] - times[0]
    else:
        dt = 0.25

    return {
        "times": times,
        "prices": prices,
        "arrivals": arrivals,
        "chargers": chargers,
        "station_limit": station_limit,
        "dt": dt,
        "n_spots": n_spots,
        "test_number": data.get("test_number", 0),
        "parking_config": parking_config,
        "car_brands": data.get("car_brands", []),
        "charger_types": data.get("charger_types", {})
    }

def load_all_test_systems(data_dir="./data"):
    """
    Carga todos los sistemas de prueba disponibles en el directorio de datos.
    Asume que cada archivo tiene un test_number único (ej. test_system_1.json).
    REPLICACIÓN EXACTA del main que funciona.
    """
    systems = {}
    json_files = []

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.json') and 'test_system' in file:
                json_files.append(os.path.join(root, file))

    json_files.sort()

    for json_file in json_files:
        try:
            config = load_data(json_file)
            test_number = config.get("test_number")
            if test_number is None:
                raise ValueError(f"Archivo {json_file} no contiene 'test_number'")
            if test_number in systems:
                raise ValueError(f"Duplicado de ID detectado: {test_number} en {json_file}")
            systems[test_number] = config
            print(f"Sistema {test_number} cargado: {len(config['arrivals'])} vehículos, {config['n_spots']} plazas, {len(config['chargers'])} cargadores")
        except Exception as e:
            print(f"Error al cargar {json_file}: {e}")

    print(f"\nTotal: {len(systems)} sistemas únicos cargados.")
    return systems

def load_system_config(filepath: str) -> dict:
    """
    Versión mejorada con manejo de errores para la nueva estructura.
    Mantiene compatibilidad pero con mejor manejo de archivos que no tienen parking_config.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Archivo de configuración del sistema no encontrado: {filepath}")

    with open(filepath, 'r') as f:
        data = json.load(f)

    # Validar datos básicos
    if "energy_prices" not in data or "arrivals" not in data:
        raise KeyError("El archivo de configuración del sistema debe contener 'energy_prices' y 'arrivals'.")

    energy_prices = sorted(data["energy_prices"], key=lambda x: x["time"])
    times = [ep["time"] for ep in energy_prices]
    prices = [ep["price"] for ep in energy_prices]

    arrivals = sorted(data["arrivals"], key=lambda x: x["id"])
    
    # Manejo flexible de parking_config
    if "parking_config" in data:
        parking_config = data["parking_config"]
        chargers = parking_config["chargers"]
        station_limit = parking_config["transformer_limit"]
        n_spots = parking_config["n_spots"]
    else:
        # Fallback si no existe parking_config
        chargers = data.get("chargers", [])
        station_limit = data.get("station_limit", 100)
        n_spots = data.get("n_spots", 10)
        parking_config = {
            "chargers": chargers,
            "transformer_limit": station_limit,
            "n_spots": n_spots
        }

    dt = data.get("dt")
    if dt is None:
        if len(times) > 1:
            dt = times[1] - times[0]
        else:
            dt = 0.25
            print(f"Advertencia: 'dt' no especificado en la configuración del sistema ni se pudo inferir. Usando dt={dt}")

    return {
        "times": times,
        "prices": prices,
        "arrivals": arrivals,
        "chargers": chargers,
        "station_limit": station_limit,
        "n_spots": n_spots,
        "dt": dt,
        "test_number": data.get("test_number", 0),
        "parking_config": parking_config,
        "car_brands": data.get("car_brands", []),
        "charger_types": data.get("charger_types", {})
    }

def load_hyperparameters(filepath: str) -> dict:
    """
    Carga un archivo de hiperparámetros (YAML).

    Args:
        filepath (str): La ruta completa al archivo YAML de hiperparámetros.

    Returns:
        dict: Un diccionario con los hiperparámetros.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        yaml.YAMLError: Si el archivo YAML es inválido.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Archivo de hiperparámetros no encontrado: {filepath}")
    
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_scatter_search_params(filepath: str) -> dict:
    """
    Carga los parámetros de configuración para el algoritmo Scatter Search (YAML).

    Args:
        filepath (str): La ruta completa al archivo YAML de parámetros de Scatter Search.

    Returns:
        dict: Un diccionario con los parámetros de Scatter Search.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        yaml.YAMLError: Si el archivo YAML es inválido.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Archivo de parámetros de Scatter Search no encontrado: {filepath}")
    
    with open(filepath, 'r') as f:
        params = yaml.safe_load(f)
    return params