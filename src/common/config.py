import json
import yaml
import os

def load_data(json_path):
    """
    Loads a JSON file and returns the system data.

    This function is an exact replication of the working main logic.

    Args:
        json_path (str): The path to the JSON file.

    Returns:
        dict: A dictionary containing the loaded system data, including:
            - times (list): Sorted list of time points.
            - prices (list): List of energy prices corresponding to each time point.
            - arrivals (list): Sorted list of vehicle arrival data.
            - chargers (list): List of charger configurations.
            - station_limit (float): The transformer limit of the station.
            - dt (float): The time step between consecutive time points.
            - n_spots (int): The number of parking spots.
            - test_number (int): The test number, defaults to 0 if not found.
            - parking_config (dict): The complete parking configuration.
            - car_brands (list): List of car brands, defaults to empty list if not found.
            - charger_types (dict): Dictionary of charger types, defaults to empty dict if not found.
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
    Loads all available test systems from the data directory.
    Assumes each file has a unique test_number (e.g., test_system_1.json).

    This function is an exact replication of the working main logic.

    Args:
        data_dir (str): The directory containing the test system JSON files.
                        Defaults to "./data".

    Returns:
        dict: A dictionary where keys are test numbers and values are the
              loaded system configurations.
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
                raise ValueError(f"File {json_file} does not contain 'test_number'")
            if test_number in systems:
                raise ValueError(f"Duplicate ID detected: {test_number} in {json_file}")
            systems[test_number] = config
            print(f"System {test_number} loaded: {len(config['arrivals'])} vehicles, {config['n_spots']} spots, {len(config['chargers'])} chargers")
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    print(f"\nTotal: {len(systems)} unique systems loaded.")
    return systems

def load_system_config(filepath: str) -> dict:
    """
    Loads a system configuration from a JSON file with improved error handling
    for the new structure, while maintaining compatibility.

    Args:
        filepath (str): The full path to the system configuration JSON file.

    Returns:
        dict: A dictionary containing the loaded system configuration, including:
            - times (list): Sorted list of time points.
            - prices (list): List of energy prices corresponding to each time point.
            - arrivals (list): Sorted list of vehicle arrival data.
            - chargers (list): List of charger configurations.
            - station_limit (float): The transformer limit of the station.
            - n_spots (int): The number of parking spots.
            - dt (float): The time step between consecutive time points.
            - test_number (int): The test number, defaults to 0 if not found.
            - parking_config (dict): The complete parking configuration.
            - car_brands (list): List of car brands, defaults to empty list if not found.
            - charger_types (dict): Dictionary of charger types, defaults to empty dict if not found.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        KeyError: If the configuration file is missing 'energy_prices' or 'arrivals'.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"System configuration file not found: {filepath}")

    with open(filepath, 'r') as f:
        data = json.load(f)

    # Validate basic data
    if "energy_prices" not in data or "arrivals" not in data:
        raise KeyError("The system configuration file must contain 'energy_prices' and 'arrivals'.")

    energy_prices = sorted(data["energy_prices"], key=lambda x: x["time"])
    times = [ep["time"] for ep in energy_prices]
    prices = [ep["price"] for ep in energy_prices]

    arrivals = sorted(data["arrivals"], key=lambda x: x["id"])
    
    # Flexible handling of parking_config
    if "parking_config" in data:
        parking_config = data["parking_config"]
        chargers = parking_config["chargers"]
        station_limit = parking_config["transformer_limit"]
        n_spots = parking_config["n_spots"]
    else:
        # Fallback if parking_config does not exist
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
            print(f"Warning: 'dt' not specified in system configuration and could not be inferred. Using dt={dt}")

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
    Loads a YAML file containing hyperparameters.

    Args:
        filepath (str): The full path to the hyperparameters YAML file.

    Returns:
        dict: A dictionary containing the hyperparameters.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the YAML file is invalid.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Hyperparameters file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_scatter_search_params(filepath: str) -> dict:
    """
    Loads the configuration parameters for the Scatter Search algorithm from a YAML file.

    Args:
        filepath (str): The full path to the Scatter Search parameters YAML file.

    Returns:
        dict: A dictionary containing the Scatter Search parameters.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the YAML file is invalid.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Scatter Search parameters file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        params = yaml.safe_load(f)
    return params