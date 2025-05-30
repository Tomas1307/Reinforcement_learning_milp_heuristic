import os
from datetime import datetime

def ensure_directory_exists(path: str):
    """
    Ensures that the specified directory exists. If it does not exist, it creates it.

    Args:
        path (str): The path of the directory to check/create.
    """
    os.makedirs(path, exist_ok=True)
    print(f"Directory ensured: {path}")

def get_timestamp_filepath(base_dir: str, prefix: str = "", suffix: str = ".json") -> str:
    """
    Generates a unique file path based on the current date and time.

    Args:
        base_dir (str): The base directory where the file will be created.
        prefix (str, optional): A prefix for the filename. Defaults to "".
        suffix (str, optional): The file suffix (extension). Defaults to ".json".

    Returns:
        str: The complete file path with the generated filename.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}{timestamp}{suffix}"
    filepath = os.path.join(base_dir, filename)
    ensure_directory_exists(base_dir)
    return filepath

def convert_time_to_index(time_value: float, times_list: list, dt: float) -> int:
    """
    Converts a real time value to its closest time step index.

    Args:
        time_value (float): The time value to convert (e.g., 8.0 for 8 AM).
        times_list (list): The list of all discrete time values in the environment.
        dt (float): The size of the time step (e.g., 0.25 for 15 minutes).

    Returns:
        int: The integer index of the time step.
    """

    if not times_list or dt == 0:
        return 0

    return int(round((time_value - times_list[0]) / dt))