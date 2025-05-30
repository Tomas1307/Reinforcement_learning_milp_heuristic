import numpy as np
from collections import defaultdict

def calculate_energy_satisfaction(delivered_energy: float, required_energy: float) -> float:
    """
    Calculates the energy satisfaction percentage for an EV or a group of EVs.

    Args:
        delivered_energy (float): The total energy delivered.
        required_energy (float): The total energy required.

    Returns:
        float: The energy satisfaction percentage (0.0 to 100.0).
               Returns 100.0 if required energy is 0 (no charging needed).
    """
    if required_energy <= 0:
        return 100.0
    return (delivered_energy / required_energy) * 100.0

def calculate_weighted_satisfaction(delivered_energy: float, required_energy: float,
                                    priority: float = 1.0, willingness_to_pay: float = 1.0) -> float:
    """
    Calculates energy satisfaction weighted by priority and willingness to pay.

    Args:
        delivered_energy (float): The total energy delivered.
        required_energy (float): The total energy required.
        priority (float): The priority factor of the EV (defaults to 1.0).
        willingness_to_pay (float): The willingness to pay of the EV (defaults to 1.0).

    Returns:
        float: The weighted satisfaction.
    """
    satisfaction_pct = calculate_energy_satisfaction(delivered_energy, required_energy)
    return satisfaction_pct * priority * willingness_to_pay

def calculate_assignment_ratio(assigned_count: int, total_vehicles: int) -> float:
    """
    Calculates the ratio of successfully assigned vehicles.

    Args:
        assigned_count (int): The number of vehicles that were assigned charging.
        total_vehicles (int): The total number of vehicles that arrived at the system.

    Returns:
        float: The assignment ratio (0.0 to 100.0).
               Returns 0.0 if there are no total vehicles.
    """
    if total_vehicles <= 0:
        return 0.0
    return (assigned_count / total_vehicles) * 100.0

def calculate_cost_per_timestep(power_delivered_kw: float, price_per_kwh: float, dt_hours: float) -> float:
    """
    Calculates the cost of energy delivered in a timestep.

    Args:
        power_delivered_kw (float): Power delivered in kW during the timestep.
        price_per_kwh (float): Price of energy in $/kWh for that timestep.
        dt_hours (float): Duration of the timestep in hours (e.g., 0.25 for 15 minutes).

    Returns:
        float: The total cost for the timestep.
    """
    return power_delivered_kw * dt_hours * price_per_kwh

def calculate_total_energy_cost(schedule: list, prices_per_timestep: list, dt_hours: float) -> float:
    """
    Calculates the total energy cost of a charging schedule.

    Args:
        schedule (list): List of tuples/lists representing charging assignments:
                         [(ev_id, t_idx, charger_id, slot_idx, power_kw), ...]
        prices_per_timestep (list): List of energy prices per timestep (indexed by t_idx).
        dt_hours (float): Duration of the timestep in hours.

    Returns:
        float: The total energy cost of the schedule.
    """
    total_cost = 0.0
    for _, t_idx, _, _, power_kw in schedule:
        if 0 <= t_idx < len(prices_per_timestep):
            total_cost += calculate_cost_per_timestep(power_kw, prices_per_timestep[t_idx], dt_hours)
    return total_cost

def calculate_peak_load(schedule: list, num_timesteps: int, dt_hours: float, station_limit_kw: float) -> dict:
    """
    Calculates the total station load per timestep and the peak load.

    Args:
        schedule (list): List of charging assignments:
                         [(ev_id, t_idx, charger_id, slot_idx, power_kw), ...]
        num_timesteps (int): The total number of timesteps in the simulation.
        dt_hours (float): Duration of the timestep in hours.
        station_limit_kw (float): The power limit of the station in kW.

    Returns:
        dict: A dictionary with 'load_profile_kw' (load per timestep), 'peak_load_kw',
              'peak_load_time_idx', and 'over_limit_duration_hours'.
    """
    load_profile = np.zeros(num_timesteps)
    for _, t_idx, _, _, power_kw in schedule:
        if 0 <= t_idx < num_timesteps:
            load_profile[t_idx] += power_kw

    peak_load_kw = np.max(load_profile) if load_profile.size > 0 else 0.0
    peak_load_time_idx = np.argmax(load_profile) if load_profile.size > 0 else -1

    over_limit_timesteps = np.sum(load_profile > station_limit_kw)
    over_limit_duration_hours = over_limit_timesteps * dt_hours

    return {
        "load_profile_kw": load_profile.tolist(),
        "peak_load_kw": peak_load_kw,
        "peak_load_time_idx": int(peak_load_time_idx),
        "over_limit_duration_hours": over_limit_duration_hours
    }

def calculate_metrics_by_priority(ev_metrics_raw: dict, priority_map: dict) -> dict:
    """
    Calculates aggregated metrics (energy, satisfaction) by priority level.

    Args:
        ev_metrics_raw (dict): Dictionary of metrics per EV, directly from the environment.
                               E.g.: {ev_id: {"required_energy": ..., "delivered_energy": ...}, ...}
        priority_map (dict): Mapping of ev_id to its priority level.

    Returns:
        dict: Aggregated metrics by priority level.
              E.g.: {1: {"count": ..., "required_energy": ..., "delivered_energy": ..., "satisfaction_pct": ...}, ...}
    """
    metrics_by_prio = defaultdict(lambda: {"count": 0, "required_energy": 0.0, "delivered_energy": 0.0})

    for ev_id, metrics in ev_metrics_raw.items():
        priority = priority_map.get(ev_id, 1)
        metrics_by_prio[priority]["count"] += 1
        metrics_by_prio[priority]["required_energy"] += metrics["required_energy"]
        metrics_by_prio[priority]["delivered_energy"] += metrics["delivered_energy"]

    result = {}
    for priority, data in metrics_by_prio.items():
        satisfaction = calculate_energy_satisfaction(data["delivered_energy"], data["required_energy"])
        result[priority] = {
            "count": data["count"],
            "required_energy": data["required_energy"],
            "delivered_energy": data["delivered_energy"],
            "satisfaction_pct": satisfaction
        }
    return result