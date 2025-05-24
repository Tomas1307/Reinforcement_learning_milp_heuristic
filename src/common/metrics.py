import numpy as np
from collections import defaultdict

def calculate_energy_satisfaction(delivered_energy: float, required_energy: float) -> float:
    """
    Calcula el porcentaje de satisfacción de energía para un EV o un grupo de EVs.

    Args:
        delivered_energy (float): La energía total entregada.
        required_energy (float): La energía total requerida.

    Returns:
        float: El porcentaje de satisfacción de energía (0.0 a 100.0).
               Retorna 100.0 si la energía requerida es 0 (no se necesitaba cargar).
    """
    if required_energy <= 0:
        return 100.0
    return (delivered_energy / required_energy) * 100.0

def calculate_weighted_satisfaction(delivered_energy: float, required_energy: float,
                                    priority: float = 1.0, willingness_to_pay: float = 1.0) -> float:
    """
    Calcula la satisfacción de energía ponderada por prioridad y disposición a pagar.

    Args:
        delivered_energy (float): La energía total entregada.
        required_energy (float): La energía total requerida.
        priority (float): El factor de prioridad del EV (por defecto 1.0).
        willingness_to_pay (float): La disposición a pagar del EV (por defecto 1.0).

    Returns:
        float: La satisfacción ponderada.
    """
    satisfaction_pct = calculate_energy_satisfaction(delivered_energy, required_energy)
    # Considerar cómo se combinan los pesos. Una simple multiplicación es un buen punto de partida.
    return satisfaction_pct * priority * willingness_to_pay

def calculate_assignment_ratio(assigned_count: int, total_vehicles: int) -> float:
    """
    Calcula el ratio de vehículos asignados con éxito.

    Args:
        assigned_count (int): Número de vehículos a los que se les asignó carga.
        total_vehicles (int): Número total de vehículos que llegaron al sistema.

    Returns:
        float: El ratio de asignación (0.0 a 100.0).
               Retorna 0.0 si no hay vehículos totales.
    """
    if total_vehicles <= 0:
        return 0.0
    return (assigned_count / total_vehicles) * 100.0

def calculate_cost_per_timestep(power_delivered_kw: float, price_per_kwh: float, dt_hours: float) -> float:
    """
    Calcula el costo de la energía entregada en un timestep.

    Args:
        power_delivered_kw (float): Potencia entregada en kW durante el timestep.
        price_per_kwh (float): Precio de la energía en $/kWh para ese timestep.
        dt_hours (float): Duración del timestep en horas (e.g., 0.25 para 15 minutos).

    Returns:
        float: El costo total para el timestep.
    """
    return power_delivered_kw * dt_hours * price_per_kwh

def calculate_total_energy_cost(schedule: list, prices_per_timestep: list, dt_hours: float) -> float:
    """
    Calcula el costo total de energía de un schedule de carga.

    Args:
        schedule (list): Lista de tuplas/listas representando las asignaciones de carga:
                         [(ev_id, t_idx, charger_id, slot_idx, power_kw), ...]
        prices_per_timestep (list): Lista de precios de energía por timestep (indexados por t_idx).
        dt_hours (float): Duración del timestep en horas.

    Returns:
        float: El costo total de energía del schedule.
    """
    total_cost = 0.0
    for _, t_idx, _, _, power_kw in schedule:
        if 0 <= t_idx < len(prices_per_timestep):
            total_cost += calculate_cost_per_timestep(power_kw, prices_per_timestep[t_idx], dt_hours)
    return total_cost

def calculate_peak_load(schedule: list, num_timesteps: int, dt_hours: float, station_limit_kw: float) -> dict:
    """
    Calcula la carga total de la estación por timestep y la carga pico.

    Args:
        schedule (list): Lista de asignaciones de carga:
                         [(ev_id, t_idx, charger_id, slot_idx, power_kw), ...]
        num_timesteps (int): El número total de timesteps en la simulación.
        dt_hours (float): Duración del timestep en horas.
        station_limit_kw (float): Límite de potencia de la estación en kW.

    Returns:
        dict: Un diccionario con 'load_profile' (carga por timestep), 'peak_load_kw',
              'peak_load_time_idx', y 'over_limit_duration_hours'.
    """
    load_profile = np.zeros(num_timesteps)
    for _, t_idx, _, _, power_kw in schedule:
        if 0 <= t_idx < num_timesteps:
            load_profile[t_idx] += power_kw

    peak_load_kw = np.max(load_profile) if load_profile.size > 0 else 0.0
    peak_load_time_idx = np.argmax(load_profile) if load_profile.size > 0 else -1

    # Calcular la duración en la que la carga excede el límite
    over_limit_timesteps = np.sum(load_profile > station_limit_kw)
    over_limit_duration_hours = over_limit_timesteps * dt_hours

    return {
        "load_profile_kw": load_profile.tolist(), # Convertir a lista para JSON serialización
        "peak_load_kw": peak_load_kw,
        "peak_load_time_idx": int(peak_load_time_idx),
        "over_limit_duration_hours": over_limit_duration_hours
    }

def calculate_metrics_by_priority(ev_metrics_raw: dict, priority_map: dict) -> dict:
    """
    Calcula métricas agregadas (energía, satisfacción) por nivel de prioridad.

    Args:
        ev_metrics_raw (dict): Diccionario de métricas por EV, directamente del entorno.
                               Ej: {ev_id: {"required_energy": ..., "delivered_energy": ...}, ...}
        priority_map (dict): Mapeo de ev_id a su nivel de prioridad.

    Returns:
        dict: Métricas agregadas por nivel de prioridad.
              Ej: {1: {"count": ..., "required_energy": ..., "delivered_energy": ..., "satisfaction": ...}, ...}
    """
    metrics_by_prio = defaultdict(lambda: {"count": 0, "required_energy": 0.0, "delivered_energy": 0.0})

    for ev_id, metrics in ev_metrics_raw.items():
        priority = priority_map.get(ev_id, 1) # Default a prioridad 1 si no está mapeado
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

# La función calculate_total_reward podría permanecer en el entorno si es muy específica de RL,
# o ser movida aquí si se desea reutilizar para MILP o otros algoritmos.
# Por ahora, es razonable mantenerla en EVChargingEnv si sus pesos son internos y fijos.
# Sin embargo, si quieres que MILP u otros módulos puedan "evaluar" una solución
# bajo la misma función de recompensa, entonces sí debería estar aquí.

# def calculate_total_reward(satisfaction_pct: float, cost: float, assign_ratio: float,
#                            energy_satisfaction_weight: float, energy_cost_weight: float,
#                            assigned_vehicle_reward: float, skipped_vehicle_penalty: float,
#                            assigned_count: int, skipped_count: int) -> float:
#     """
#     Calcula la recompensa total para un episodio en el entorno RL.
#     Esta es una versión más compleja que combina varias métricas.
#     """
#     norm_satisfaction = satisfaction_pct / 100.0
#     reward = (norm_satisfaction * energy_satisfaction_weight) - (cost * energy_cost_weight)
#     reward += (assigned_count * assigned_vehicle_reward)
#     reward -= (skipped_count * skipped_vehicle_penalty)
#     return reward