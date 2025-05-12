#!/usr/bin/env python
import os
import re
import glob
import json
import pandas as pd

def load_instance(json_path):
    """
    Carga la configuración de una instancia a partir de un archivo JSON.
    Se espera que el JSON contenga las claves:
      - "energy_prices": lista de diccionarios con "time" y "price"
      - "arrivals": lista de vehículos (cada uno con al menos "id" y "required_energy")
      - "parking_config": diccionario con "chargers", "transformer_limit" y "n_spots"
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
    dt = times[1] - times[0] if len(times) > 1 else 0.25
    return {
        "times": times,
        "prices": prices,
        "arrivals": arrivals,
        "chargers": chargers,
        "station_limit": station_limit,
        "dt": dt,
        "n_spots": n_spots,
        "test_number": data.get("test_number", os.path.basename(json_path))
    }


def calculate_solution_cost(schedule, prices, dt, times, arrivals=None, include_penalty=True):
    """
    Calcula el costo total de una solución (schedule), incluyendo penalización
    por energía no entregada, usando exactamente el mismo método que heuristic.py.

    Args:
        schedule: Diccionario con la programación {ev_id: [(t_start, t_end, charger, slot, power), ...]}
        prices: Lista de precios por intervalo.
        dt: Delta de tiempo (en horas).
        times: Lista de tiempos (para identificar el índice de tiempo).
        arrivals: Lista de vehículos de la instancia (requerido si include_penalty=True)
        include_penalty: Si es True, incluye penalización por energía no entregada
    Returns:
        float: Costo total.
    """
    # Costo operacional (original)
    costo_operacion = 0.0
    for ev_id, ev_intervals in schedule.items():
        for interval in ev_intervals:
            t_start, t_end, charger, slot, power = interval
            try:
                idx = times.index(t_start)
            except ValueError:
                idx = 0
            energy = power * dt
            costo_operacion += energy * prices[idx]

    # Si no se requiere penalización o no hay arrivals, retornar solo costo operacional
    if not include_penalty or arrivals is None:
        return costo_operacion

    # Cálculo de penalización EXACTAMENTE igual que en heuristic.py

    # Calcular energía entregada por EV
    energia_entregada = {}
    for ev_id, ev_intervals in schedule.items():
        energia_entregada[ev_id] = sum(interval[4] * dt for interval in ev_intervals)

    # Obtener energía requerida para todos los EVs
    energia_requerida = {}
    for ev in arrivals:
        ev_id = str(ev.get("id"))  # Asegurar que las claves sean strings para coincidir con schedule
        energia_requerida[ev_id] = ev.get("required_energy", 0)

    # Penalización por energía no entregada, siguiendo EXACTAMENTE la lógica de heuristic.py
    penalizacion_base = 1000.0
    costo_penalizacion = 0.0

    for ev_id, required in energia_requerida.items():
        delivered = energia_entregada.get(ev_id, 0)
        energia_no_entregada = max(0, required - delivered)

        if energia_no_entregada > 0:
            # Penalización progresiva: mayor cuanto mayor sea el porcentaje
            porcentaje_no_entregado = energia_no_entregada / required if required > 0 else 0
            factor_penalizacion = penalizacion_base * porcentaje_no_entregado
            costo_penalizacion += energia_no_entregada * factor_penalizacion

    return costo_operacion + costo_penalizacion

def calculate_energy_percentage(schedule, dt, arrivals):
    """
    Calcula el porcentaje de energía entregada respecto a la energía requerida total.
    
    Args:
        schedule: Diccionario del schedule {ev_id: [(t_start, t_end, charger, slot, power), ...]}
        dt: Delta de tiempo (horas).
        arrivals: Lista de vehículos de la instancia; se espera que cada vehículo tenga "required_energy".
    Returns:
        float: Porcentaje de energía entregada (redondeado a dos decimales).
    """
    total_delivered = 0.0
    for ev_intervals in schedule.values():
        for interval in ev_intervals:
            power = interval[4]
            total_delivered += power * dt
    total_required = sum(ev.get("required_energy", 0) for ev in arrivals)
    if total_required > 0:
        pct = (total_delivered / total_required) * 100
    else:
        pct = 0.0
    return round(pct, 2)


def summarize_result(result_file, instance_file, include_penalty=True):
    """
    Lee un archivo de resultados y la instancia correspondiente,
    calcula el costo (BKS), el porcentaje de energía entregada y extrae la información necesaria.

    Args:
        result_file: Ruta al archivo de resultados
        instance_file: Ruta al archivo de instancia
        include_penalty: Si es True, incluye penalización por energía no entregada en el costo
    """
    config = load_instance(instance_file)
    with open(result_file, "r") as f:
        result_data = json.load(f)

    # Convertir el formato del schedule si es necesario
    # El formato de schedule en result_data puede ser {ev_id: [...]} o una lista de tuplas
    schedule = result_data.get("schedule", {})

    extra_info = result_data.get("extra_info", {})
    if "total_time" in extra_info:
        total_time = extra_info["total_time"]
    elif "rl_time" in extra_info:
        total_time = extra_info["rl_time"]
    elif "time" in extra_info:
        total_time = extra_info["time"]
    else:
        total_time = None

    # Calcular el costo con penalización (usando el mismo método que heuristic.py)
    cost_with_penalty = calculate_solution_cost(
        schedule,
        config["prices"],
        config["dt"],
        config["times"],
        config["arrivals"],
        include_penalty=True
    )

    # Calcular el costo sin penalización
    cost_no_penalty = calculate_solution_cost(
        schedule,
        config["prices"],
        config["dt"],
        config["times"],
        include_penalty=False
    )

    energy_pct = calculate_energy_percentage(schedule, config["dt"], config["arrivals"])

    # Verificar si existe una estadística "costo_total" en result_data (viene de heuristic.py)
    heuristic_cost = None
    if "estadisticas" in result_data and "costo_total" in result_data["estadisticas"]:
        heuristic_cost = result_data["estadisticas"]["costo_total"]

    summary = {
        "instancia": config.get("test_number", "N/A"),
        "BKS": round(cost_with_penalty, 2),
        "BKS_sin_penalizacion": round(cost_no_penalty, 2),
        "costo_original_heuristica": heuristic_cost,  # Agregar el costo calculado por la heurística para comparar
        "tiempo": total_time,
        "energía entregada (%)": energy_pct,
        "vehículos": len(config["arrivals"]),
        "slots": config["n_spots"],
        "cargadores": len(config["chargers"])
    }
    return summary

def main():
    # Cargar las instancias originales
    instance_paths = glob.glob(os.path.join("data", "test_system_*.json"))
    if not instance_paths:
        print("No se encontraron archivos de instancias en 'data' con patrón 'test_system_*.json'")
        return
    instance_files = {}
    for path in instance_paths:
        basename = os.path.basename(path)
        m = re.search(r"test_system_(\d+)\.json", basename)
        if m:
            instance_number = int(m.group(1))
            instance_files[instance_number] = path

    print("Seleccione 1 para resultados RL+MILP o 2 para resultados de la heurística:")
    option = input("Opción: ").strip()
    summaries = []
    if option == "1":
        result_files = glob.glob(os.path.join("results/reinforcement_milp", "resultados_*.json"))
        if not result_files:
            print("No se encontraron archivos RL+MILP en results con patrón 'resultados_*.json'")
            return
        for res_file in result_files:
            basename = os.path.basename(res_file)
            m = re.search(r"instancia_(\d+)", basename)
            if not m:
                print(f"No se pudo extraer el número de instancia de {basename}")
                continue
            instance_number = int(m.group(1))
            if instance_number not in instance_files:
                print(f"No se encontró archivo de instancia para la instancia {instance_number}")
                continue
            summary = summarize_result(res_file, instance_files[instance_number])
            summary["solución"] = "MILP" if "milp" in basename.lower() else "RL"
            summaries.append(summary)
    elif option == "2":
        result_files = glob.glob(os.path.join("results/heuristic_results", "test_system_*_resultado.json"))
        if not result_files:
            print("No se encontraron archivos de heurística en heuristic_results con patrón 'test_system_*_resultado.json'")
            return
        for res_file in result_files:
            basename = os.path.basename(res_file)
            m = re.search(r"test_system_(\d+)_resultado\.json", basename)
            if not m:
                print(f"No se pudo extraer el número de instancia de {basename}")
                continue
            instance_number = int(m.group(1))
            if instance_number not in instance_files:
                print(f"No se encontró archivo de instancia para la instancia {instance_number}")
                continue
            summary = summarize_result(res_file, instance_files[instance_number])
            summary["solución"] = "Heurística"
            summaries.append(summary)
    else:
        print("Opción no válida. Debe ser 1 o 2.")
        return

    if not summaries:
        print("No se generó ningún resumen. Verifica que los archivos tengan los nombres esperados.")
        return

    df = pd.DataFrame(summaries)
    df = df.sort_values(by=["instancia", "solución"]).reset_index(drop=True)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print("Resumen de Resultados:")
    print(df.to_string())

if __name__ == "__main__":
    main()
