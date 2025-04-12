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

def calculate_solution_cost(schedule, prices, dt, times):
    """
    Calcula el costo total de una solución (schedule).
    
    Args:
        schedule: Diccionario con la programación {ev_id: [(t_start, t_end, charger, slot, power), ...]}
        prices: Lista de precios por intervalo.
        dt: Delta de tiempo (en horas).
        times: Lista de tiempos (para identificar el índice de tiempo).
    Returns:
        float: Costo total.
    """
    total_cost = 0.0
    for ev_intervals in schedule.values():
        for interval in ev_intervals:
            t_start, t_end, charger, slot, power = interval
            try:
                idx = times.index(t_start)
            except ValueError:
                idx = 0
            energy = power * dt
            total_cost += energy * prices[idx]
    return total_cost

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

def summarize_result(result_file, instance_file):
    """
    Lee un archivo de resultados y la instancia correspondiente,
    calcula el costo (BKS), el porcentaje de energía entregada y extrae la información necesaria.
    
    Para el tiempo computacional, se verifica:
      - Si existe "total_time" en extra_info (MILP),
      - o "rl_time" (RL),
      - o "time" (heurística),
    en ese orden.
    """
    config = load_instance(instance_file)
    with open(result_file, "r") as f:
        result_data = json.load(f)
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
    cost = calculate_solution_cost(schedule, config["prices"], config["dt"], config["times"])
    energy_pct = calculate_energy_percentage(schedule, config["dt"], config["arrivals"])
    summary = {
        "instancia": config.get("test_number", "N/A"),
        "BKS": round(cost, 2),
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
        result_files = glob.glob(os.path.join("results", "resultados_*.json"))
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
        result_files = glob.glob(os.path.join("heuristic_results", "test_system_*_resultado.json"))
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
