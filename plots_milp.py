import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def compare_solution_costs(results_dir="./results", data_dir="./data"):
    """
    Compara los costos de las soluciones RL vs RL+MILP refinado para todas las instancias disponibles.
    
    Args:
        results_dir: Directorio donde se encuentran los resultados
        data_dir: Directorio donde se encuentran los archivos de configuración
        
    Returns:
        Diccionario con la comparación de costos por instancia
    """
    # Encontrar todos los archivos de resultados
    rl_files = {}
    milp_files = {}
    
    for filename in os.listdir(results_dir):
        if filename.startswith("resultados_rl_instancia_"):
            instance_id = filename.replace("resultados_rl_instancia_", "").replace(".json", "")
            rl_files[instance_id] = os.path.join(results_dir, filename)
        elif filename.startswith("resultados_milp_instancia_"):
            instance_id = filename.replace("resultados_milp_instancia_", "").replace(".json", "")
            milp_files[instance_id] = os.path.join(results_dir, filename)
    
    # Para cada instancia con ambos resultados, calcular costos
    comparison = {}
    
    for instance_id in sorted(set(rl_files.keys()) & set(milp_files.keys())):
        # Cargar configuración para obtener precios
        config_file = os.path.join(data_dir, f"test_system_{instance_id}.json")
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                
            # Extraer tiempos y precios
            energy_prices = sorted(config["energy_prices"], key=lambda x: x["time"])
            times = [ep["time"] for ep in energy_prices]
            prices = [ep["price"] for ep in energy_prices]
            
            # Calcular dt
            if len(times) > 1:
                dt = times[1] - times[0]
            else:
                dt = 0.25
                
            # Cargar soluciones
            with open(rl_files[instance_id], 'r') as f:
                rl_data = json.load(f)
                if "schedule" in rl_data:
                    rl_schedule = rl_data["schedule"]
                else:
                    rl_schedule = rl_data
            
            with open(milp_files[instance_id], 'r') as f:
                milp_data = json.load(f)
                if "schedule" in milp_data:
                    milp_schedule = milp_data["schedule"]
                else:
                    milp_schedule = milp_data
            
            # Calcular costo total por solución
            rl_cost = 0
            rl_energy = 0
            for ev_id, intervals in rl_schedule.items():
                for interval in intervals:
                    t_start = interval[0]
                    power = interval[4]
                    # Encontrar el índice del tiempo más cercano
                    t_idx = min(range(len(times)), key=lambda i: abs(times[i] - t_start))
                    price = prices[t_idx]
                    energy = power * dt
                    rl_energy += energy
                    rl_cost += energy * price
            
            milp_cost = 0
            milp_energy = 0
            for ev_id, intervals in milp_schedule.items():
                for interval in intervals:
                    t_start = interval[0]
                    power = interval[4]
                    # Encontrar el índice del tiempo más cercano
                    t_idx = min(range(len(times)), key=lambda i: abs(times[i] - t_start))
                    price = prices[t_idx]
                    energy = power * dt
                    milp_energy += energy
                    milp_cost += energy * price
            
            # Extraer tiempos de ejecución si están disponibles
            rl_time = rl_data.get("extra_info", {}).get("rl_time", 0)
            milp_time = milp_data.get("extra_info", {}).get("milp_time", 0)
            total_time = rl_time + milp_time
            
            # Calcular mejora
            if rl_cost > 0:
                cost_improvement = (rl_cost - milp_cost) / rl_cost * 100
            else:
                cost_improvement = 0
                
            # Calcular energía satisfecha
            total_energy_required = 0
            for arr in config["arrivals"]:
                total_energy_required += arr["required_energy"]
            
            rl_energy_percent = rl_energy / total_energy_required * 100
            milp_energy_percent = milp_energy / total_energy_required * 100
            
            # Número de EVs atendidos
            rl_evs_served = len(rl_schedule)
            milp_evs_served = len(milp_schedule)
            total_evs = len(config["arrivals"])
            
            # Guardar comparación
            comparison[instance_id] = {
                "instance_id": instance_id,
                "total_evs": total_evs,
                "rl_cost": rl_cost,
                "milp_cost": milp_cost,
                "cost_improvement": cost_improvement,
                "rl_time": rl_time,
                "milp_time": milp_time,
                "total_time": total_time,
                "rl_energy": rl_energy,
                "milp_energy": milp_energy,
                "total_energy_required": total_energy_required,
                "rl_energy_percent": rl_energy_percent,
                "milp_energy_percent": milp_energy_percent,
                "rl_evs_served": rl_evs_served,
                "milp_evs_served": milp_evs_served
            }
            
        except Exception as e:
            print(f"Error procesando instancia {instance_id}: {e}")
    
    return comparison

def plot_solution_comparison(comparison):
    """
    Visualiza la comparación de costos entre soluciones RL y MILP.
    
    Args:
        comparison: Diccionario con la comparación de costos por instancia
    """
    # Preparar datos para gráficas
    instance_ids = []
    rl_costs = []
    milp_costs = []
    improvements = []
    rl_times = []
    milp_times = []
    rl_energy_pcts = []
    milp_energy_pcts = []
    rl_evs_pcts = []
    milp_evs_pcts = []
    
    for instance_id, data in sorted(comparison.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
        instance_ids.append(instance_id)
        rl_costs.append(data["rl_cost"])
        milp_costs.append(data["milp_cost"])
        improvements.append(data["cost_improvement"])
        rl_times.append(data["rl_time"])
        milp_times.append(data["milp_time"])
        rl_energy_pcts.append(data["rl_energy_percent"])
        milp_energy_pcts.append(data["milp_energy_percent"])
        rl_evs_pcts.append(data["rl_evs_served"] / data["total_evs"] * 100)
        milp_evs_pcts.append(data["milp_evs_served"] / data["total_evs"] * 100)
    
    # Crear gráfica de comparación de costos
    plt.figure(figsize=(14, 10))
    
    # 1. Comparación de costos
    plt.subplot(2, 2, 1)
    x = np.arange(len(instance_ids))
    width = 0.35
    
    plt.bar(x - width/2, rl_costs, width, label='RL', color='cornflowerblue')
    plt.bar(x + width/2, milp_costs, width, label='RL+MILP', color='darkorange')
    
    plt.xlabel('Instancia')
    plt.ylabel('Costo ($)')
    plt.title('Comparación de Costos: RL vs RL+MILP')
    plt.xticks(x, instance_ids, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Mejora porcentual
    plt.subplot(2, 2, 2)
    plt.bar(instance_ids, improvements, color='green')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    plt.xlabel('Instancia')
    plt.ylabel('Mejora (%)')
    plt.title('Mejora porcentual de costo con MILP')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 3. Energía entregada (%)
    plt.subplot(2, 2, 3)
    
    plt.bar(x - width/2, rl_energy_pcts, width, label='RL', color='cornflowerblue')
    plt.bar(x + width/2, milp_energy_pcts, width, label='RL+MILP', color='darkorange')
    
    plt.xlabel('Instancia')
    plt.ylabel('Energía Entregada (%)')
    plt.title('Porcentaje de Energía Entregada')
    plt.xticks(x, instance_ids, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. EVs atendidos (%)
    plt.subplot(2, 2, 4)
    
    plt.bar(x - width/2, rl_evs_pcts, width, label='RL', color='cornflowerblue')
    plt.bar(x + width/2, milp_evs_pcts, width, label='RL+MILP', color='darkorange')
    
    plt.xlabel('Instancia')
    plt.ylabel('EVs Atendidos (%)')
    plt.title('Porcentaje de EVs Atendidos')
    plt.xticks(x, instance_ids, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./plots/cost_comparison.png')
    plt.show()
    
    # Crear gráfica de tiempos de ejecución
    plt.figure(figsize=(12, 6))
    
    plt.bar(x - width/2, rl_times, width, label='RL', color='cornflowerblue')
    plt.bar(x + width/2, milp_times, width, label='MILP', color='darkorange')
    
    plt.xlabel('Instancia')
    plt.ylabel('Tiempo (segundos)')
    plt.title('Tiempos de Ejecución: RL vs MILP')
    plt.xticks(x, instance_ids, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./plots/time_comparison.png')
    plt.show()

def analyze_parking_congestion(config):
    """
    Analiza la congestión del parqueadero a lo largo del tiempo.
    
    Args:
        config: Configuración del sistema
        
    Returns:
        Diccionario con análisis de congestión por hora
    """
    # Extraer datos de configuración
    arrivals = config["arrivals"]
    times = config["times"]
    n_spots = config["n_spots"]
    dt = config["dt"] if "dt" in config else (times[1] - times[0]) if len(times) > 1 else 0.25
    
    # Contar vehículos presentes en cada intervalo
    vehicles_present = defaultdict(int)
    interval_to_time = {}
    
    for t_idx, t in enumerate(times):
        interval_to_time[t_idx] = t
        for ev in arrivals:
            if ev["arrival_time"] <= t < ev["departure_time"]:
                vehicles_present[t_idx] += 1
    
    # Crear análisis de congestión
    congestion_analysis = {
        "times": times,
        "vehicles_present": [vehicles_present[t_idx] for t_idx in range(len(times))],
        "parking_capacity": n_spots,
        "overflow": [max(0, vehicles_present[t_idx] - n_spots) for t_idx in range(len(times))],
        "utilization": [min(1.0, vehicles_present[t_idx] / n_spots) for t_idx in range(len(times))],
        "dt": dt
    }
    
    return congestion_analysis

def plot_parking_congestion(config, title_prefix=""):
    """
    Visualiza la congestión del parqueadero a lo largo del tiempo.
    
    Args:
        config: Configuración del sistema
        title_prefix: Prefijo para los títulos de las gráficas
    """
    # Obtener análisis de congestión
    congestion = analyze_parking_congestion(config)
    
    # Crear gráfica
    plt.figure(figsize=(14, 8))
    
    # Gráfica 1: Vehículos presentes vs capacidad
    plt.subplot(2, 1, 1)
    plt.plot(congestion["times"], congestion["vehicles_present"], 'b-', linewidth=2, label='Vehículos presentes')
    plt.axhline(y=congestion["parking_capacity"], color='r', linestyle='--', label=f'Capacidad ({congestion["parking_capacity"]} slots)')
    
    plt.fill_between(congestion["times"], congestion["vehicles_present"], 
                     [congestion["parking_capacity"]]*len(congestion["times"]),
                     where=[v > congestion["parking_capacity"] for v in congestion["vehicles_present"]],
                     color='red', alpha=0.3, label='Exceso de demanda')
    
    plt.xlabel('Tiempo (horas)')
    plt.ylabel('Número de vehículos')
    plt.title(f'{title_prefix}Congestión del parqueadero a lo largo del tiempo')
    plt.grid(True)
    plt.legend()
    
    # Gráfica 2: Porcentaje de utilización y overflow
    plt.subplot(2, 1, 2)
    plt.plot(congestion["times"], [u*100 for u in congestion["utilization"]], 'g-', linewidth=2, label='Utilización (%)')
    plt.plot(congestion["times"], congestion["overflow"], 'r-', linewidth=2, label='Vehículos sin espacio')
    
    plt.xlabel('Tiempo (horas)')
    plt.ylabel('Utilización (%) / Vehículos excedentes')
    plt.title(f'{title_prefix}Utilización del parqueadero y vehículos sin espacio')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./plots/parking_congestion.png')
    plt.show()
    
    # Calcular estadísticas adicionales
    max_vehicles = max(congestion["vehicles_present"])
    max_overflow = max(congestion["overflow"])
    avg_utilization = np.mean(congestion["utilization"]) * 100
    
    overflow_times = [t for t, o in zip(congestion["times"], congestion["overflow"]) if o > 0]
    overflow_durations = []
    if overflow_times:
        current_start = overflow_times[0]
        last_time = overflow_times[0]
        
        for t in overflow_times[1:]:
            if t - last_time > congestion["dt"] * 1.1:  # Tolerancia para inexactitudes numéricas
                overflow_durations.append((current_start, last_time))
                current_start = t
            last_time = t
            
        overflow_durations.append((current_start, last_time))
    
    # Imprimir estadísticas
    print(f"\nEstadísticas de congestión del parqueadero:")
    print(f"- Capacidad: {congestion['parking_capacity']} slots")
    print(f"- Máximo número de vehículos simultáneos: {max_vehicles}")
    print(f"- Máximo exceso de demanda: {max_overflow} vehículos")
    print(f"- Utilización promedio: {avg_utilization:.2f}%")
    
    if overflow_durations:
        print(f"\nPeríodos con exceso de demanda:")
        for start, end in overflow_durations:
            print(f"- Desde {start} hasta {end} horas")
        
        total_overflow_time = sum(end - start for start, end in overflow_durations)
        total_time = congestion["times"][-1] - congestion["times"][0]
        overflow_percent = total_overflow_time / total_time * 100
        
        print(f"\nTiempo total con exceso de demanda: {total_overflow_time:.2f} horas ({overflow_percent:.2f}% del tiempo)")
    else:
        print("\nNo hay períodos con exceso de demanda.")
    
    return congestion

# --- Nuevas funciones para análisis avanzado ---

def analyze_rejected_vehicles(config):
    """
    Analiza los vehículos que serían rechazados por falta de espacio de estacionamiento
    y calcula el costo de oportunidad.
    
    Args:
        config: Configuración del sistema
        
    Returns:
        Diccionario con análisis detallado de vehículos rechazados
    """
    # Extraer datos relevantes
    arrivals = config["arrivals"]
    times = config["times"]
    prices = config["prices"]
    n_spots = config["n_spots"]
    dt = config["dt"] if "dt" in config else (times[1] - times[0]) if len(times) > 1 else 0.25
    
    # Para cada intervalo de tiempo, determinar qué vehículos están presentes
    vehicles_by_time = defaultdict(list)
    for t_idx, t in enumerate(times):
        for ev in arrivals:
            if ev["arrival_time"] <= t < ev["departure_time"]:
                vehicles_by_time[t_idx].append(ev["id"])
    
    # Para cada intervalo, determinar cuáles serían rechazados (más allá de la capacidad)
    rejected_by_time = {}
    for t_idx, present_vehicles in vehicles_by_time.items():
        if len(present_vehicles) > n_spots:
            # Ordenar por algún criterio (ej: tiempo de llegada)
            # Aquí asumimos que los primeros n_spots vehículos son aceptados
            # y el resto rechazados
            rejected_by_time[t_idx] = present_vehicles[n_spots:]
    
    # Contabilizar rechazos por vehículo (cuántos intervalos rechazado)
    rejection_count = defaultdict(int)
    for t_idx, rejected in rejected_by_time.items():
        for ev_id in rejected:
            rejection_count[ev_id] += 1
    
    # Identificar vehículos completamente rechazados
    # (aquellos rechazados en todos sus intervalos de presencia)
    completely_rejected = []
    for ev in arrivals:
        ev_id = ev["id"]
        # Contar cuántos intervalos estaría presente
        presence_intervals = sum(1 for t in times 
                                if ev["arrival_time"] <= t < ev["departure_time"])
        # Si está rechazado en todos los intervalos
        if ev_id in rejection_count and rejection_count[ev_id] == presence_intervals:
            completely_rejected.append(ev_id)
    
    # Calcular costo de oportunidad
    # Asumimos un margen del 50% sobre el costo de la energía
    opportunity_cost = 0
    revenue_potential = {}
    
    for ev_id in completely_rejected:
        ev = next(e for e in arrivals if e["id"] == ev_id)
        energy_required = ev["required_energy"]
        
        # Calcular el precio promedio durante su estadía
        stay_prices = []
        for t_idx, t in enumerate(times):
            if ev["arrival_time"] <= t < ev["departure_time"]:
                stay_prices.append(prices[t_idx])
        
        avg_price = np.mean(stay_prices) if stay_prices else np.mean(prices)
        
        # Calcular ingreso potencial con margen del 50%
        potential_revenue = energy_required * avg_price * 0.5
        revenue_potential[ev_id] = potential_revenue
        opportunity_cost += potential_revenue
    
    # Recolectar características de vehículos rechazados
    rejected_features = {
        "stay_duration": [],
        "energy_required": [],
        "arrival_times": [],
        "departure_times": [],
        "revenue_potential": []
    }
    
    for ev_id in completely_rejected:
        ev = next(e for e in arrivals if e["id"] == ev_id)
        rejected_features["stay_duration"].append(ev["departure_time"] - ev["arrival_time"])
        rejected_features["energy_required"].append(ev["required_energy"])
        rejected_features["arrival_times"].append(ev["arrival_time"])
        rejected_features["departure_times"].append(ev["departure_time"])
        rejected_features["revenue_potential"].append(revenue_potential[ev_id])
    
    # Agregar estadísticas por hora del día
    # Convertir tiempos a horas del día (0-24)
    hour_bins = np.arange(0, 25)
    arrivals_by_hour = np.zeros(24)
    rejections_by_hour = np.zeros(24)
    
    for ev in arrivals:
        arrival_hour = int(ev["arrival_time"] % 24)
        arrivals_by_hour[arrival_hour] += 1
        
        if ev["id"] in completely_rejected:
            rejections_by_hour[arrival_hour] += 1
    
    # Preparar resultados
    result = {
        "rejection_by_time": rejected_by_time,
        "rejection_count": dict(rejection_count),
        "completely_rejected": completely_rejected,
        "total_vehicles": len(arrivals),
        "rejection_rate": len(completely_rejected) / len(arrivals) if arrivals else 0,
        "opportunity_cost": opportunity_cost,
        "revenue_potential": revenue_potential,
        "rejected_features": rejected_features,
        "times": times,
        "arrivals_by_hour": arrivals_by_hour.tolist(),
        "rejections_by_hour": rejections_by_hour.tolist()
    }
    
    return result


def plot_rejected_vehicles_analysis(config, title_prefix=""):
    """
    Visualiza el análisis de vehículos rechazados y costo de oportunidad.
    
    Args:
        config: Configuración del sistema
        title_prefix: Prefijo para los títulos de las gráficas
    """
    # Obtener análisis
    rejected_analysis = analyze_rejected_vehicles(config)
    
    # Crear gráficas
    plt.figure(figsize=(14, 10))
    
    # 1. Rechazos por hora del día
    plt.subplot(2, 2, 1)
    hours = np.arange(24)
    width = 0.35
    
    plt.bar(hours - width/2, rejected_analysis["arrivals_by_hour"], width,
            label='Total llegadas', color='cornflowerblue')
    plt.bar(hours + width/2, rejected_analysis["rejections_by_hour"], width,
            label='Rechazados', color='crimson')
    
    plt.xlabel('Hora del día')
    plt.ylabel('Número de vehículos')
    plt.title(f'{title_prefix}Llegadas y rechazos por hora del día')
    plt.xticks(hours, [f"{h}:00" for h in hours], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Características de vehículos rechazados
    plt.subplot(2, 2, 2)
    
    if rejected_analysis["completely_rejected"]:
        # Calcular ratios para normalizar en la visualización
        max_duration = max(rejected_analysis["rejected_features"]["stay_duration"])
        max_energy = max(rejected_analysis["rejected_features"]["energy_required"])
        max_revenue = max(rejected_analysis["rejected_features"]["revenue_potential"])
        
        normalized_duration = [d/max_duration for d in rejected_analysis["rejected_features"]["stay_duration"]]
        normalized_energy = [e/max_energy for e in rejected_analysis["rejected_features"]["energy_required"]]
        normalized_revenue = [r/max_revenue for r in rejected_analysis["rejected_features"]["revenue_potential"]]
        
        # Ordenar por potencial de ingreso
        indices = np.argsort(normalized_revenue)[::-1]  # Mayor a menor
        
        # Limitar a mostrar máximo 20 vehículos para claridad
        indices = indices[:min(20, len(indices))]
        
        # Datos ordenados para visualización
        ev_ids = [rejected_analysis["completely_rejected"][i] for i in indices]
        duration_data = [normalized_duration[i] for i in indices]
        energy_data = [normalized_energy[i] for i in indices]
        revenue_data = [normalized_revenue[i] for i in indices]
        
        # Crear gráfica
        x = np.arange(len(ev_ids))
        
        plt.bar(x - width, duration_data, width, label='Duración estadía (norm.)', color='skyblue')
        plt.bar(x, energy_data, width, label='Energía requerida (norm.)', color='lightgreen')
        plt.bar(x + width, revenue_data, width, label='Ingreso potencial (norm.)', color='coral')
        
        plt.xlabel('ID de vehículo rechazado')
        plt.ylabel('Valor normalizado')
        plt.title(f'{title_prefix}Características de vehículos rechazados')
        plt.xticks(x, [f"EV {id}" for id in ev_ids], rotation=90)
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, "No hay vehículos completamente rechazados", 
                 ha='center', va='center', fontsize=12)
        plt.axis('off')
    
    # 3. Congestión y rechazos a lo largo del tiempo
    plt.subplot(2, 1, 2)
    
    # Calcular vehículos presentes y rechazados en cada tiempo
    vehicles_present = []
    vehicles_rejected = []
    times = rejected_analysis["times"]
    
    for t_idx, t in enumerate(times):
        # Contar vehículos presentes
        present = sum(1 for ev in config["arrivals"] 
                     if ev["arrival_time"] <= t < ev["departure_time"])
        vehicles_present.append(present)
        
        # Contar vehículos rechazados en este tiempo
        rejected = len(rejected_analysis["rejection_by_time"].get(t_idx, []))
        vehicles_rejected.append(rejected)
    
    # Graficar
    plt.plot(times, vehicles_present, 'b-', linewidth=2, label='Vehículos presentes')
    plt.fill_between(times, [config["n_spots"]]*len(times), vehicles_present,
                     where=[v > config["n_spots"] for v in vehicles_present],
                     color='red', alpha=0.3, label='Exceso de demanda')
    plt.bar(times, vehicles_rejected, width=0.08, color='darkred', 
            label='Vehículos rechazados', alpha=0.7)
    
    plt.axhline(y=config["n_spots"], color='r', linestyle='--', 
                label=f'Capacidad ({config["n_spots"]} slots)')
    
    plt.xlabel('Tiempo (horas)')
    plt.ylabel('Número de vehículos')
    plt.title(f'{title_prefix}Congestión y rechazos a lo largo del tiempo')
    plt.grid(True)
    plt.legend()
    
    # Ajustar disposición
    plt.tight_layout()
    plt.savefig('./plots/rejected_vehicles_analysis.png')
    plt.show()
    
    # Imprimir estadísticas adicionales
    print(f"\nEstadísticas de vehículos rechazados:")
    print(f"- Total de vehículos: {rejected_analysis['total_vehicles']}")
    print(f"- Vehículos completamente rechazados: {len(rejected_analysis['completely_rejected'])} "
          f"({rejected_analysis['rejection_rate']*100:.2f}%)")
    print(f"- Costo de oportunidad total: ${rejected_analysis['opportunity_cost']:.2f}")
    
    if rejected_analysis["completely_rejected"]:
        avg_duration = np.mean(rejected_analysis["rejected_features"]["stay_duration"])
        avg_energy = np.mean(rejected_analysis["rejected_features"]["energy_required"])
        avg_revenue = np.mean(rejected_analysis["rejected_features"]["revenue_potential"])
        
        print(f"\nCaracterísticas promedio de vehículos rechazados:")
        print(f"- Duración de estadía: {avg_duration:.2f} horas")
        print(f"- Energía requerida: {avg_energy:.2f} kWh")
        print(f"- Ingreso potencial: ${avg_revenue:.2f}")
    
    # Horas con mayor rechazo
    rejection_hours = np.argsort(rejected_analysis["rejections_by_hour"])[::-1]
    top_rejection_hours = [h for h in rejection_hours if rejected_analysis["rejections_by_hour"][h] > 0][:3]
    
    if top_rejection_hours:
        print(f"\nHoras con mayor tasa de rechazo:")
        for h in top_rejection_hours:
            rejection_rate_h = (rejected_analysis["rejections_by_hour"][h] / 
                               rejected_analysis["arrivals_by_hour"][h] * 100 
                               if rejected_analysis["arrivals_by_hour"][h] > 0 else 0)
            print(f"- {h}:00: {rejected_analysis['rejections_by_hour'][h]:.0f} rechazos "
                  f"({rejection_rate_h:.2f}% de las llegadas en esta hora)")
    
    return rejected_analysis


def analyze_transformer_usage(config, solution=None):
    """
    Analiza la utilización del transformador a lo largo del tiempo
    y la compara con el límite establecido.
    
    Args:
        config: Configuración del sistema
        solution: Solución RL o MILP (opcional)
        
    Returns:
        Diccionario con análisis de uso del transformador
    """
    # Extraer datos relevantes
    times = config["times"]
    station_limit = config.get("station_limit", 0)
    if "transformer_limit" in config:
        station_limit = config["transformer_limit"]
    elif "parking_config" in config and "transformer_limit" in config["parking_config"]:
        station_limit = config["parking_config"]["transformer_limit"]
    
    # Si no se proporciona solución, solo analizamos potencial de demanda
    if solution is None:
        # Estimamos demanda potencial máxima por intervalo
        # Suponemos una potencia promedio de 7 kW por EV
        avg_power_per_ev = 7  # kW
        
        # Calculamos EVs presentes en cada intervalo
        evs_by_time = []
        for t in times:
            evs_present = sum(1 for ev in config["arrivals"] 
                             if ev["arrival_time"] <= t < ev["departure_time"])
            evs_by_time.append(evs_present)
        
        # Potencia potencial demandada
        potential_power = [min(evs * avg_power_per_ev, station_limit) for evs in evs_by_time]
        
        # Preparar resultados
        result = {
            "times": times,
            "transformer_limit": station_limit,
            "evs_by_time": evs_by_time,
            "potential_power": potential_power,
            "avg_potential_usage": np.mean(potential_power) / station_limit if station_limit > 0 else 0,
            "max_potential_usage": max(potential_power) / station_limit if station_limit > 0 else 0
        }
    else:
        # Analizar utilización real basada en la solución
        # Si solution es un diccionario en formato {ev_id: [(t_start, t_end, charger, slot, power), ...]}
        power_usage = np.zeros(len(times))
        
        for ev_id, intervals in solution.items():
            for interval in intervals:
                t_start, t_end, charger, slot, power = interval
                
                # Encontrar índices de tiempo correspondientes
                start_idx = next((i for i, t in enumerate(times) if t >= t_start), None)
                end_idx = next((i for i, t in enumerate(times) if t >= t_end), len(times))
                
                if start_idx is not None:
                    for t_idx in range(start_idx, end_idx):
                        power_usage[t_idx] += power
        
        # Preparar resultados con utilización real
        result = {
            "times": times,
            "transformer_limit": station_limit,
            "power_usage": power_usage.tolist(),
            "avg_usage": np.mean(power_usage) / station_limit if station_limit > 0 else 0,
            "max_usage": max(power_usage) / station_limit if station_limit > 0 else 0,
            "time_at_limit": sum(1 for p in power_usage if p >= 0.95 * station_limit) / len(times)
        }
    
    return result


def plot_transformer_usage(config, rl_solution=None, milp_solution=None, title_prefix=""):
    """
    Visualiza el uso del transformador para ambas soluciones.
    
    Args:
        config: Configuración del sistema
        rl_solution: Solución RL (opcional)
        milp_solution: Solución MILP (opcional)
        title_prefix: Prefijo para los títulos de las gráficas
    """
    # Analizar uso del transformador
    potential_analysis = analyze_transformer_usage(config)
    rl_analysis = analyze_transformer_usage(config, rl_solution) if rl_solution else None
    milp_analysis = analyze_transformer_usage(config, milp_solution) if milp_solution else None
    
    # Crear figura
    plt.figure(figsize=(12, 8))
    
    # Graficar uso del transformador
    times = potential_analysis["times"]
    
    # Graficar potencia potencial (si no hay soluciones)
    if rl_solution is None and milp_solution is None:
        plt.plot(times, potential_analysis["potential_power"], 'b--', 
                 label='Potencia potencial (estimación)', alpha=0.7)
    
    # Graficar soluciones reales
    if rl_analysis:
        plt.plot(times, rl_analysis["power_usage"], 'g-', 
                 label='Uso RL', linewidth=2)
    
    if milp_analysis:
        plt.plot(times, milp_analysis["power_usage"], 'r-', 
                 label='Uso RL+MILP', linewidth=2)
    
    # Graficar límite del transformador
    plt.axhline(y=potential_analysis["transformer_limit"], color='k', linestyle='--',
                label=f'Límite del transformador ({potential_analysis["transformer_limit"]} kW)')
    
    # Configuración del gráfico
    plt.xlabel('Tiempo (horas)')
    plt.ylabel('Potencia (kW)')
    plt.title(f'{title_prefix}Utilización del transformador a lo largo del tiempo')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./plots/transformer_usage.png')
    plt.show()
    
    # Imprimir estadísticas
    print(f"\nEstadísticas de utilización del transformador:")
    print(f"- Límite del transformador: {potential_analysis['transformer_limit']} kW")
    
    if rl_analysis:
        print(f"\nUtilización con solución RL:")
        print(f"- Utilización promedio: {rl_analysis['avg_usage']*100:.2f}%")
        print(f"- Utilización máxima: {rl_analysis['max_usage']*100:.2f}%")
        print(f"- Tiempo cerca del límite (>95%): {rl_analysis['time_at_limit']*100:.2f}% del tiempo")
    
    if milp_analysis:
        print(f"\nUtilización con solución RL+MILP:")
        print(f"- Utilización promedio: {milp_analysis['avg_usage']*100:.2f}%")
        print(f"- Utilización máxima: {milp_analysis['max_usage']*100:.2f}%")
        print(f"- Tiempo cerca del límite (>95%): {milp_analysis['time_at_limit']*100:.2f}% del tiempo")
    
    # Comparación si hay ambas soluciones
    if rl_analysis and milp_analysis:
        efficiency_gain = ((milp_analysis['avg_usage'] - rl_analysis['avg_usage']) / 
                          rl_analysis['avg_usage'] * 100 if rl_analysis['avg_usage'] > 0 else 0)
        
        print(f"\nComparación RL vs RL+MILP:")
        print(f"- Cambio en utilización promedio: {efficiency_gain:.2f}%")
    
    return {
        "potential": potential_analysis,
        "rl": rl_analysis,
        "milp": milp_analysis
    }


def simulate_optimal_capacity(config, capacity_range=None):
    """
    Simula diferentes capacidades de estacionamiento para encontrar el punto óptimo
    basado en rechazos y retorno de inversión.
    
    Args:
        config: Configuración del sistema
        capacity_range: Lista de capacidades a evaluar (opcional)
        
    Returns:
        Diccionario con resultados de la simulación para diferentes capacidades
    """
    if capacity_range is None:
        # Determinar un rango sensato basado en la demanda máxima
        max_evs = 0
        for t in config["times"]:
            present_evs = sum(1 for ev in config["arrivals"] 
                            if ev["arrival_time"] <= t < ev["departure_time"])
            max_evs = max(max_evs, present_evs)
        
        # Rango desde la mitad de la capacidad actual hasta el doble de la máxima demanda
        current_capacity = config["n_spots"]
        min_capacity = max(1, current_capacity // 2)
        max_capacity = min(max_evs * 2, 200)  # Limitar a un máximo razonable
        
        capacity_range = list(range(min_capacity, max_capacity + 1, max(1, (max_capacity - min_capacity) // 20)))
    
    # Para cada capacidad, calcular métricas relevantes
    results = {}
    
    for capacity in capacity_range:
        # Crear configuración temporal con esta capacidad
        temp_config = config.copy()
        temp_config["n_spots"] = capacity
        
        # Analizar rechazos con esta capacidad
        rejected_analysis = analyze_rejected_vehicles(temp_config)
        
        # Estimar ingresos potenciales
        total_energy_required = sum(ev["required_energy"] for ev in config["arrivals"])
        avg_price = np.mean(config["prices"])
        
        # Con esta capacidad, ¿cuánta energía se podría entregar?
        rejection_rate = rejected_analysis["rejection_rate"]
        estimated_energy_delivered = total_energy_required * (1 - rejection_rate)
        estimated_revenue = estimated_energy_delivered * avg_price * 0.5  # 50% de margen
        
        # Estimar costo de capital para esta capacidad
        # Supongamos $5,000 por plaza de estacionamiento
        cost_per_spot = 5000  # USD
        capital_cost = capacity * cost_per_spot
        
        # Cálculo simple de ROI anual (suponiendo que los datos son de un día típico y 365 días)
        daily_profit = estimated_revenue - (0.001 * capital_cost)  # 0.1% de depreciación diaria
        annual_profit = daily_profit * 365
        roi = annual_profit / capital_cost if capital_cost > 0 else 0
        
        # Guardar resultados
        results[capacity] = {
            "capacity": capacity,
            "rejection_rate": rejection_rate,
            "rejected_count": len(rejected_analysis["completely_rejected"]),
            "estimated_energy_delivered": estimated_energy_delivered,
            "estimated_revenue": estimated_revenue,
            "capital_cost": capital_cost,
            "annual_profit": annual_profit,
            "roi": roi
        }
    
    return results


def plot_capacity_simulation(config, title_prefix=""):
    """
    Visualiza la simulación de diferentes capacidades de estacionamiento.
    
    Args:
        config: Configuración del sistema
        title_prefix: Prefijo para los títulos de las gráficas
    """
    # Simular diferentes capacidades
    simulation = simulate_optimal_capacity(config)
    
    # Preparar datos para gráficos
    capacities = sorted(simulation.keys())
    rejection_rates = [simulation[c]["rejection_rate"] * 100 for c in capacities]
    roi_values = [simulation[c]["roi"] * 100 for c in capacities]
    revenue_values = [simulation[c]["estimated_revenue"] for c in capacities]
    
    # Crear gráficas
    plt.figure(figsize=(12, 10))
    
    # 1. Tasa de rechazo vs capacidad
    plt.subplot(2, 1, 1)
    plt.plot(capacities, rejection_rates, 'r-o', linewidth=2)
    
    # Marcar capacidad actual
    current_capacity = config["n_spots"]
    if current_capacity not in simulation:
        closest_capacity = min(capacities, key=lambda x: abs(x - current_capacity))
        print(f"Nota: La capacidad actual ({current_capacity}) no está en el rango simulado.")
        print(f"Usando la capacidad más cercana ({closest_capacity}) para la comparación.")
        current_capacity = closest_capacity
        
    current_rejection = simulation[current_capacity]["rejection_rate"] * 100
    plt.plot(current_capacity, current_rejection, 'bo', markersize=10, 
             label=f'Capacidad actual ({current_capacity} spots)')
    
    # Encontrar el punto donde la reducción marginal es menor al 0.5%
    diminishing_idx = 0
    for i in range(1, len(rejection_rates)):
        if rejection_rates[i-1] - rejection_rates[i] < 0.5 and rejection_rates[i] < 5:
            diminishing_idx = i
            break
    
    if diminishing_idx > 0:
        suggested_capacity = capacities[diminishing_idx]
        suggested_rejection = rejection_rates[diminishing_idx]
        plt.plot(suggested_capacity, suggested_rejection, 'go', markersize=10,
                label=f'Capacidad sugerida ({suggested_capacity} spots)')
    
    plt.xlabel('Capacidad (número de slots)')
    plt.ylabel('Tasa de rechazo (%)')
    plt.title(f'{title_prefix}Tasa de rechazo vs. capacidad del estacionamiento')
    plt.grid(True)
    plt.legend()
    
    # 2. ROI y Revenue vs capacidad
    plt.subplot(2, 1, 2)
    
    # Plotear ROI
    color = 'tab:blue'
    plt.plot(capacities, roi_values, 'o-', color=color, linewidth=2)
    plt.xlabel('Capacidad (número de slots)')
    plt.ylabel('ROI anual estimado (%)', color=color)
    plt.tick_params(axis='y', labelcolor=color)
    
    # Encontrar capacidad óptima para ROI
    optimal_roi_idx = np.argmax(roi_values)
    optimal_capacity = capacities[optimal_roi_idx]
    optimal_roi = roi_values[optimal_roi_idx]
    
    plt.plot(optimal_capacity, optimal_roi, 'bo', markersize=10,
            label=f'Capacidad óptima para ROI ({optimal_capacity} spots)')
    
    # Plotear Revenue en eje secundario
    ax2 = plt.twinx()
    color = 'tab:red'
    ax2.plot(capacities, revenue_values, 'o-', color=color, linewidth=2, alpha=0.7)
    ax2.set_ylabel('Ingreso diario estimado ($)', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f'{title_prefix}ROI y Revenue vs. capacidad del estacionamiento')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./plots/capacity_simulation.png')
    plt.show()
    
    # Imprimir recomendaciones
    print(f"\nRecomendaciones de capacidad:")
    print(f"- Capacidad actual: {current_capacity} slots")
    print(f"  - Tasa de rechazo: {current_rejection:.2f}%")
    print(f"  - ROI anual estimado: {simulation[current_capacity]['roi']*100:.2f}%")
    print(f"  - Ingreso diario estimado: ${simulation[current_capacity]['estimated_revenue']:.2f}")
    
    if diminishing_idx > 0:
        print(f"\n- Capacidad sugerida (rendimientos decrecientes): {suggested_capacity} slots")
        print(f"  - Tasa de rechazo: {suggested_rejection:.2f}%")
        print(f"  - ROI anual estimado: {simulation[suggested_capacity]['roi']*100:.2f}%")
        print(f"  - Ingreso diario estimado: ${simulation[suggested_capacity]['estimated_revenue']:.2f}")
    
    print(f"\n- Capacidad óptima para ROI: {optimal_capacity} slots")
    print(f"  - Tasa de rechazo: {simulation[optimal_capacity]['rejection_rate']*100:.2f}%")
    print(f"  - ROI anual estimado: {optimal_roi:.2f}%")
    print(f"  - Ingreso diario estimado: ${simulation[optimal_capacity]['estimated_revenue']:.2f}")
    
    return simulation

import os
import json
import matplotlib.pyplot as plt

# Actualización de la función run_analysis para incluir los nuevos análisis
def run_advanced_analysis(results_dir="./results", data_dir="./data"):
    """
    Ejecuta análisis avanzado que incluye vehículos rechazados, uso del transformador
    y simulación de capacidad óptima.
    
    Args:
        results_dir: Directorio donde se encuentran los resultados
        data_dir: Directorio donde se encuentran los archivos de configuración
    """
    print("=== ANÁLISIS AVANZADO DE SISTEMAS DE CARGA DE VEHÍCULOS ELÉCTRICOS ===")
    
    # Cargar todas las configuraciones disponibles
    configs = {}
    rl_solutions = {}
    milp_solutions = {}
    
    # Cargar configuraciones
    for filename in os.listdir(data_dir):
        if filename.startswith("test_system_") and filename.endswith(".json"):
            instance_id = filename.replace("test_system_", "").replace(".json", "")
            try:
                with open(os.path.join(data_dir, filename), 'r') as f:
                    config = json.load(f)
                
                # Preparar configuración en formato esperado
                energy_prices = sorted(config["energy_prices"], key=lambda x: x["time"])
                times = [ep["time"] for ep in energy_prices]
                prices = [ep["price"] for ep in energy_prices]
                
                if len(times) > 1:
                    dt = times[1] - times[0]
                else:
                    dt = 0.25
                
                # Transformador
                transformer_limit = config["parking_config"]["transformer_limit"]
                
                processed_config = {
                    "times": times,
                    "prices": prices,
                    "arrivals": config["arrivals"],
                    "n_spots": config["parking_config"]["n_spots"],
                    "transformer_limit": transformer_limit,
                    "dt": dt
                }
                
                configs[instance_id] = processed_config
                
                # Buscar soluciones correspondientes
                rl_file = os.path.join(results_dir, f"resultados_rl_instancia_{instance_id}.json")
                milp_file = os.path.join(results_dir, f"resultados_milp_instancia_{instance_id}.json")
                
                if os.path.exists(rl_file):
                    with open(rl_file, 'r') as f:
                        rl_data = json.load(f)
                        if "schedule" in rl_data:
                            rl_solutions[instance_id] = rl_data["schedule"]
                        else:
                            rl_solutions[instance_id] = rl_data
                
                if os.path.exists(milp_file):
                    with open(milp_file, 'r') as f:
                        milp_data = json.load(f)
                        if "schedule" in milp_data:
                            milp_solutions[instance_id] = milp_data["schedule"]
                        else:
                            milp_solutions[instance_id] = milp_data
                
            except Exception as e:
                print(f"Error al cargar configuración {filename}: {e}")
    
    if not configs:
        print("No se encontraron configuraciones para analizar. Verifique la carpeta de datos.")
        return
    
    print(f"\nSe encontraron {len(configs)} configuraciones disponibles.")
    
    # Menú principal
    while True:
        print("\n=== MENÚ DE ANÁLISIS ===")
        print("1. Comparación de costos (RL vs MILP)")
        print("2. Análisis de congestión del parqueadero")
        print("3. Análisis de vehículos rechazados")
        print("4. Análisis de uso del transformador")
        print("5. Simulación de capacidad óptima")
        print("6. Salir")
        
        option = input("\nSeleccione una opción (1-6): ")
        
        if option == "1":
            # Comparación de costos (ya implementada)
            print("\n1. COMPARACIÓN DE COSTOS RL VS MILP")
            comparison = compare_solution_costs(results_dir, data_dir)
            if comparison:
                plot_solution_comparison(comparison)
            else:
                print("No se encontraron suficientes datos para la comparación.")
        
        elif option == "2":
            # Análisis de congestión (ya implementado)
            print("\n2. ANÁLISIS DE CONGESTIÓN DEL PARQUEADERO")
            select_instance(configs, lambda instance_id, config: 
                            plot_parking_congestion(config, f"Instancia {instance_id}: "))
        
        elif option == "3":
            # Análisis de vehículos rechazados (nuevo)
            print("\n3. ANÁLISIS DE VEHÍCULOS RECHAZADOS")
            select_instance(configs, lambda instance_id, config: 
                            plot_rejected_vehicles_analysis(config, f"Instancia {instance_id}: "))
        
        elif option == "4":
            # Análisis de uso del transformador (nuevo)
            print("\n4. ANÁLISIS DE USO DEL TRANSFORMADOR")
            
            # Esta función requiere tanto la configuración como las soluciones
            def analyze_transformer(instance_id, config):
                rl_sol = rl_solutions.get(instance_id)
                milp_sol = milp_solutions.get(instance_id)
                
                if not rl_sol and not milp_sol:
                    print(f"No se encontraron soluciones para la instancia {instance_id}.")
                    return
                
                plot_transformer_usage(
                    config, rl_sol, milp_sol, f"Instancia {instance_id}: "
                )
            
            select_instance(configs, analyze_transformer)
        
        elif option == "5":
            # Simulación de capacidad óptima (nuevo)
            print("\n5. SIMULACIÓN DE CAPACIDAD ÓPTIMA")
            select_instance(configs, lambda instance_id, config: 
                            plot_capacity_simulation(config, f"Instancia {instance_id}: "))
        
        elif option == "6":
            print("\nSaliendo del programa. ¡Hasta pronto!")
            break
        
        else:
            print("Opción no válida. Intente de nuevo.")


def select_instance(configs, analysis_function):
    """
    Función auxiliar para seleccionar una instancia para analizar.
    
    Args:
        configs: Diccionario de configuraciones
        analysis_function: Función a ejecutar con la instancia seleccionada
    """
    while True:
        print("\nSeleccione una opción:")
        print("1. Analizar todas las instancias")
        print("2. Analizar una instancia específica")
        print("3. Analizar las 3 instancias más congestionadas")
        print("4. Volver al menú principal")
        
        option = input("Opción (1-4): ")
        
        if option == "1":
            # Analizar todas las instancias
            for instance_id, config in sorted(configs.items()):
                print(f"\nAnalizando instancia {instance_id}...")
                analysis_function(instance_id, config)
            break
        
        elif option == "2":
            # Analizar instancia específica
            instance_ids = sorted(configs.keys())
            print(f"\nInstancias disponibles: {', '.join(instance_ids)}")
            
            instance_id = input("Introduzca el ID de la instancia: ")
            if instance_id in configs:
                print(f"\nAnalizando instancia {instance_id}...")
                analysis_function(instance_id, configs[instance_id])
                break
            else:
                print(f"Instancia {instance_id} no encontrada.")
        
        elif option == "3":
            # Analizar las 3 instancias más congestionadas
            max_congestion = {}
            for instance_id, config in configs.items():
                max_evs = 0
                for t in config["times"]:
                    evs_present = sum(1 for ev in config["arrivals"] 
                                     if ev["arrival_time"] <= t < ev["departure_time"])
                    max_evs = max(max_evs, evs_present)
                
                max_congestion[instance_id] = max_evs - config["n_spots"]
            
            # Ordenar por nivel de congestión (mayor a menor)
            most_congested = sorted(max_congestion.items(), key=lambda x: x[1], reverse=True)[:3]
            
            for instance_id, congestion_level in most_congested:
                print(f"\nAnalizando instancia {instance_id} (exceso de demanda: {congestion_level} vehículos)...")
                analysis_function(instance_id, configs[instance_id])
            break
        
        elif option == "4":
            # Volver al menú principal
            return
        
        else:
            print("Opción no válida. Intente de nuevo.")


if __name__ == "__main__":
    # Ejecutar el análisis avanzado (o puedes llamar a run_analysis() para el básico)
    run_advanced_analysis()