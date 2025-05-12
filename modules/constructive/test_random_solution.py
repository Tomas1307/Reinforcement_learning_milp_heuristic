import json
import time
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from modules.constructive.heuristic import load_data, plot_charging_schedule, plot_parking_schedule, plot_cost_evolution
from modules.metaheuristic.scatter_search import generar_solucion_aleatoria, evaluar_costo
# Si la función está en otro archivo, impórtala
# from modules.constructive.random_heuristic import generar_solucion_aleatoria

# Si la función está definida directamente en el script, asegúrate de incluirla


def check_capacity_manually(config):
    """
    Verifica manualmente si hay intervalos con más vehículos que plazas.
    """
    print("Verificación manual de capacidad de parqueadero:")
    n_spots = config["n_spots"]
    print(f"Número de plazas disponibles: {n_spots}")
    
    times = config["times"]
    arrivals = config["arrivals"]
    
    # Mapear llegadas y salidas
    arrival_time = {arr["id"]: arr["arrival_time"] for arr in arrivals}
    departure_time = {arr["id"]: arr["departure_time"] for arr in arrivals}
    
    # Contar vehículos presentes en cada intervalo
    max_vehicles = 0
    critical_intervals = []
    
    for t_idx, t in enumerate(times):
        vehicles_present = [
            arr["id"] for arr in arrivals
            if arrival_time[arr["id"]] <= t < departure_time[arr["id"]]
        ]
        count = len(vehicles_present)
        max_vehicles = max(max_vehicles, count)
        
        if count > n_spots:
            critical_intervals.append((t_idx, t, count, count - n_spots))
    
    print(f"Máximo número de vehículos presentes simultáneamente: {max_vehicles}")
    print(f"Capacidad excedida en {len(critical_intervals)} intervalos")
    
    if critical_intervals:
        print("\nDetalles de intervalos críticos (primeros 5):")
        for idx, (t_idx, t, count, excess) in enumerate(critical_intervals[:5]):
            print(f"  Intervalo {t_idx} (t={t:.2f}h): {count} vehículos, {excess} en exceso")
        
        # También mostrar los vehículos en un intervalo crítico
        if critical_intervals:
            t_idx, _, _, _ = critical_intervals[0]
            vehicles_in_interval = [
                arr["id"] for arr in arrivals
                if arrival_time[arr["id"]] <= times[t_idx] < departure_time[arr["id"]]
            ]
            print(f"\nVehículos presentes en intervalo crítico {t_idx} (primeros 10):")
            for v in sorted(vehicles_in_interval)[:10]:
                print(f"  EV {v}: {arrival_time[v]:.2f}h a {departure_time[v]:.2f}h")
    
    return len(critical_intervals) > 0, max_vehicles, len(critical_intervals)

def convert_schedule_to_dict(schedule, times, dt):
    """
    Convierte una lista de tuplas a un diccionario para visualización.
    
    Args:
        schedule: Lista de tuplas (ev_id, t_idx, charger_id, slot, power)
        times: Lista de tiempos
        dt: Intervalo de tiempo
        
    Returns:
        dict: Diccionario {ev_id: [(t_start, t_end, charger_id, slot, power), ...]}
    """
    resultado = defaultdict(list)
    for (ev_id, t_idx, charger_id, slot, power) in schedule:
        # Solo incluir intervalos con carga real
        if charger_id is not None and power > 0:
            t_start = times[t_idx]
            t_end = t_start + dt
            resultado[ev_id].append([t_start, t_end, charger_id, slot, power])
    return dict(resultado)

def test_solucion_aleatoria(json_path):
    """
    Prueba la generación de solución aleatoria y visualiza resultados.
    
    Args:
        json_path: Ruta al archivo JSON con los datos
    """
    print(f"Cargando datos desde {json_path}...")
    config = load_data(json_path)
    
    print(f"Datos cargados: {len(config['arrivals'])} vehículos, {config['n_spots']} plazas, {len(config['chargers'])} cargadores")
    
    # Verificar restricciones de capacidad
    has_capacity_issues, max_vehicles, num_critical = check_capacity_manually(config)
    if has_capacity_issues:
        print(f"\nEste caso tiene problemas de capacidad: {max_vehicles} vehículos máximos con {config['n_spots']} plazas")
        print(f"Posibles vehículos rechazados: {max_vehicles - config['n_spots']} en {num_critical} intervalos")
    else:
        print("\nEste caso NO tiene problemas de capacidad, no se espera ver vehículos rechazados.")
    
    # Medir el tiempo de ejecución
    start_time = time.time()
    
    # Generar solución aleatoria
    random_schedule = generar_solucion_aleatoria(config)
    
    # Tiempo total
    elapsed_time = time.time() - start_time
    
    # Calcular costo
    costo = evaluar_costo(random_schedule, config)
    
    # Convertir a formato adecuado para visualización
    schedule_dict = convert_schedule_to_dict(random_schedule, config["times"], config["dt"])
    
    # Contar vehículos con carga
    evs_con_carga = len(schedule_dict)
    
    # Contar asignaciones
    asignaciones_totales = len(random_schedule)
    asignaciones_con_carga = sum(1 for _, _, c, _, p in random_schedule if c is not None and p > 0)
    
    # Calcular energía entregada
    energia_entregada = defaultdict(float)
    for (ev_id, t_idx, charger_id, slot, power) in random_schedule:
        if charger_id is not None and power > 0:
            energia_entregada[ev_id] += power * config["dt"]
    
    total_energia_entregada = sum(energia_entregada.values())
    total_energia_requerida = sum(arr["required_energy"] for arr in config["arrivals"])
    
    # Imprimir estadísticas
    print("\nEstadísticas de la solución aleatoria:")
    print(f"- Costo total: ${costo:.2f}")
    print(f"- Energía requerida total: {total_energia_requerida:.2f} kWh")
    print(f"- Energía entregada total: {total_energia_entregada:.2f} kWh")
    print(f"- Porcentaje de satisfacción: {(total_energia_entregada/total_energia_requerida*100):.2f}%")
    print(f"- EVs con alguna carga: {evs_con_carga}/{len(config['arrivals'])} ({evs_con_carga/len(config['arrivals'])*100:.2f}%)")
    print(f"- Asignaciones totales: {asignaciones_totales} (con carga: {asignaciones_con_carga})")
    print(f"- Tiempo de ejecución: {elapsed_time:.2f} segundos")
    
    # Generar gráficos
    print("\nGenerando gráficos...")
    
    # Graficar perfil de carga
    fig_charge = plot_charging_schedule(config, schedule_dict)
    plt.savefig("random_charging_profile.png")
    plt.show()
    
    # Graficar asignación de parqueo
    fig_parking = plot_parking_schedule(config, schedule_dict)
    plt.savefig("random_parking_assignment.png")
    plt.show()
    
    return random_schedule, costo

if __name__ == "__main__":
    # Especifica la ruta a tu archivo JSON
    json_path = "./data/test_system_4.json"  # Valor por defecto
    
    random_schedule, costo = test_solucion_aleatoria(json_path)
    print(f"\n¡Prueba completada! Costo final: ${costo:.2f}")