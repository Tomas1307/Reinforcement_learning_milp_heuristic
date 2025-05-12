import json
import time
import matplotlib.pyplot as plt
from .heuristic import load_data, plot_charging_schedule, plot_parking_schedule
from .time_slot_heuristic import HeuristicaPorVentanaTiempo


def plot_gantt_chart(config, schedule):
    """
    Genera un gráfico de Gantt para visualizar la programación de carga de los EVs.
    
    Args:
        config: Configuración del sistema
        schedule: Lista de tuplas (ev_id, t_idx, charger_id, slot, power)
    
    Returns:
        Figure: El objeto figura de matplotlib
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    from collections import defaultdict
    
    # Crear mapeos de tiempos de llegada y salida
    # Verificamos la estructura para obtener los datos correctamente
    if "arrivals" in config:
        # Si config contiene la estructura original
        arrival_time = {arr["id"]: arr["arrival_time"] for arr in config["arrivals"]}
        departure_time = {arr["id"]: arr["departure_time"] for arr in config["arrivals"]}
    else:
        # Si config ya está procesado por load_data
        arrival_time = config["arrival_time"] if "arrival_time" in config else {}
        departure_time = config["departure_time"] if "departure_time" in config else {}
    
    # Agrupar asignaciones por vehículo y ordenar por tiempo
    ev_assignments = defaultdict(list)
    for (ev_id, t_idx, charger_id, slot, power) in schedule:
        # Solo considerar asignaciones con carga real
        if charger_id is not None and power > 0:
            t_start = config["times"][t_idx]
            t_end = t_start + config["dt"]
            ev_assignments[ev_id].append((t_idx, t_start, t_end, charger_id, power))
    
    # Ordenar EVs por ID
    ev_ids = sorted(ev_assignments.keys())
    
    # Crear un mapa de colores para los cargadores
    charger_ids = set()
    for assignments in ev_assignments.values():
        for _, _, _, charger_id, _ in assignments:
            charger_ids.add(charger_id)
    
    import matplotlib.cm as cm
    # Usar plt.cm.tab10 directamente para evitar el warning de deprecación
    cmap = plt.cm.tab10
    charger_colors = {c: cmap(i % 10) for i, c in enumerate(sorted(charger_ids))}
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Ajustar límites de tiempo
    t_min = min(config["times"])
    t_max = max(config["times"]) + config["dt"]
    
    # Dibujar barras para cada asignación de EV
    y_ticks = []
    y_labels = []
    
    for i, ev_id in enumerate(ev_ids):
        y_pos = i * 2  # Espaciado vertical
        y_ticks.append(y_pos)
        y_labels.append(f"EV {ev_id}")
        
        # Dibujar las ventanas de tiempo de cada EV (si tenemos los datos)
        if ev_id in arrival_time and ev_id in departure_time:
            arrival = arrival_time[ev_id]
            departure = departure_time[ev_id]
            ax.add_patch(
                patches.Rectangle(
                    (arrival, y_pos - 0.4),
                    departure - arrival,
                    0.8,
                    facecolor='lightgray',
                    alpha=0.3,
                    edgecolor='gray',
                    linewidth=1,
                    zorder=1
                )
            )
        
        # Dibujar las sesiones de carga
        assignments = sorted(ev_assignments[ev_id], key=lambda x: x[1])  # Ordenar por tiempo de inicio
        
        for _, t_start, t_end, charger_id, power in assignments:
            # Altura proporcional a la potencia
            # Obtenemos el máximo poder de los cargadores disponibles
            if "max_charger_power" in config:
                max_power = max(config["max_charger_power"].values())
            elif "chargers" in config:
                max_power = max(c["power"] for c in config["chargers"])
            else:
                # Valor por defecto si no podemos determinar el máximo
                max_power = 10
            
            height = 0.8 * (power / max_power)
            
            # Dibujar barra de carga
            ax.add_patch(
                patches.Rectangle(
                    (t_start, y_pos - height/2),
                    t_end - t_start,
                    height,
                    facecolor=charger_colors[charger_id],
                    edgecolor='black',
                    linewidth=1,
                    alpha=0.8,
                    zorder=2
                )
            )
            
            # Añadir texto con potencia dentro de la barra si es suficientemente ancha
            if (t_end - t_start) > 0.5:
                ax.text(
                    (t_start + t_end) / 2,
                    y_pos,
                    f"{power:.1f}kW",
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=8,
                    zorder=3
                )
    
    # Añadir línea para los precios de energía
    ax2 = ax.twinx()
    
    # Obtener los precios según la estructura
    if "energy_prices" in config:
        price_times = [ep["time"] for ep in config["energy_prices"]]
        prices = [ep["price"] for ep in config["energy_prices"]]
    else:
        price_times = config["times"]
        prices = config["prices"]
    
    ax2.step(price_times, prices, 'r-', where='post', linewidth=2, alpha=0.7, label='Precio Energía')
    
    # Configuración de ejes y leyenda
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(-1, len(ev_ids) * 2)
    
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    
    ax.set_xlabel('Tiempo (horas)')
    ax.set_ylabel('Vehículos')
    ax2.set_ylabel('Precio ($/kWh)')
    
    ax.set_title('Diagrama de Gantt: Programación de Carga de EVs')
    
    # Crear leyenda para cargadores
    handles = [patches.Patch(color=charger_colors[c], label=f'Charger {c}') for c in sorted(charger_ids)]
    handles.append(plt.Line2D([0], [0], color='r', linewidth=2, label='Precio Energía'))
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=min(5, len(handles)), frameon=True)
    
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return fig

def test_heuristica_ventana(json_path):
    """
    Prueba la heurística basada en ventanas de tiempo y muestra los resultados.
    
    Args:
        json_path: Ruta al archivo JSON con los datos
    """
    print(f"Cargando datos desde {json_path}...")
    config = load_data(json_path)
    
    # Inicializar la heurística por ventana de tiempo
    heuristica = HeuristicaPorVentanaTiempo(config)
    
    # Medir el tiempo de ejecución
    start_time = time.time()
    
    # Ejecutar la heurística
    result = heuristica.run(track_progress=True)
    
    # Extraer schedule y costos
    if isinstance(result, tuple) and len(result) == 2:
        schedule, costos_por_iteracion = result
    else:
        schedule = result
        costos_por_iteracion = None
    
    # Tiempo total
    elapsed_time = time.time() - start_time
    
    # Obtener resultados
    resultado = heuristica.get_resultados()
    stats = resultado["estadisticas"]
    
    # Imprimir estadísticas
    print("\nEstadísticas de la solución (Heurística por Ventana de Tiempo):")
    print(f"- Costo total: ${stats['costo_total']:.2f}")
    print(f"- Energía requerida total: {stats['energia_requerida_total']:.2f} kWh")
    print(f"- Energía entregada total: {stats['energia_entregada_total']:.2f} kWh")
    print(f"- Porcentaje de satisfacción: {stats['porcentaje_satisfaccion']:.2f}%")
    print(f"- EVs satisfechos completamente: {stats['evs_satisfechos_completamente']}/{stats['evs_totales']} ({stats['porcentaje_evs_satisfechos']:.2f}%)")
    print(f"- Porcentaje de EVs con alguna carga: {stats['porcentaje_evs_con_alguna_carga']:.2f}%")
    print(f"- Tiempo de ejecución: {elapsed_time:.2f} segundos")
    print(f"- EVs rechazados por capacidad: {stats.get('evs_rechazados_por_capacidad', 0)}")

    # Generar gráfico de Gantt
    print("\nGenerando gráfico de Gantt...")
    fig_gantt = plot_gantt_chart(config, schedule)
    plt.savefig("heuristica_ventana_gantt.png", dpi=300, bbox_inches='tight')
    print("Gráfico de Gantt guardado como 'heuristica_ventana_gantt.png'")
    plt.show()
    
    # Imprimir estructura del schedule
    print(f"\nTipo de datos del schedule: {type(schedule)}")
    print(f"Longitud del schedule: {len(schedule)}")
    if len(schedule) > 0:
        print(f"Ejemplo de entrada: {schedule[0]}")

if __name__ == "__main__":
    # Especifica la ruta a tu archivo JSON
    json_path = "./data/test_system_4.json"  # Cambia esto
    test_heuristica_ventana(json_path)