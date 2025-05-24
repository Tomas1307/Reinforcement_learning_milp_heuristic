import json
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
from collections import defaultdict

# --- Funciones de visualización para el progreso de entrenamiento de un DQN individual ---

def plot_dqn_learning_progress(episode_data_log: list, save_dir: str, title: str = "DQN Learning Progress"):
    """
    Genera y guarda gráficas de progreso de entrenamiento para un agente DQN individual.
    Muestra la recompensa total, el porcentaje de satisfacción energética y la evolución de epsilon.

    Args:
        episode_data_log (list): Lista de diccionarios, donde cada diccionario contiene
                                 las métricas de un episodio (e.g., el resultado de SimpleEpisodeLogger.get_episode_data()).
        save_dir (str): Directorio donde se guardarán las gráficas.
        title (str, optional): Título principal para las gráficas. Por defecto, "DQN Learning Progress".
    """
    if not episode_data_log:
        print(f"No hay datos de episodios para graficar el progreso de {title}.")
        return

    episodes = [d['episode'] for d in episode_data_log]
    rewards = [d['reward'] for d in episode_data_log]
    satisfactions = [d['satisfaction_pct'] for d in episode_data_log]
    epsilons = [d['epsilon'] for d in episode_data_log]
    assign_ratios = [d['assign_ratio'] for d in episode_data_log]

    fig, axs = plt.subplots(4, 1, figsize=(12, 18), sharex=True)
    fig.suptitle(title, fontsize=18)

    # Gráfica 1: Recompensa Total
    axs[0].plot(episodes, rewards, label='Total Reward', color='blue')
    axs[0].set_ylabel('Reward', fontsize=12)
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].set_title('Total Reward per Episode', fontsize=14)

    # Gráfica 2: Satisfacción Energética
    axs[1].plot(episodes, satisfactions, label='Energy Satisfaction (%)', color='green')
    axs[1].set_ylabel('Satisfaction (%)', fontsize=12)
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].set_title('Energy Satisfaction per Episode', fontsize=14)

    # Gráfica 3: Ratio de Asignación
    axs[2].plot(episodes, assign_ratios, label='Assignment Ratio (%)', color='purple')
    axs[2].set_ylabel('Assignment Ratio (%)', fontsize=12)
    axs[2].legend()
    axs[2].grid(True, linestyle='--', alpha=0.6)
    axs[2].set_title('Vehicle Assignment Ratio per Episode', fontsize=14)

    # Gráfica 4: Epsilon
    axs[3].plot(episodes, epsilons, label='Epsilon (Exploration Rate)', color='red', linestyle='--')
    axs[3].set_xlabel('Episode', fontsize=12)
    axs[3].set_ylabel('Epsilon', fontsize=12)
    axs[3].legend()
    axs[3].grid(True, linestyle='--', alpha=0.6)
    axs[3].set_title('Epsilon Decay over Episodes', fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Ajusta el layout para que el título no se superponga
    
    filename = os.path.join(save_dir, "dqn_learning_progress.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig) # Cierra la figura para liberar memoria
    print(f"Gráfica de progreso de DQN guardada en: {filename}")

# --- Funciones de visualización de soluciones (adaptadas de tu visualize.py original) ---

def visualize_solution(schedule: list, config: dict, save_dir: str = None, filename: str = "schedule_plot.png"):
    """
    Visualiza el horario de carga de vehículos eléctricos.
    Genera un diagrama de Gantt para mostrar las asignaciones de vehículos a cargadores.

    Args:
        schedule (list): Lista de tuplas/listas representando el horario:
                         [(ev_id, t_idx, charger_id, slot_idx, power), ...]
        config (dict): La configuración del sistema que incluye times, chargers, arrivals.
        save_dir (str, optional): Directorio para guardar la gráfica. Si es None, no se guarda.
        filename (str, optional): Nombre del archivo de la gráfica si se guarda.
    """
    if not schedule:
        print("No hay schedule para visualizar.")
        return

    # Extraer información relevante de la configuración
    times = config["times"]
    
    # Manejar diferentes estructuras de cargadores
    if "chargers" in config:
        chargers_info = config["chargers"]
    elif "parking_config" in config and "chargers" in config["parking_config"]:
        chargers_info = config["parking_config"]["chargers"]
    else:
        print("Error: No se encontró información de cargadores en la configuración.")
        return
    
    arrivals_info = config["arrivals"]
    
    # Extraer IDs de cargadores - manejar diferentes nombres de campos
    charger_ids = []
    for c in chargers_info:
        if "charger_id" in c:
            charger_ids.append(c["charger_id"])
        elif "id" in c:
            charger_ids.append(c["id"])
        else:
            print(f"Warning: No se encontró ID de cargador en: {c}")
    
    charger_ids = sorted(charger_ids)
    ev_ids = sorted(list(set([s[0] for s in schedule]))) # Obtener EV IDs únicos del schedule

    # Crear un mapeo de EV ID a color
    cmap = plt.cm.get_cmap('tab20', len(ev_ids))
    ev_colors = {ev_id: cmap(i) for i, ev_id in enumerate(ev_ids)}

    # Preparar datos para el diagrama de Gantt
    charger_schedules = defaultdict(list)
    for ev_id, t_idx, charger_id, slot_idx, power in schedule:
        if t_idx < len(times): # Asegurar que el índice de tiempo es válido
            start_time = times[t_idx]
            duration = config.get("dt", 0.25) # Asume dt si no está en la configuración
            charger_schedules[charger_id].append({
                'ev_id': ev_id,
                'start': start_time,
                'end': start_time + duration,
                'power': power,
                'color': ev_colors[ev_id]
            })

    # Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(15, 8))

    # Graficar cada cargador
    y_pos = np.arange(len(charger_ids))
    for i, charger_id in enumerate(charger_ids):
        for task in charger_schedules[charger_id]:
            ax.barh(y_pos[i], task['end'] - task['start'], left=task['start'],
                    color=task['color'], edgecolor='black', alpha=0.8,
                    label=f"EV {task['ev_id']} (Power: {task['power']:.1f}kW)")
            # Añadir etiqueta de EV_ID y Power en la barra
            ax.text(task['start'] + (task['end'] - task['start']) / 2, y_pos[i], 
                    f"EV {task['ev_id']}\n{task['power']:.1f}kW",
                    ha='center', va='center', color='white', fontsize=8, weight='bold')

    # Configurar los ejes
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Charger {cid}" for cid in charger_ids], fontsize=10)
    ax.set_xlabel("Time (hours)", fontsize=12)
    ax.set_ylabel("Charger", fontsize=12)
    ax.set_title("EV Charging Schedule (Gantt Chart)", fontsize=16)
    ax.set_xlim(min(times), max(times) + config.get("dt", 0.25)) # Ajustar el límite X

    # Leyenda para EVs (si hay muchos, puede ser complicado)
    # Aquí podríamos simplificar y solo mostrar los colores generales si hay muchos EVs
    handles = []
    labels = []
    for ev_id, color in ev_colors.items():
        patch = plt.Line2D([0], [0], marker='s', color='w', label=f'EV {ev_id}',
                           markerfacecolor=color, markersize=10)
        handles.append(patch)
        labels.append(f'EV {ev_id}')
    
    # Colocar la leyenda fuera del gráfico si es necesario
    ax.legend(handles=handles, labels=labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Gráfica de schedule guardada en: {filepath}")
    
    plt.close(fig) # Cierra la figura para liberar memoria
    # plt.show() # Solo descomentar para depuración interactiva


# --- Funciones de visualización de progreso de Scatter Search (adaptadas de tu visualize.py original) ---

def load_and_plot_progress(progress_dir="./learning_progress"):
    """
    Carga y visualiza el progreso de aprendizaje de todos los sistemas (o runs de Scatter Search).
    Esta función asume un formato de archivo específico que podrías adaptar
    para los logs de Scatter Search.
    """
    files = glob(os.path.join(progress_dir, "system_*_progress.json")) # Esto es del visualize.py antiguo
    
    if not files:
        print(f"No se encontraron archivos de progreso en {progress_dir}. (Esto es para logs de sistema, no de DQN individual)")
        return
    
    systems_data = {}
    
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            system_id = data["system_info"]["system_id"]
            systems_data[system_id] = data
    
    print(f"Cargados datos de {len(systems_data)} sistemas")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Progreso de Aprendizaje del DQN Agent', fontsize=16)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(systems_data)))
    
    # Aquí iría tu lógica original para graficar la satisfacción energética, recompensa, etc.
    # para MÚLTIPLES SISTEMAS.
    # Necesitarás adaptar esta función para que cargue los logs del Scatter Search
    # (e.g., population_history.json) y grafique las métricas del conjunto de referencia.

    # Ejemplo de placeholder para tu lógica original, deberás revisarla y adaptarla
    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[1, 0]
    ax4 = axes[1, 1]

    for i, (system_id, data) in enumerate(sorted(systems_data.items())):
        episodes = [ep['episode'] for ep in data['episodes_data']]
        satisfactions = [ep['satisfaction_pct'] for ep in data['episodes_data']]
        rewards = [ep['reward'] for ep in data['episodes_data']]
        assign_ratios = [ep['assign_ratio'] for ep in data['episodes_data']]
        
        ax1.plot(episodes, satisfactions, label=f'Sistema {system_id}', color=colors[i])
        ax2.plot(episodes, rewards, label=f'Sistema {system_id}', color=colors[i])
        ax3.plot(episodes, assign_ratios, label=f'Sistema {system_id}', color=colors[i])

    ax1.set_title('Satisfacción Energética')
    ax1.set_ylabel('Satisfacción (%)')
    ax1.legend()
    ax1.grid(True)

    ax2.set_title('Recompensa Total')
    ax2.set_ylabel('Recompensa')
    ax2.legend()
    ax2.grid(True)

    ax3.set_title('Ratio de Asignación')
    ax3.set_xlabel('Episodio')
    ax3.set_ylabel('Ratio (%)')
    ax3.legend()
    ax3.grid(True)
    
    # ax4 podría ser para tiempo o alguna otra métrica agregada
    # ax4.set_title('Tiempo por Episodio')
    # ax4.set_xlabel('Episodio')
    # ax4.set_ylabel('Tiempo (s)')
    # ax4.grid(True)
    
    # Puedes añadir la impresión de tabla resumen si lo deseas, adaptado a los nuevos datos
    # print("\n" + "="*80)
    # print("RESUMEN DE APRENDIZAJE")
    # print("="*80)
    # ...

    output_file = os.path.join(progress_dir, "learning_progress_all_systems.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Gráfica de progreso de todos los sistemas guardada en: {output_file}")
    # plt.show() # Descomentar para mostrar la gráfica al ejecutar


# --- Funciones de visualización de progreso de Scatter Search (adaptadas de tu visualize.py original) ---

def load_and_plot_progress(progress_dir="./learning_progress"):
    """
    Carga y visualiza el progreso de aprendizaje de todos los sistemas (o runs de Scatter Search).
    Esta función asume un formato de archivo específico que podrías adaptar
    para los logs de Scatter Search.
    """
    files = glob(os.path.join(progress_dir, "system_*_progress.json")) # Esto es del visualize.py antiguo
    
    if not files:
        print(f"No se encontraron archivos de progreso en {progress_dir}. (Esto es para logs de sistema, no de DQN individual)")
        return
    
    systems_data = {}
    
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            system_id = data["system_info"]["system_id"]
            systems_data[system_id] = data
    
    print(f"Cargados datos de {len(systems_data)} sistemas")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Progreso de Aprendizaje del DQN Agent', fontsize=16)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(systems_data)))
    
    # Aquí iría tu lógica original para graficar la satisfacción energética, recompensa, etc.
    # para MÚLTIPLES SISTEMAS.
    # Necesitarás adaptar esta función para que cargue los logs del Scatter Search
    # (e.g., population_history.json) y grafique las métricas del conjunto de referencia.

    # Ejemplo de placeholder para tu lógica original, deberás revisarla y adaptarla
    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[1, 0]
    ax4 = axes[1, 1]

    for i, (system_id, data) in enumerate(sorted(systems_data.items())):
        episodes = [ep['episode'] for ep in data['episodes_data']]
        satisfactions = [ep['satisfaction_pct'] for ep in data['episodes_data']]
        rewards = [ep['reward'] for ep in data['episodes_data']]
        assign_ratios = [ep['assign_ratio'] for ep in data['episodes_data']]
        
        ax1.plot(episodes, satisfactions, label=f'Sistema {system_id}', color=colors[i])
        ax2.plot(episodes, rewards, label=f'Sistema {system_id}', color=colors[i])
        ax3.plot(episodes, assign_ratios, label=f'Sistema {system_id}', color=colors[i])

    ax1.set_title('Satisfacción Energética')
    ax1.set_ylabel('Satisfacción (%)')
    ax1.legend()
    ax1.grid(True)

    ax2.set_title('Recompensa Total')
    ax2.set_ylabel('Recompensa')
    ax2.legend()
    ax2.grid(True)

    ax3.set_title('Ratio de Asignación')
    ax3.set_xlabel('Episodio')
    ax3.set_ylabel('Ratio (%)')
    ax3.legend()
    ax3.grid(True)
    
    # ax4 podría ser para tiempo o alguna otra métrica agregada
    # ax4.set_title('Tiempo por Episodio')
    # ax4.set_xlabel('Episodio')
    # ax4.set_ylabel('Tiempo (s)')
    # ax4.grid(True)
    
    # Puedes añadir la impresión de tabla resumen si lo deseas, adaptado a los nuevos datos
    # print("\n" + "="*80)
    # print("RESUMEN DE APRENDIZAJE")
    # print("="*80)
    # ...

    output_file = os.path.join(progress_dir, "learning_progress_all_systems.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Gráfica de progreso de todos los sistemas guardada en: {output_file}")
    # plt.show() # Descomentar para mostrar la gráfica al ejecutar


# --- Adaptación de la función de visualización de solución RL ---
# Esta función es un placeholder para la que tenías en tu 'main.py'
# Se usará cuando se genere una 'solución final' después del entrenamiento del DQN.
def visualize_rl_solution(rl_schedule: list, system_config: dict, output_dir: str = None, filename: str = "rl_solution_schedule.png"):
    """
    Visualiza una solución de carga generada por un agente RL.
    Es un wrapper para visualize_solution que maneja el formato de RL.
    """
    # En el entorno RL, el 'schedule' es una lista de (ev_id, t_idx, charger_id, slot_idx, power)
    # La función visualize_solution ya espera este formato.
    print("Generando visualización de la solución RL...")
    visualize_solution(rl_schedule, system_config, save_dir=output_dir, filename=filename)
    print(f"Visualización de la solución RL guardada en: {os.path.join(output_dir, filename) if output_dir else 'no guardada'}")


# --- Adaptación de la función de visualización de solución MILP ---
# Esta función es un placeholder para la que tenías en tu 'main.py'
# Se usará cuando se genere una 'solución final' después de la optimización MILP.
def visualize_milp_solution(milp_schedule: dict, system_config: dict, output_dir: str = None, filename: str = "milp_solution_schedule.png"):
    """
    Visualiza una solución de carga generada por el optimizador MILP.
    Adapta el formato de la solución MILP para ser compatible con visualize_solution.
    """
    # milp_schedule es un diccionario como: {ev_id: [(t_idx, charger_id, power), ...]}
    # Necesitamos convertirlo al formato de visualize_solution: [(ev_id, t_idx, charger_id, slot_idx, power), ...]
    converted_schedule = []
    # Aquí asumimos que MILP no usa slot_idx directamente, así que ponemos un placeholder
    # Si tu MILP maneja slots, tendrías que adaptar esta parte.
    for ev_id, assignments in milp_schedule.items():
        for t_idx, charger_id, power in assignments:
            converted_schedule.append((ev_id, t_idx, charger_id, 0, power)) # 0 como slot_idx placeholder

    print("Generando visualización de la solución MILP...")
    visualize_solution(converted_schedule, system_config, save_dir=output_dir, filename=filename)
    print(f"Visualización de la solución MILP guardada en: {os.path.join(output_dir, filename) if output_dir else 'no guardada'}")