import json
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob

def load_and_plot_progress(progress_dir="./learning_progress"):
    """
    Carga y visualiza el progreso de aprendizaje de todos los sistemas.
    """
    # Buscar todos los archivos de progreso
    files = glob(os.path.join(progress_dir, "system_*_progress.json"))
    
    if not files:
        print(f"No se encontraron archivos de progreso en {progress_dir}")
        return
    
    systems_data = {}
    
    # Cargar datos
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            system_id = data["system_info"]["system_id"]
            systems_data[system_id] = data
    
    print(f"Cargados datos de {len(systems_data)} sistemas")
    
    # Crear gráficas
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Progreso de Aprendizaje del DQN Agent', fontsize=16)
    
    # Colores para cada sistema
    colors = plt.cm.tab10(np.linspace(0, 1, len(systems_data)))
    
    # Gráfica 1: Satisfacción energética
    ax1 = axes[0, 0]
    for i, (system_id, data) in enumerate(sorted(systems_data.items())):
        episodes = [ep["episode"] for ep in data["episode_by_episode"]]
        satisfactions = [ep["satisfaction_pct"] for ep in data["episode_by_episode"]]
        
        ax1.plot(episodes, satisfactions, 'o-', color=colors[i], 
                label=f'Sistema {system_id} ({data["system_info"]["total_vehicles"]} EVs)',
                linewidth=2, markersize=4)
    
    ax1.set_xlabel('Episodio')
    ax1.set_ylabel('Satisfacción Energética (%)')
    ax1.set_title('Progreso de Satisfacción Energética')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfica 2: Recompensas
    ax2 = axes[0, 1]
    for i, (system_id, data) in enumerate(sorted(systems_data.items())):
        episodes = [ep["episode"] for ep in data["episode_by_episode"]]
        rewards = [ep["reward"] for ep in data["episode_by_episode"]]
        
        ax2.plot(episodes, rewards, 'o-', color=colors[i], 
                label=f'Sistema {system_id}', linewidth=2, markersize=4)
    
    ax2.set_xlabel('Episodio')
    ax2.set_ylabel('Recompensa')
    ax2.set_title('Progreso de Recompensas')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Gráfica 3: Ratio de asignación
    ax3 = axes[1, 0]
    for i, (system_id, data) in enumerate(sorted(systems_data.items())):
        episodes = [ep["episode"] for ep in data["episode_by_episode"]]
        assign_ratios = [ep["assign_ratio"] for ep in data["episode_by_episode"]]
        
        ax3.plot(episodes, assign_ratios, 'o-', color=colors[i], 
                label=f'Sistema {system_id}', linewidth=2, markersize=4)
    
    ax3.set_xlabel('Episodio')
    ax3.set_ylabel('Ratio de Asignación (%)')
    ax3.set_title('Vehículos Asignados vs Saltados')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Gráfica 4: Comparación final
    ax4 = axes[1, 1]
    system_names = []
    final_satisfactions = []
    improvements = []
    
    for system_id, data in sorted(systems_data.items()):
        system_names.append(f"Sistema {system_id}")
        final_satisfactions.append(data["performance_summary"]["final_satisfaction"])
        improvements.append(data["performance_summary"]["satisfaction_improvement"])
    
    bars = ax4.bar(system_names, final_satisfactions, color=colors[:len(system_names)], alpha=0.7)
    ax4.set_ylabel('Satisfacción Final (%)')
    ax4.set_title('Satisfacción Final por Sistema')
    ax4.tick_params(axis='x', rotation=45)
    
    # Añadir valores de mejora encima de las barras
    for i, (bar, improvement) in enumerate(zip(bars, improvements)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'+{improvement:.1f}%' if improvement > 0 else f'{improvement:.1f}%',
                ha='center', va='bottom', fontsize=10)
    
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar gráfica
    output_file = os.path.join(progress_dir, "learning_progress.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Gráfica guardada en: {output_file}")
    
    plt.show()
    
    # Imprimir tabla resumen
    print("\n" + "="*80)
    print("RESUMEN DE APRENDIZAJE")
    print("="*80)
    print(f"{'Sistema':<10} {'Vehículos':<10} {'Satisf. Final':<15} {'Mejora':<10} {'Tiempo/Ep':<12}")
    print("-"*80)
    
    for system_id, data in sorted(systems_data.items()):
        vehicles = data["system_info"]["total_vehicles"]
        final_sat = data["performance_summary"]["final_satisfaction"]
        improvement = data["performance_summary"]["satisfaction_improvement"]
        avg_time = data["performance_summary"]["avg_episode_time"]
        
        print(f"{system_id:<10} {vehicles:<10} {final_sat:<15.1f} {improvement:<10.1f} {avg_time:<12.1f}")
    
    print("-"*80)
    
    # Estadísticas globales
    all_improvements = [data["performance_summary"]["satisfaction_improvement"] for data in systems_data.values()]
    all_final_sats = [data["performance_summary"]["final_satisfaction"] for data in systems_data.values()]
    
    print(f"Mejora promedio: {np.mean(all_improvements):.1f}%")
    print(f"Satisfacción final promedio: {np.mean(all_final_sats):.1f}%")
    print(f"Sistemas que mejoraron: {sum(1 for imp in all_improvements if imp > 0)}/{len(all_improvements)}")

if __name__ == "__main__":
    # Ejecutar análisis
    load_and_plot_progress()