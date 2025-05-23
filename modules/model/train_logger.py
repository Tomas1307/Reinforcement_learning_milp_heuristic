import json
import os
import numpy as np
from datetime import datetime

class SystemProgressLogger:
    """
    Logger simple para guardar el progreso de aprendizaje por cada test_system.json
    """
    
    def __init__(self, results_dir="./learning_progress"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        print(f"Progreso de aprendizaje se guardará en: {results_dir}")
    
    def save_system_progress(self, system_id, episodes_data, system_info):
        """
        Guarda el progreso de aprendizaje para un sistema específico.
        
        Args:
            system_id: ID del sistema (ej: 1, 2, 3...)
            episodes_data: Lista con datos de cada episodio
            system_info: Información del sistema (num vehículos, spots, etc.)
        """
        
        # Calcular métricas agregadas
        rewards = [ep['reward'] for ep in episodes_data]
        satisfactions = [ep['satisfaction_pct'] for ep in episodes_data]
        assign_ratios = [ep['assign_ratio'] for ep in episodes_data]
        epsilons = [ep['epsilon'] for ep in episodes_data]
        episode_times = [ep['episode_time'] for ep in episodes_data]
        
        # Calcular tendencias (mejora promedio)
        def calculate_trend(values):
            if len(values) < 2:
                return 0
            x = np.arange(len(values))
            trend = np.polyfit(x, values, 1)[0]  # Pendiente de regresión lineal
            return float(trend)
        
        progress_data = {
            "system_info": {
                "system_id": system_id,
                "total_vehicles": system_info.get("total_vehicles", 0),
                "n_spots": system_info.get("n_spots", 0),
                "n_chargers": system_info.get("n_chargers", 0),
                "total_episodes": len(episodes_data),
                "timestamp": datetime.now().isoformat()
            },
            
            "performance_summary": {
                "final_reward": rewards[-1] if rewards else 0,
                "best_reward": max(rewards) if rewards else 0,
                "avg_reward": np.mean(rewards) if rewards else 0,
                "reward_improvement": rewards[-1] - rewards[0] if len(rewards) > 1 else 0,
                
                "final_satisfaction": satisfactions[-1] if satisfactions else 0,
                "best_satisfaction": max(satisfactions) if satisfactions else 0,
                "avg_satisfaction": np.mean(satisfactions) if satisfactions else 0,
                "satisfaction_improvement": satisfactions[-1] - satisfactions[0] if len(satisfactions) > 1 else 0,
                
                "final_assign_ratio": assign_ratios[-1] if assign_ratios else 0,
                "best_assign_ratio": max(assign_ratios) if assign_ratios else 0,
                "avg_assign_ratio": np.mean(assign_ratios) if assign_ratios else 0,
                
                "final_epsilon": epsilons[-1] if epsilons else 1.0,
                "avg_episode_time": np.mean(episode_times) if episode_times else 0
            },
            
            "learning_trends": {
                "reward_trend": calculate_trend(rewards),
                "satisfaction_trend": calculate_trend(satisfactions),
                "assign_ratio_trend": calculate_trend(assign_ratios),
                "episode_time_trend": calculate_trend(episode_times)
            },
            
            "episode_by_episode": [
                {
                    "episode": i + 1,
                    "reward": rewards[i] if i < len(rewards) else 0,
                    "satisfaction_pct": satisfactions[i] if i < len(satisfactions) else 0,
                    "assign_ratio": assign_ratios[i] if i < len(assign_ratios) else 0,
                    "vehicles_assigned": episodes_data[i].get('vehicles_assigned', 0),
                    "vehicles_skipped": episodes_data[i].get('vehicles_skipped', 0),
                    "epsilon": epsilons[i] if i < len(epsilons) else 1.0,
                    "episode_time": episode_times[i] if i < len(episode_times) else 0
                }
                for i in range(len(episodes_data))
            ]
        }
        
        # Guardar archivo JSON para este sistema
        filename = f"system_{system_id}_progress.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        print(f"Progreso del sistema {system_id} guardado en: {filepath}")
        
        # Mostrar resumen en consola
        print(f"RESUMEN SISTEMA {system_id}:")
        print(f"  Mejora en recompensa: {progress_data['performance_summary']['reward_improvement']:.2f}")
        print(f"  Mejora en satisfacción: {progress_data['performance_summary']['satisfaction_improvement']:.1f}%")
        print(f"  Satisfacción final: {progress_data['performance_summary']['final_satisfaction']:.1f}%")
        print(f"  Ratio asignación final: {progress_data['performance_summary']['final_assign_ratio']:.1f}%")
        
        return filepath


# Función para agregar a la función train_agent en main.py
def create_episode_data(episode_num, total_reward, energy_metrics, assign_count, skip_count, 
                       total_vehicles, episode_time, agent_epsilon):
    """
    Crea el diccionario de datos del episodio para el logger.
    """
    satisfaction_pct = energy_metrics["total_satisfaction_pct"]
    assign_ratio = (assign_count / total_vehicles * 100) if total_vehicles > 0 else 0
    
    return {
        "episode": episode_num,
        "reward": total_reward,
        "satisfaction_pct": satisfaction_pct,
        "vehicles_assigned": assign_count,
        "vehicles_skipped": skip_count,
        "total_vehicles": total_vehicles,
        "assign_ratio": assign_ratio,
        "epsilon": agent_epsilon,
        "episode_time": episode_time
    }


# Código para agregar al final de cada sistema en train_agent
def save_system_results(logger, system_num, episodes_data, system_config):
    """
    Guarda los resultados del entrenamiento para un sistema.
    """
    system_info = {
        "total_vehicles": len(system_config["arrivals"]),
        "n_spots": system_config["n_spots"],
        "n_chargers": len(system_config["parking_config"]["chargers"])
    }
    
    return logger.save_system_progress(system_num, episodes_data, system_info)