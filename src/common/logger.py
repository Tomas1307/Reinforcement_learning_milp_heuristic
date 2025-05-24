import json
import os
import time
from datetime import datetime
import numpy as np # Para calcular tendencias y estadísticas

class SimpleEpisodeLogger:
    """
    Un logger simple diseñado para registrar los datos de cada episodio
    durante el entrenamiento de un agente DQN o cualquier proceso iterativo.
    Los datos se almacenan en memoria y pueden ser recuperados como una lista de diccionarios.
    """
    def __init__(self, log_frequency: int = 100):
        """
        Inicializa el logger de episodios.

        Args:
            log_frequency (int): La frecuencia (en número de episodios) con la que
                                 se imprimirán los logs a la consola.
        """
        self.episode_data = []
        self.start_time = time.time()
        self.log_frequency = log_frequency

    def log_episode(self, episode_num: int, total_reward: float, metrics: dict,
                    assign_count: int, skip_count: int, total_vehicles: int,
                    episode_time: float, agent_epsilon: float):
        """
        Registra los datos de un episodio y opcionalmente los imprime en la consola.

        Args:
            episode_num (int): Número del episodio actual.
            total_reward (float): Recompensa total obtenida en el episodio.
            metrics (dict): Diccionario de métricas del entorno (de EVChargingEnv.get_metrics()).
            assign_count (int): Número de vehículos asignados en el episodio.
            skip_count (int): Número de vehículos saltados (no asignados) en el episodio.
            total_vehicles (int): Número total de vehículos considerados en el episodio.
            episode_time (float): Duración del episodio en segundos.
            agent_epsilon (float): Valor actual de epsilon del agente (tasa de exploración).
        """
        satisfaction_pct = metrics.get("total_satisfaction_pct", 0.0)
        assign_ratio = (assign_count / total_vehicles * 100) if total_vehicles > 0 else 0

        data = {
            "episode": episode_num,
            "reward": total_reward,
            "satisfaction_pct": satisfaction_pct,
            "vehicles_assigned": assign_count,
            "vehicles_skipped": skip_count,
            "total_vehicles": total_vehicles,
            "assign_ratio": assign_ratio,
            "epsilon": agent_epsilon,
            "episode_time": episode_time,
            "cumulative_time": time.time() - self.start_time
        }
        self.episode_data.append(data)

        if episode_num % self.log_frequency == 0 or episode_num == 1:
            print(f"Ep {episode_num}: Reward={total_reward:.2f}, Sat={satisfaction_pct:.2f}%, "
                  f"Assign={assign_ratio:.1f}%, Epsilon={agent_epsilon:.2f}, Time={episode_time:.2f}s")

    def get_episode_data(self) -> list:
        """
        Retorna la lista completa de datos de los episodios registrados.

        Returns:
            list: Una lista de diccionarios, cada uno con los datos de un episodio.
        """
        return self.episode_data

    def save_episode_data(self, filepath: str):
        """
        Guarda los datos de los episodios en un archivo JSON.

        Args:
            filepath (str): La ruta completa del archivo donde se guardarán los datos.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.episode_data, f, indent=4)
        print(f"Datos de episodios guardados en: {filepath}")

    

# NOTA: La clase SystemProgressLogger y las funciones relacionadas que tenías en
# train_logger.py se adaptarían para el Scatter Search. Por ahora, nos enfocamos en
# la SimpleEpisodeLogger para el entrenamiento del DQN, ya que es la que se usará
# directamente en training.py.

# El antiguo SystemProgressLogger podría ser refactorizado para el Scatter Search
# en un módulo diferente, o una versión más general aquí si se usa para logs
# de alto nivel de toda la ejecución del Scatter Search.

# Por ahora, mantendremos solo SimpleEpisodeLogger aquí para una funcionalidad clara.