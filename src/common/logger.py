import json
import os
import time
from datetime import datetime
import numpy as np

class SimpleEpisodeLogger:
    """
    A simple logger designed to record data for each episode
    during the training of a DQN agent or any iterative process.
    Data is stored in memory and can be retrieved as a list of dictionaries.
    """
    def __init__(self, log_frequency: int = 100):
        """
        Initializes the episode logger.

        Args:
            log_frequency (int): The frequency (in number of episodes) at which
                                 logs will be printed to the console.
        """
        self.episode_data = []
        self.start_time = time.time()
        self.log_frequency = log_frequency

    def log_episode(self, episode_num: int, total_reward: float, metrics: dict,
                    assign_count: int, skip_count: int, total_vehicles: int,
                    episode_time: float, agent_epsilon: float):
        """
        Records episode data and optionally prints it to the console.

        Args:
            episode_num (int): The current episode number.
            total_reward (float): The total reward obtained in the episode.
            metrics (dict): Dictionary of environment metrics (from EVChargingEnv.get_metrics()).
            assign_count (int): The number of vehicles assigned in the episode.
            skip_count (int): The number of vehicles skipped (not assigned) in the episode.
            total_vehicles (int): The total number of vehicles considered in the episode.
            episode_time (float): The duration of the episode in seconds.
            agent_epsilon (float): The agent's current epsilon value (exploration rate).
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
            print(f"Episode {episode_num}: Reward={total_reward:.2f}, Satisfaction={satisfaction_pct:.2f}%, "
                  f"Assignment={assign_ratio:.1f}%, Epsilon={agent_epsilon:.2f}, Time={episode_time:.2f}s")

    def get_episode_data(self) -> list:
        """
        Returns the complete list of logged episode data.

        Returns:
            list: A list of dictionaries, each containing the data for an episode.
        """
        return self.episode_data

    def save_episode_data(self, filepath: str):
        """
        Saves the episode data to a JSON file.

        Args:
            filepath (str): The full path to the file where the data will be saved.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.episode_data, f, indent=4)
        print(f"Episode data saved to: {filepath}")