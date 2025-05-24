"""
Motor de evaluación multi-nivel para soluciones de hiperparámetros
"""
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Any, Optional
import yaml
# Imports from your existing modules
from ..dqn_agent.agent import EnhancedDQNAgentPyTorch
from ..dqn_agent.environment import EVChargingEnv
from ..dqn_agent.training import train_dqn_agent
from ..common.config import load_system_config

class EvaluationEngine:
    """Maneja la evaluación de soluciones en diferentes niveles"""
    
    def __init__(self, config_path: str, systems_data: Dict):
        self.load_evaluation_config(config_path)
        self.systems_data = systems_data
        self.setup_evaluation_systems()
        
    def load_evaluation_config(self, config_path: str):
        """Carga configuración de evaluación"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.eval_config = config['evaluation']
        self.comp_config = config['computation']
        
    def setup_evaluation_systems(self):
        """Prepara sistemas para cada nivel de evaluación"""
        self.eval_systems = {
            'fast': [self.systems_data[i] for i in self.eval_config['fast']['systems']],
            'medium': [self.systems_data[i] for i in self.eval_config['medium']['systems']],
            'full': list(self.systems_data.values())
        }
        
    def evaluate_solution(self, solution: Dict[str, Any], 
                         level: str = "fast", 
                         parallel: bool = True) -> float:
        """Evalúa una solución en el nivel especificado"""
        
        eval_params = self.eval_config[level]
        systems_to_use = self.eval_systems[level]
        episodes = eval_params['episodes']
        
        if parallel and len(systems_to_use) > 1:
            return self._evaluate_parallel(solution, systems_to_use, episodes)
        else:
            return self._evaluate_sequential(solution, systems_to_use, episodes)
            
    def _evaluate_sequential(self, solution: Dict[str, Any], 
                           systems: List[Dict], episodes: int) -> float:
        """Evaluación secuencial en múltiples sistemas"""
        fitness_scores = []
        
        for system_config in systems:
            fitness = self._evaluate_single_system(solution, system_config, episodes)
            fitness_scores.append(fitness)
            
        return np.mean(fitness_scores)
        
    def _evaluate_parallel(self, solution: Dict[str, Any],
                          systems: List[Dict], episodes: int) -> float:
        """Evaluación paralela en múltiples sistemas"""
        max_workers = min(self.comp_config['max_workers'], len(systems))
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._evaluate_single_system, solution, system, episodes)
                for system in systems
            ]
            
            fitness_scores = [future.result() for future in futures]
            
        return np.mean(fitness_scores)
        
    def _evaluate_single_system(self, solution: Dict[str, Any], 
                               system_config: Dict, episodes: int) -> float:
        """Evalúa una solución en un sistema específico"""
        try:
            # Separar hiperparámetros DQN y pesos de recompensa
            dqn_params = self._extract_dqn_params(solution)
            reward_weights = self._extract_reward_weights(solution)
            
            # Crear environment y actualizar pesos
            env = EVChargingEnv(system_config)
            env.update_reward_weights(reward_weights)
            
            # Crear agente con hiperparámetros
            agent = EnhancedDQNAgentPyTorch(**dqn_params)
            
            # Entrenar y obtener resultados
            results = train_dqn_agent(agent, env, episodes)
            
            # Calcular fitness (promedio últimos 10 episodios)
            recent_rewards = [ep['reward'] for ep in results[-10:]]
            fitness = np.mean(recent_rewards)
            
            return fitness
            
        except Exception as e:
            print(f"Error evaluating solution: {e}")
            return -float('inf')  # Penalización por error
            
    def _extract_dqn_params(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae hiperparámetros DQN de una solución"""
        dqn_param_names = [
            'learning_rate', 'gamma', 'epsilon_start', 'epsilon_min',
            'epsilon_decay', 'batch_size', 'target_update_freq', 
            'memory_size', 'dueling_network'
        ]
        
        return {param: solution[param] for param in dqn_param_names if param in solution}
        
    def _extract_reward_weights(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae pesos de recompensa de una solución"""
        weight_param_names = [
            'energy_satisfaction_weight', 'energy_cost_weight',
            'penalty_skipped_vehicle', 'reward_assigned_vehicle'
        ]
        
        return {param: solution[param] for param in weight_param_names if param in solution}
        
    def batch_evaluate(self, solutions: List[Dict[str, Any]], 
                      level: str = "fast") -> List[float]:
        """Evalúa un batch de soluciones eficientemente"""
        if self.comp_config.get('batch_parallel', True):
            max_workers = self.comp_config['max_workers']
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self.evaluate_solution, sol, level, False)
                    for sol in solutions
                ]
                return [future.result() for future in futures]
        else:
            return [self.evaluate_solution(sol, level, False) for sol in solutions]