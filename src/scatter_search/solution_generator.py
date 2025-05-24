"""
Generación y combinación de soluciones para Scatter Search
"""
import random
import numpy as np
from typing import Dict, Any, List, Tuple

class SolutionGenerator:
    """Maneja la generación, combinación y mejora de soluciones"""
    
    def __init__(self, hyperparameter_space):
        self.hyperparameter_space = hyperparameter_space
        self.setup_combination_methods()
        
    def setup_combination_methods(self):
        """Configura métodos de combinación por tipo de parámetro"""
        self.combination_methods = {
            'float': ['weighted_average', 'geometric_mean', 'random_selection'],
            'int_discrete': ['random_selection', 'majority_vote'],
            'bool': ['random_selection', 'majority_vote']
        }
        
    def combine_solutions(self, parent1: Dict[str, Any], 
                         parent2: Dict[str, Any], 
                         method: str = "weighted_average") -> Dict[str, Any]:
        """Combina dos soluciones padre para crear una hijo"""
        
        child = {}
        alpha = random.uniform(0.2, 0.8)  # Factor de combinación
        
        # Combinar parámetros DQN
        for param, config in self.hyperparameter_space.dqn_params.items():
            child[param] = self._combine_parameter(
                param, parent1[param], parent2[param], config, alpha
            )
            
        # Combinar pesos de recompensa
        for param, config in self.hyperparameter_space.reward_weights.items():
            child[param] = self._combine_parameter(
                param, parent1[param], parent2[param], config, alpha
            )
            
        # Asegurar validez de la solución
        child = self.hyperparameter_space.clip_solution(child)
        
        return child
        
    def _combine_parameter(self, param_name: str, value1: Any, value2: Any, 
                          config: Dict, alpha: float) -> Any:
        """Combina un parámetro específico según su tipo"""
        
        param_type = config['type']
        
        if param_type == 'float':
            # Interpolación lineal con posible extrapolación
            if random.random() < 0.1:  # 10% chance de extrapolación
                alpha = random.uniform(-0.2, 1.2)
            
            combined_value = alpha * value1 + (1 - alpha) * value2
            
            # Clip al rango válido
            combined_value = np.clip(combined_value, config['min'], config['max'])
            return combined_value
            
        elif param_type == 'int_discrete':
            # Selección aleatoria de uno de los padres
            return random.choice([value1, value2])
            
        elif param_type == 'bool':
            # Selección aleatoria de uno de los padres
            return random.choice([value1, value2])
            
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
            
    def combine_multiple_solutions(self, parents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combina múltiples soluciones (3 o más padres)"""
        if len(parents) < 2:
            raise ValueError("Need at least 2 parents for combination")
        elif len(parents) == 2:
            return self.combine_solutions(parents[0], parents[1])
        
        child = {}
        
        # Para cada parámetro, usar método apropiado para múltiples padres
        all_params = {**self.hyperparameter_space.dqn_params, 
                     **self.hyperparameter_space.reward_weights}
        
        for param, config in all_params.items():
            param_values = [parent[param] for parent in parents]
            child[param] = self._combine_multiple_values(param_values, config)
            
        return self.hyperparameter_space.clip_solution(child)
        
    def _combine_multiple_values(self, values: List[Any], config: Dict) -> Any:
        """Combina múltiples valores de un parámetro"""
        param_type = config['type']
        
        if param_type == 'float':
            # Promedio ponderado con pesos aleatorios
            weights = np.random.dirichlet(np.ones(len(values)))
            return sum(w * v for w, v in zip(weights, values))
            
        elif param_type in ['int_discrete', 'bool']:
            # Selección por mayoría o aleatoria
            if len(set(values)) == 1:
                return values[0]  # Todos iguales
            else:
                return random.choice(values)  # Selección aleatoria
                
    def local_improvement(self, solution: Dict[str, Any], 
                         max_iterations: int = 3) -> Dict[str, Any]:
        """Aplica mejora local a una solución"""
        
        improved = solution.copy()
        
        for iteration in range(max_iterations):
            # Seleccionar parámetro aleatorio para perturbar
            all_params = list(solution.keys())
            param_to_improve = random.choice(all_params)
            
            # Aplicar perturbación específica por tipo
            improved_candidate = self._perturb_parameter(improved, param_to_improve)
            
            # Nota: En implementación real, evaluaríamos si la perturbación mejora
            # Por ahora, aplicamos la perturbación directamente
            improved = improved_candidate
            
        return self.hyperparameter_space.clip_solution(improved)
        
    def _perturb_parameter(self, solution: Dict[str, Any], param_name: str) -> Dict[str, Any]:
        """Aplica perturbación a un parámetro específico"""
        
        perturbed = solution.copy()
        
        # Obtener configuración del parámetro
        param_config = (self.hyperparameter_space.dqn_params.get(param_name) or 
                       self.hyperparameter_space.reward_weights.get(param_name))
        
        if not param_config:
            return perturbed
            
        param_type = param_config['type']
        current_value = solution[param_name]
        
        if param_type == 'float':
            # Perturbación gaussiana limitada al 5% del rango
            param_range = param_config['max'] - param_config['min']
            std_dev = 0.05 * param_range
            
            perturbation = np.random.normal(0, std_dev)
            new_value = current_value + perturbation
            
            # Clip al rango válido
            perturbed[param_name] = np.clip(new_value, param_config['min'], param_config['max'])
            
        elif param_type == 'int_discrete':
            # Cambiar a valor adyacente en la lista
            values_list = param_config['values']
            current_index = values_list.index(current_value)
            
            # Seleccionar índice adyacente
            possible_indices = []
            if current_index > 0:
                possible_indices.append(current_index - 1)
            if current_index < len(values_list) - 1:
                possible_indices.append(current_index + 1)
                
            if possible_indices:
                new_index = random.choice(possible_indices)
                perturbed[param_name] = values_list[new_index]
                
        elif param_type == 'bool':
            # Invertir valor booleano
            perturbed[param_name] = not current_value
            
        return perturbed
        
    def generate_diverse_population(self, size: int, 
                                   existing_solutions: List[Dict] = None) -> List[Dict]:
        """Genera población diversa evitando soluciones existentes"""
        
        population = []
        max_attempts = size * 3  # Límite para evitar bucles infinitos
        attempts = 0
        
        while len(population) < size and attempts < max_attempts:
            attempts += 1
            
            # Generar solución candidata
            candidate = self.hyperparameter_space.generate_random_solution()
            
            # Verificar diversidad mínima
            is_diverse = True
            if existing_solutions:
                for existing in existing_solutions + population:
                    # Usar distance manager si está disponible
                    # Por simplicidad, aquí usamos verificación básica
                    if self._solutions_too_similar(candidate, existing):
                        is_diverse = False
                        break
                        
            if is_diverse:
                population.append(candidate)
                
        # Si no logramos generar suficientes soluciones diversas, llenar con aleatorias
        while len(population) < size:
            candidate = self.hyperparameter_space.generate_random_solution()
            population.append(candidate)
            
        return population
        
    def _solutions_too_similar(self, sol1: Dict[str, Any], sol2: Dict[str, Any], 
                              threshold: float = 0.1) -> bool:
        """Verifica si dos soluciones son demasiado similares"""
        # Implementación simple - en producción usar DiversityManager
        differences = 0
        total_params = 0
        
        for param in sol1.keys():
            if param in sol2:
                total_params += 1
                if sol1[param] != sol2[param]:
                    differences += 1
                    
        similarity_ratio = 1.0 - (differences / max(1, total_params))
        return similarity_ratio > (1.0 - threshold)