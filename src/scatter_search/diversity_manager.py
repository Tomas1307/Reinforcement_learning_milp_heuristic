"""
Manejo de diversidad y métricas de distancia entre soluciones
"""
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

class DiversityManager:
    """Maneja la diversidad del RefSet y clasificación de arquetipos"""
    
    def __init__(self, hyperparameter_space):
        self.hyperparameter_space = hyperparameter_space
        self.setup_distance_metrics()
        
    def setup_distance_metrics(self):
        """Configura métricas de distancia para diferentes tipos de parámetros"""
        self.continuous_params = []
        self.discrete_params = []
        self.boolean_params = []
        
        # Clasificar parámetros por tipo
        all_params = {**self.hyperparameter_space.dqn_params, 
                     **self.hyperparameter_space.reward_weights}
        
        for param, config in all_params.items():
            if config['type'] == 'float':
                self.continuous_params.append(param)
            elif config['type'] == 'int_discrete':
                self.discrete_params.append(param)
            elif config['type'] == 'bool':
                self.boolean_params.append(param)
                
    def calculate_distance(self, solution1: Dict, solution2: Dict) -> float:
        """Calcula distancia normalizada entre dos soluciones"""
        total_distance = 0.0
        param_count = 0
        
        # Distancia para parámetros continuos
        for param in self.continuous_params:
            if param in solution1 and param in solution2:
                param_config = (self.hyperparameter_space.dqn_params.get(param) or 
                              self.hyperparameter_space.reward_weights.get(param))
                
                # Normalizar por rango del parámetro
                param_range = param_config['max'] - param_config['min']
                if param_range > 0:
                    normalized_dist = abs(solution1[param] - solution2[param]) / param_range
                    total_distance += normalized_dist
                    param_count += 1
                    
        # Distancia para parámetros discretos
        for param in self.discrete_params:
            if param in solution1 and param in solution2:
                # Distancia binaria (0 si iguales, 1 si diferentes)
                if solution1[param] != solution2[param]:
                    total_distance += 1.0
                param_count += 1
                
        # Distancia para parámetros booleanos
        for param in self.boolean_params:
            if param in solution1 and param in solution2:
                if solution1[param] != solution2[param]:
                    total_distance += 1.0
                param_count += 1
                
        # Retornar distancia promedio normalizada
        return total_distance / max(1, param_count)
        
    def select_diverse_subset(self, candidates: List[Dict], 
                             reference_solutions: List[Dict], 
                             count: int) -> List[Dict]:
        """Selecciona un subconjunto diverso de candidatos"""
        if len(candidates) <= count:
            return candidates
            
        selected = []
        remaining = candidates.copy()
        
        for _ in range(count):
            if not remaining:
                break
                
            # Encontrar candidato más diverso
            best_candidate = None
            max_min_distance = -1
            
            for candidate in remaining:
                # Calcular distancia mínima a soluciones ya seleccionadas y de referencia
                min_distance = float('inf')
                
                for ref_sol in reference_solutions + selected:
                    distance = self.calculate_distance(candidate, ref_sol)
                    min_distance = min(min_distance, distance)
                    
                # Seleccionar candidato con mayor distancia mínima
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_candidate = candidate
                    
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
                
        return selected
        
    def select_balanced_refset(self, evaluated_candidates: List[Tuple[Dict, float]], 
                              ref_set_size: int, elite_count: int) -> List[Dict]:
        """Selecciona RefSet balanceando calidad y diversidad"""
        
        # Ordenar por fitness
        evaluated_candidates.sort(key=lambda x: x[1], reverse=True)
        
        selected = []
        
        # Seleccionar elite (mejores por fitness)
        for i in range(min(elite_count, len(evaluated_candidates))):
            selected.append(evaluated_candidates[i][0])
            
        # Seleccionar diversos del resto
        remaining_candidates = [sol for sol, _ in evaluated_candidates[elite_count:]]
        diverse_count = ref_set_size - len(selected)
        
        if diverse_count > 0:
            diverse_solutions = self.select_diverse_subset(
                remaining_candidates, selected, diverse_count
            )
            selected.extend(diverse_solutions)
            
        return selected[:ref_set_size]
        
    def classify_solution_archetype(self, solution: Dict[str, Any]) -> str:
        """Clasifica una solución según su arquetipo de comportamiento"""
        
        # Extraer pesos de recompensa
        energy_satisfaction = solution.get('energy_satisfaction_weight', 1.0)
        energy_cost = solution.get('energy_cost_weight', 0.1)
        penalty_skipped = solution.get('penalty_skipped_vehicle', 50.0)
        reward_assigned = solution.get('reward_assigned_vehicle', 20.0)
        
        # Clasificación basada en pesos dominantes
        if energy_cost > 0.5:
            return "cost_minimizer"
        elif energy_satisfaction > 2.0:
            return "satisfaction_maximizer"
        elif penalty_skipped > 100.0:
            return "urgency_focused"
        elif reward_assigned > 60.0:
            return "efficiency_focused"
        else:
            return "balanced_optimizer"
            
    def get_diversity_metrics(self, solutions: List[Dict]) -> Dict[str, float]:
        """Calcula métricas de diversidad para un conjunto de soluciones"""
        if len(solutions) < 2:
            return {"avg_distance": 0.0, "min_distance": 0.0, "max_distance": 0.0}
            
        distances = []
        for i in range(len(solutions)):
            for j in range(i + 1, len(solutions)):
                distance = self.calculate_distance(solutions[i], solutions[j])
                distances.append(distance)
                
        return {
            "avg_distance": np.mean(distances),
            "min_distance": np.min(distances),
            "max_distance": np.max(distances),
            "std_distance": np.std(distances)
        }