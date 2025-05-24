"""
Definición y manejo del espacio de búsqueda de hiperparámetros
"""
import yaml
import random
import numpy as np
from typing import Dict, Any, List, Tuple

class HyperparameterSpace:
    """Maneja la definición y sampling del espacio de hiperparámetros"""
    
    def __init__(self, config_path: str):
        self.load_space_definition(config_path)
        self.setup_parameter_types()
        
    def load_space_definition(self, config_path: str):
        """Carga definición desde hyperparameter_ranges.yaml"""
        with open(config_path, 'r') as f:
            self.space_config = yaml.safe_load(f)
            
        self.dqn_params = self.space_config['dqn_hyperparameters']
        self.reward_weights = self.space_config['reward_weights'] 
        self.archetypes = self.space_config['archetypes']
        
    def generate_random_solution(self) -> Dict[str, Any]:
        """Genera una solución aleatoria válida"""
        solution = {}
        
        # DQN parameters
        for param, config in self.dqn_params.items():
            solution[param] = self._sample_parameter(param, config)
            
        # Reward weights  
        for param, config in self.reward_weights.items():
            solution[param] = self._sample_parameter(param, config)
            
        return solution
        
    def generate_archetype_solution(self, archetype_name: str) -> Dict[str, Any]:
        """Genera solución sesgada hacia un arquetipo específico"""
        solution = self.generate_random_solution()
        
        # Apply archetype constraints
        if archetype_name in self.archetypes:
            constraints = self.archetypes[archetype_name]
            for param, range_constraint in constraints.items():
                if param != 'description':
                    min_val, max_val = range_constraint
                    solution[param] = random.uniform(min_val, max_val)
                    
        return solution
        
    def _sample_parameter(self, param_name: str, config: Dict) -> Any:
        """Samplea un parámetro individual según su tipo"""
        param_type = config['type']
        
        if param_type == 'float':
            return random.uniform(config['min'], config['max'])
        elif param_type == 'int_discrete':
            return random.choice(config['values'])
        elif param_type == 'bool':
            return random.choice(config['values'])
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
            
    def validate_solution(self, solution: Dict[str, Any]) -> bool:
        """Valida que una solución esté dentro de rangos válidos"""
        # Implementation for validation
        return True
        
    def clip_solution(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        """Clipea una solución a rangos válidos"""
        clipped = solution.copy()
        
        for param, config in {**self.dqn_params, **self.reward_weights}.items():
            if config['type'] == 'float':
                clipped[param] = np.clip(solution[param], config['min'], config['max'])
            elif config['type'] in ['int_discrete', 'bool']:
                if solution[param] not in config['values']:
                    clipped[param] = random.choice(config['values'])
                    
        return clipped