"""
Similitud coseno entre configuraciones de hiperparámetros para Scatter Search
"""
import numpy as np
from typing import Dict, List, Any
from sklearn.metrics.pairwise import cosine_similarity

class CosineSimilarityCalculator:
    """Calcula similitud coseno entre configuraciones de hiperparámetros"""
    
    def __init__(self, hyperparameter_space):
        self.hyperparameter_space = hyperparameter_space
        self.setup_parameter_mapping()
        
    def setup_parameter_mapping(self):
        """Configura mapeo de parámetros para vectorización"""
        
        self.param_ranges = {}
        
        # DQN hyperparameters
        for param, config in self.hyperparameter_space.dqn_params.items():
            if config['type'] == 'float':
                self.param_ranges[param] = {
                    'min': config['min'],
                    'max': config['max'],
                    'type': 'float'
                }
            elif config['type'] == 'int_discrete':
                # Mapear valores discretos a índices
                values = config['values']
                self.param_ranges[param] = {
                    'mapping': {val: idx for idx, val in enumerate(values)},
                    'max_idx': len(values) - 1,
                    'type': 'discrete'
                }
            elif config['type'] == 'bool':
                self.param_ranges[param] = {
                    'type': 'bool'
                }
        
        # Reward weights
        for param, config in self.hyperparameter_space.reward_weights.items():
            self.param_ranges[param] = {
                'min': config['min'],
                'max': config['max'],
                'type': 'float'
            }
        
        # Orden consistente de parámetros
        self.param_order = sorted(self.param_ranges.keys())
    
    def config_to_vector(self, config: Dict[str, Any]) -> np.ndarray:
        """Convierte configuración de hiperparámetros a vector normalizado"""
        
        vector = []
        
        for param_name in self.param_order:
            if param_name not in config:
                vector.append(0.5)  # Valor por defecto
                continue
                
            param_info = self.param_ranges[param_name]
            value = config[param_name]
            
            if param_info['type'] == 'float':
                # Normalizar a [0, 1]
                normalized = (value - param_info['min']) / (param_info['max'] - param_info['min'])
                vector.append(np.clip(normalized, 0, 1))
                
            elif param_info['type'] == 'discrete':
                # Convertir a índice normalizado
                idx = param_info['mapping'].get(value, 0)
                normalized = idx / param_info['max_idx'] if param_info['max_idx'] > 0 else 0
                vector.append(normalized)
                
            elif param_info['type'] == 'bool':
                vector.append(1.0 if value else 0.0)
        
        return np.array(vector, dtype=np.float32)
    
    def cosine_similarity(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> float:
        """Calcula similitud coseno entre dos configuraciones"""
        
        vector1 = self.config_to_vector(config1)
        vector2 = self.config_to_vector(config2)
        
        # Calcular similitud coseno usando sklearn
        similarity = cosine_similarity([vector1], [vector2])[0, 0]
        
        return float(similarity)
    
    def cosine_distance(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> float:
        """Calcula distancia coseno (1 - similitud coseno)"""
        return 1.0 - self.cosine_similarity(config1, config2)
    
    def find_most_diverse_configs(self, candidates: List[Dict[str, Any]], 
                                 reference_configs: List[Dict[str, Any]], 
                                 count: int) -> List[Dict[str, Any]]:
        """Encuentra configuraciones más diversas usando similitud coseno"""
        
        if len(candidates) <= count:
            return candidates
        
        selected = []
        remaining = candidates.copy()
        
        for _ in range(count):
            if not remaining:
                break
            
            best_candidate = None
            min_max_similarity = float('inf')  # Queremos MÍNIMA similitud (máxima diversidad)
            
            for candidate in remaining:
                # Calcular similitud máxima con configs de referencia y ya seleccionadas
                all_reference = reference_configs + selected
                
                if not all_reference:
                    # Si no hay referencia, seleccionar el primero
                    best_candidate = candidate
                    break
                
                max_similarity = max(
                    self.cosine_similarity(candidate, ref_config) 
                    for ref_config in all_reference
                )
                
                # Seleccionar candidato con MENOR similitud máxima (más diverso)
                if max_similarity < min_max_similarity:
                    min_max_similarity = max_similarity
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
        
        return selected
    
    def calculate_diversity_metrics(self, configs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calcula métricas de diversidad usando similitud coseno"""
        
        if len(configs) < 2:
            return {
                "avg_similarity": 0.0,
                "min_similarity": 0.0,
                "max_similarity": 0.0,
                "diversity_score": 1.0  # Máxima diversidad si solo hay 1 config
            }
        
        similarities = []
        
        # Calcular similitud entre todos los pares
        for i in range(len(configs)):
            for j in range(i + 1, len(configs)):
                similarity = self.cosine_similarity(configs[i], configs[j])
                similarities.append(similarity)
        
        similarities = np.array(similarities)
        
        return {
            "avg_similarity": float(np.mean(similarities)),
            "min_similarity": float(np.min(similarities)),
            "max_similarity": float(np.max(similarities)),
            "std_similarity": float(np.std(similarities)),
            "diversity_score": 1.0 - float(np.mean(similarities))  # Diversidad = 1 - similitud promedio
        }
    
    def is_too_similar(self, config1: Dict[str, Any], config2: Dict[str, Any], 
                      threshold: float = 0.95) -> bool:
        """Verifica si dos configuraciones son demasiado similares"""
        
        similarity = self.cosine_similarity(config1, config2)
        return similarity > threshold