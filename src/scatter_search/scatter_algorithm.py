"""
Implementación del algoritmo Scatter Search principal
"""
import json
import random
import numpy as np
from typing import Dict, List, Tuple, Any
import time
from datetime import datetime
import yaml
from .hyperparameter_space import HyperparameterSpace
from .cosine_similarity_hyperparams import CosineSimilarityCalculator
from ..dqn_agent.training import train_dqn_agent
from ..dqn_agent.agent import EnhancedDQNAgentPyTorch
from ..dqn_agent.environment import EVChargingEnv
import os
class ScatterSearchOptimizer:
    """Implementación principal del algoritmo Scatter Search"""
    
    def __init__(self, config_path: str, systems_data: Dict, output_dir: str = None):
        self.load_config(config_path)
        self.systems_data = systems_data
        
        self.hyperparameter_space = HyperparameterSpace(
            "./src/configs/hyperparameter_ranges.yaml"
        )
        self.similarity_calculator = CosineSimilarityCalculator(self.hyperparameter_space)
        self.main_output_dir = output_dir or "./results"
        self.population = []
        self.reference_set = []
        self.best_solutions = []
        self.iteration_history = []
        self.start_time = None
        self.current_iteration = 0
        
    def load_config(self, config_path: str):
        """Carga configuración del algoritmo"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.algo_config = self.config['algorithm']
        self.eval_config = self.config['evaluation']
        self.output_config = self.config['output']
        
    def load_from_checkpoint(self, checkpoint_path: str) -> bool:
        """Carga estado desde checkpoint y prepara para continuar"""
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            self.current_iteration = checkpoint_data['iteration']
            self.reference_set = checkpoint_data['reference_set']
            self.best_solutions = checkpoint_data['best_solutions']
            self.iteration_history = checkpoint_data['iteration_history']
            
            print(f" Checkpoint cargado: iteración {self.current_iteration}")
            return True
        except Exception as e:
            print(f" Error cargando checkpoint: {e}")
            return False
        
    def run_optimization(self, resume_from: str = None) -> Dict[str, Any]:
        """Ejecuta el algoritmo completo de Scatter Search"""
        print("Iniciando Scatter Search Optimization")
        print(f"Tiempo máximo: {self.algo_config['max_time_hours']} horas")
        print(f"Población: {self.algo_config['population_size']}")
        print(f"RefSet: {self.algo_config['ref_set_size']}")
        
        
        if resume_from:
            print(f"Intentando resumir desde: {resume_from}")
            if self.load_from_checkpoint(resume_from):
                return self._resume_optimization_loop()
            else:
                print("No se pudo cargar checkpoint, iniciando desde cero")
        
        print("Iniciando nueva optimización")
        
        
        self.start_time = time.time()
        
        try:
            print("\nFase 1: Generando población inicial...")
            self.population = self._generate_initial_population()
            
            print("\nFase 2: Construyendo Reference Set inicial...")
            self.reference_set = self._build_initial_reference_set()
            
            print("\nFase 3: Iteraciones principales...")
            self._main_optimization_loop()
            
            print("\nFase 4: Evaluación final...")
            final_results = self._final_evaluation()
            
            return final_results
            
        except KeyboardInterrupt:
            print("\nOptimización interrumpida por usuario")
            return self._emergency_results()
        except Exception as e:
            print(f"\nError durante optimización: {e}")
            return self._emergency_results()
            
    def _generate_initial_population(self) -> List[Dict[str, Any]]:
        """Genera población inicial con diversidad garantizada"""
        population = []
        pop_size = self.algo_config['population_size']
        
        archetype_names = list(self.hyperparameter_space.archetypes.keys())
        individuals_per_archetype = max(2, pop_size // (len(archetype_names) * 3))
        
        for archetype in archetype_names:
            for _ in range(individuals_per_archetype):
                solution = self.hyperparameter_space.generate_archetype_solution(archetype)
                population.append(solution)
                
        while len(population) < pop_size:
            solution = self.hyperparameter_space.generate_random_solution()
            population.append(solution)
            
        print(f"Población generada: {len(population)} individuos")
        print(f"   - Por arquetipo: {individuals_per_archetype}")
        print(f"   - Aleatorios: {pop_size - len(archetype_names) * individuals_per_archetype}")
        
        return population
        
    def _build_initial_reference_set(self) -> List[Dict[str, Any]]:
        """Construye el Reference Set inicial"""
        print("Evaluando población inicial...")
        
        fitness_scores = self._evaluate_population(self.population, level="fast")
        
        evaluated_population = list(zip(self.population, fitness_scores))
        evaluated_population.sort(key=lambda x: x[1], reverse=True)
        
        ref_set_size = self.algo_config['ref_set_size']
        elite_count = self.algo_config['elite_count']
        diverse_count = self.algo_config['diverse_count']
        
        reference_set = []
        
        for i in range(elite_count):
            reference_set.append(evaluated_population[i][0])
            
        remaining_solutions = [sol for sol, _ in evaluated_population[elite_count:]]
        diverse_solutions = self.similarity_calculator.find_most_diverse_configs(
            remaining_solutions, reference_set, diverse_count
        )
        reference_set.extend(diverse_solutions)
        
        print(f"RefSet construido: {len(reference_set)} soluciones")
        print(f"   - Elite: {elite_count}")
        print(f"   - Diverse: {len(diverse_solutions)}")
        
        self.best_solutions = [sol for sol, _ in evaluated_population[:5]]
        
        return reference_set
    
    def _resume_optimization_loop(self):
        """Continúa el loop principal desde donde se quedó"""
        print(f"🚀 Resumiendo desde iteración {self.current_iteration + 1}")
        
        max_iterations = self.algo_config['max_iterations']
        max_time = self.algo_config['max_time_hours'] * 3600
        
        if self.iteration_history:
            elapsed_time_from_history = self.iteration_history[-1]['time']
            self.start_time = time.time() - elapsed_time_from_history
            print(f"Tiempo ya transcurrido: {elapsed_time_from_history/3600:.2f}h")
        else:
            self.start_time = time.time()
            print("No hay historial de tiempo, reiniciando contador")
        
        try:
            start_iteration = self.current_iteration + 1
            print(f"RefSet actual: {len(self.reference_set)} soluciones")
            print(f"Mejores soluciones: {len(self.best_solutions)}")
            
            for iteration in range(start_iteration, max_iterations):
                self.current_iteration = iteration
                iteration_start = time.time()
                
                elapsed_time = time.time() - self.start_time
                if elapsed_time > max_time:
                    print(f"Límite de tiempo alcanzado ({elapsed_time/3600:.1f}h)")
                    break
                    
                print(f"\nIteración {iteration + 1}/{max_iterations} (RESUMIDA)")
                
                print(" Fase de combinación...")
                new_solutions = self._combination_phase()
                
                print(" Fase de mejora...")
                improved_solutions = self._improvement_phase(new_solutions)
                
                print(" Actualizando RefSet...")
                self.reference_set = self._update_reference_set(improved_solutions)
                
                iteration_time = time.time() - iteration_start
                self._log_iteration_progress(iteration, iteration_time)
                
                checkpoint_frequency = self.output_config.get('save_frequency', 1)  
                if iteration % checkpoint_frequency == 0:
                    self._save_checkpoint(iteration)
                    print(f"Checkpoint automático guardado")
            
            print("\nFase 4: Evaluación final...")
            final_results = self._final_evaluation()
            
            final_results['resumed_from_iteration'] = start_iteration - 1
            final_results['total_iterations_including_resume'] = len(self.iteration_history)
            
            return final_results
            
        except KeyboardInterrupt:
            print("\nOptimización resumida interrumpida por usuario")
            return self._emergency_results()
        except Exception as e:
            print(f"\nError durante optimización resumida: {e}")
            return self._emergency_results()
        
    def _main_optimization_loop(self):
        """Loop principal de optimización"""
        max_iterations = self.algo_config['max_iterations']
        max_time = self.algo_config['max_time_hours'] * 3600
        
        for iteration in range(max_iterations):
            self.current_iteration = iteration
            iteration_start = time.time()
            
            elapsed_time = time.time() - self.start_time
            if elapsed_time > max_time:
                print(f"Límite de tiempo alcanzado ({elapsed_time/3600:.1f}h)")
                break
                
            print(f"\nIteración {iteration + 1}/{max_iterations}")
            
            print("  Fase de combinación...")
            new_solutions = self._combination_phase()
            
            print("  Fase de mejora...")
            improved_solutions = self._improvement_phase(new_solutions)
            
            print("  Actualizando RefSet...")
            self.reference_set = self._update_reference_set(improved_solutions)
            
            iteration_time = time.time() - iteration_start
            self._log_iteration_progress(iteration, iteration_time)
            
            if iteration % self.output_config.get('save_frequency', 2) == 0:
                self._save_checkpoint(iteration)
                
    def _combination_phase(self) -> List[Dict[str, Any]]:
        """Combina hiperparámetros del RefSet para crear nuevas soluciones"""
        new_solutions = []
        
        for i in range(len(self.reference_set)):
            for j in range(i + 1, len(self.reference_set)):
                if random.random() < self.algo_config['combination_probability']:
                    child = self._combine_hyperparameters(
                        self.reference_set[i], self.reference_set[j]
                    )
                    new_solutions.append(child)
        
        filtered_solutions = self._filter_similar_solutions(new_solutions, self.current_iteration)
        
        print(f"    Generadas {len(new_solutions)} combinaciones, {len(filtered_solutions)} únicas")
        return filtered_solutions
        
    def _combine_hyperparameters(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Combina hiperparámetros de dos configuraciones padre"""
        child = {}
        alpha = random.uniform(0.3, 0.7)
        
        for param, config in self.hyperparameter_space.dqn_params.items():
            if config['type'] == 'float':
                child[param] = alpha * parent1[param] + (1 - alpha) * parent2[param]
                child[param] = np.clip(child[param], config['min'], config['max'])
            elif config['type'] == 'int_discrete':
                child[param] = random.choice([parent1[param], parent2[param]])
            elif config['type'] == 'bool':
                child[param] = random.choice([parent1[param], parent2[param]])
        
        for param, config in self.hyperparameter_space.reward_weights.items():
            child[param] = alpha * parent1[param] + (1 - alpha) * parent2[param]
            child[param] = np.clip(child[param], config['min'], config['max'])
        
        return child
        
    def _improvement_phase(self, solutions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fase de mejora local"""
        improved_solutions = []
        
        for solution in solutions:
            if random.random() < self.algo_config['improvement_probability']:
                improved = self._local_improvement(solution)
                improved_solutions.append(improved)
            else:
                improved_solutions.append(solution)
                
        print(f"    Mejoradas {len(improved_solutions)} soluciones")
        return improved_solutions
        
    def _local_improvement(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica mejora local a una configuración de hiperparámetros"""
        improved = solution.copy()
        
        all_params = list(solution.keys())
        param_to_improve = random.choice(all_params)
        
        param_config = (self.hyperparameter_space.dqn_params.get(param_to_improve) or 
                       self.hyperparameter_space.reward_weights.get(param_to_improve))
        
        if param_config:
            if param_config['type'] == 'float':
                param_range = param_config['max'] - param_config['min']
                std_dev = 0.05 * param_range
                perturbation = np.random.normal(0, std_dev)
                new_value = solution[param_to_improve] + perturbation
                improved[param_to_improve] = np.clip(new_value, param_config['min'], param_config['max'])
                
            elif param_config['type'] == 'int_discrete':
                values_list = param_config['values']
                current_index = values_list.index(solution[param_to_improve])
                possible_indices = []
                if current_index > 0:
                    possible_indices.append(current_index - 1)
                if current_index < len(values_list) - 1:
                    possible_indices.append(current_index + 1)
                if possible_indices:
                    new_index = random.choice(possible_indices)
                    improved[param_to_improve] = values_list[new_index]
                    
            elif param_config['type'] == 'bool':
                improved[param_to_improve] = not solution[param_to_improve]
        
        return improved
        
    def _get_adaptive_similarity_threshold(self, iteration: int) -> float:
        """Threshold basado en mejora del fitness Y progreso temporal"""
        
        max_iterations = self.algo_config['max_iterations']
        progress = iteration / max_iterations
        temporal_threshold = 0.95 - (0.20 * progress)  
        
        if len(self.iteration_history) >= 3:
            recent_fitness = [h['best_fitness'] for h in self.iteration_history[-3:]]
            improvement = (recent_fitness[-1] - recent_fitness[0]) / abs(recent_fitness[0] + 1e-6)
            
            if improvement > 0.05: 
                fitness_threshold = 0.80  # Explotar
            elif improvement > 0.01: 
                fitness_threshold = 0.85  # Balance
            else: 
                fitness_threshold = 0.95  # Explorar
            
            threshold = 0.6 * fitness_threshold + 0.4 * temporal_threshold
        else:
            threshold = temporal_threshold
        
        return np.clip(threshold, 0.75, 0.95)

    def _filter_similar_solutions(self, solutions: List[Dict[str, Any]], iteration: int = 0) -> List[Dict[str, Any]]:
        """Filtra configuraciones con threshold adaptativo"""
        if not solutions:
            return solutions
        
        threshold = self._get_adaptive_similarity_threshold(iteration)
        
        filtered = [solutions[0]]
        
        for solution in solutions[1:]:
            is_unique = True
            for existing in filtered:
                if self.similarity_calculator.is_too_similar(solution, existing, threshold):
                    is_unique = False
                    break
            if is_unique:
                filtered.append(solution)
        
        print(f"    Threshold similitud: {threshold:.2f}, Filtradas: {len(solutions)} -> {len(filtered)}")
        return filtered
        
    def _update_reference_set(self, new_solutions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Actualiza el Reference Set con nuevas soluciones"""
        all_candidates = self.reference_set + new_solutions
        
        fitness_scores = self._evaluate_population(all_candidates, level="medium")
        
        new_ref_set = self._select_diverse_reference_set(
            all_candidates, fitness_scores, self.algo_config['ref_set_size']
        )
        
        evaluated_candidates = list(zip(all_candidates, fitness_scores))
        evaluated_candidates.sort(key=lambda x: x[1], reverse=True)
        self.best_solutions = [sol for sol, _ in evaluated_candidates[:5]]
        
        print(f"    RefSet actualizado")
        return new_ref_set
        
    def _select_diverse_reference_set(self, population: List[Dict[str, Any]], 
                                     fitness_scores: List[float], ref_set_size: int) -> List[Dict[str, Any]]:
        """Selecciona RefSet balanceando calidad y diversidad usando similitud coseno"""
        
        sorted_configs = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
        
        elite_count = max(1, int(ref_set_size * 0.6))
        elite_configs = [config for config, _ in sorted_configs[:elite_count]]
        
        remaining_configs = [config for config, _ in sorted_configs[elite_count:]]
        diverse_count = ref_set_size - elite_count
        
        if diverse_count > 0 and remaining_configs:
            diverse_configs = self.similarity_calculator.find_most_diverse_configs(
                remaining_configs, elite_configs, diverse_count
            )
            return elite_configs + diverse_configs
        else:
            return elite_configs[:ref_set_size]
    
    def _evaluate_population(self, population: List[Dict[str, Any]], level: str = "fast") -> List[float]:
        """Evalúa una población de configuraciones de hiperparámetros"""
        fitness_scores = []
        
        if level == "fast":
            systems_to_use = [self.systems_data[i] for i in self.eval_config['fast']['systems']]
            episodes = self.eval_config['fast']['episodes']
        elif level == "medium":
            systems_to_use = [self.systems_data[i] for i in self.eval_config['medium']['systems']]
            episodes = self.eval_config['medium']['episodes']
        else:  
            systems_to_use = list(self.systems_data.values())
            episodes = self.eval_config['full']['episodes']
        
        print(f"    Evaluando {len(population)} configuraciones en {len(systems_to_use)} sistemas ({episodes} episodios)")
        
        total_configs = len(population)
        milestone_interval = max(1, total_configs // 5) 
        
        for i, config in enumerate(population):
            if i % milestone_interval == 0 or i == total_configs - 1:
                progress = (i + 1) / total_configs * 100
                print(f"      Progreso: {progress:.0f}% ({i+1}/{total_configs})")
            
            dqn_params = self._extract_dqn_params(config)
            reward_weights = self._extract_reward_weights(config)
            
            system_fitness = []
            for system_config in systems_to_use:
                try:
                    fitness = self._evaluate_single_config(dqn_params, reward_weights, system_config, episodes)
                    system_fitness.append(fitness)
                except Exception as e:
                    if "missing" in str(e) or "Error" in str(e):
                        print(f"        ⚠️ Error en configuración {i+1}: {str(e)[:50]}...")
                    system_fitness.append(-1000.0)
            
            avg_fitness = np.mean(system_fitness)
            fitness_scores.append(avg_fitness)
        
        best_fitness = max(fitness_scores)
        worst_fitness = min(fitness_scores)
        avg_fitness_all = np.mean(fitness_scores)
        
        print(f"     Evaluación completada:")
        print(f"       Mejor fitness: {best_fitness:.2f}")
        print(f"       Peor fitness: {worst_fitness:.2f}")
        print(f"       Promedio: {avg_fitness_all:.2f}")
        
        return fitness_scores
    
    def _evaluate_single_config(self, dqn_params: Dict, reward_weights: Dict, 
                           system_config: Dict, episodes: int) -> float:
        """Evalúa una configuración en un sistema específico"""
        
        env = EVChargingEnv(system_config)
        env.update_reward_weights(reward_weights)
        
        state_size = 40 
        action_size = 60  
        
        agent = EnhancedDQNAgentPyTorch(state_size, action_size, **dqn_params)
        
        results = train_dqn_agent(agent, env, episodes,verbose=False)
        
        if len(results) >= 10:
            recent_rewards = [ep['reward'] for ep in results[-10:]]
        else:
            recent_rewards = [ep['reward'] for ep in results]
        
        return np.mean(recent_rewards)
    
    def _extract_dqn_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae hiperparámetros DQN de una configuración"""
        dqn_param_names = [
            'learning_rate', 'gamma', 'epsilon_start', 'epsilon_min',
            'epsilon_decay', 'batch_size', 'target_update_freq', 
            'memory_size', 'dueling_network'
        ]
        
        dqn_params = {}
        for param in dqn_param_names:
            if param in config:
                if param == 'epsilon_start':
                    dqn_params['epsilon'] = config[param]
                else:
                    dqn_params[param] = config[param]
        
        return dqn_params
    
    def _extract_reward_weights(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae pesos de recompensa de una configuración"""
        weight_param_names = [
            'energy_satisfaction_weight', 'energy_cost_weight',
            'penalty_skipped_vehicle', 'reward_assigned_vehicle'
        ]
        
        return {param: config[param] for param in weight_param_names if param in config}
    
    def _train_and_save_best_models(self, candidates, fitness_scores):
        """Entrena y guarda solo los mejores modelos"""
        print("Entrenando y guardando modelos óptimos...")
        
        models_dir = os.path.join(self.main_output_dir, "trained_models")
        os.makedirs(models_dir, exist_ok=True)
        
        for i, (config, fitness) in enumerate(zip(candidates, fitness_scores)):
            if fitness > -500:  
                print(f"Entrenando modelo óptimo {i+1}/5...")
                
                trained_agent = self._train_final_model(config, episodes=100)
                
                archetype = self._classify_solution_archetype(config)
                model_path = f"{models_dir}/{archetype}_rank_{i+1}.pt"
                trained_agent.save(model_path)
                
                print(f"Modelo guardado: {model_path}")

    def _final_evaluation(self) -> Dict[str, Any]:
        """Evaluación final completa de los mejores candidatos"""
        print("Evaluación final completa...")
        
        top_candidates = self.best_solutions[:5]
        final_fitness = self._evaluate_population(top_candidates, level="full")
        
        trained_models = self._train_and_save_best_models(top_candidates, final_fitness)

        final_results = []
        for i, (solution, fitness) in enumerate(zip(top_candidates, final_fitness)):
            archetype = self._classify_solution_archetype(solution)
            
            result = {
                'rank': i + 1,
                'fitness': fitness,
                'archetype': archetype,
                'hyperparameters': solution,
                'configuration_namef': f"optimized_{archetype}_{i+1}"
            }
            final_results.append(result)
            
        final_results.sort(key=lambda x: x['fitness'], reverse=True)
        
        return {
            'best_solutions': final_results,
            'optimization_summary': self._create_optimization_summary(),
            'execution_time': time.time() - self.start_time
        }
    
    def _classify_solution_archetype(self, solution: Dict[str, Any]) -> str:
        """Clasifica una solución según su arquetipo de comportamiento"""
        
        energy_satisfaction = solution.get('energy_satisfaction_weight', 1.0)
        energy_cost = solution.get('energy_cost_weight', 0.1)
        penalty_skipped = solution.get('penalty_skipped_vehicle', 50.0)
        reward_assigned = solution.get('reward_assigned_vehicle', 20.0)
        
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
        
    def _log_iteration_progress(self, iteration: int, iteration_time: float):
        """Log del progreso de la iteración"""
        elapsed_total = time.time() - self.start_time
        
        if self.best_solutions:
            best_config = self.best_solutions[0]
            dqn_params = self._extract_dqn_params(best_config)
            reward_weights = self._extract_reward_weights(best_config)
            
            first_system = list(self.systems_data.values())[0]
            current_best_fitness = self._evaluate_single_config(
                dqn_params, reward_weights, first_system, 10
            )
        else:
            current_best_fitness = 0
            
        print(f"      Tiempo iteración: {iteration_time:.1f}s")
        print(f"      Tiempo total: {elapsed_total/3600:.2f}h")
        print(f"      Mejor fitness actual: {current_best_fitness:.2f}")
        
        diversity_metrics = self.similarity_calculator.calculate_diversity_metrics(self.reference_set)
        print(f"      Diversidad RefSet: {diversity_metrics['diversity_score']:.3f}")
        
        self.iteration_history.append({
            'iteration': iteration + 1,
            'time': elapsed_total,
            'best_fitness': current_best_fitness,
            'iteration_duration': iteration_time,
            'diversity_score': diversity_metrics['diversity_score']
        })
    
    def _create_optimization_summary(self) -> Dict[str, Any]:
        """Crea resumen de la optimización"""
        
        if not self.iteration_history:
            return {}
        
        return {
            'total_iterations': len(self.iteration_history),
            'total_time_hours': (time.time() - self.start_time) / 3600,
            'final_best_fitness': self.iteration_history[-1]['best_fitness'],
            'initial_best_fitness': self.iteration_history[0]['best_fitness'],
            'improvement': self.iteration_history[-1]['best_fitness'] - self.iteration_history[0]['best_fitness'],
            'final_diversity': self.iteration_history[-1]['diversity_score'],
            'iteration_history': self.iteration_history
        }
    
    def _save_checkpoint(self, iteration: int):
        """Guarda checkpoint del algoritmo"""
        checkpoint_dir = os.path.join(self.main_output_dir, "checkpoints")
        checkpoint_path = f"{checkpoint_dir}/scatter_checkpoint_iter_{iteration}.json"
        checkpoint_data = {
            'iteration': iteration,
            'reference_set': self.reference_set,
            'best_solutions': self.best_solutions,
            'iteration_history': self.iteration_history,
            'config': self.config
        }
        
        checkpoint_path = f"{self.output_config.get('checkpoint_dir', './checkpoints')}/scatter_checkpoint_iter_{iteration}.json"
        
        import json
        import os
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        print(f"      Checkpoint guardado: {checkpoint_path}")
    
    def _emergency_results(self) -> Dict[str, Any]:
        """Retorna resultados de emergencia en caso de error"""
        
        emergency_solutions = []
        
        if hasattr(self, 'best_solutions') and self.best_solutions:
            print("⚡ Evaluación rápida de emergencia para obtener fitness real...")
            
            # Evaluar con nivel "fast" para obtener fitness real rápidamente
            try:
                emergency_fitness = self._evaluate_population(self.best_solutions, level="fast")
                
                for i, (solution, fitness) in enumerate(zip(self.best_solutions, emergency_fitness)):
                    emergency_solutions.append({
                        'rank': i + 1,
                        'fitness': fitness,  # ← FITNESS REAL evaluado rápidamente
                        'archetype': self._classify_solution_archetype(solution),
                        'hyperparameters': solution,
                        'configuration_name': f"emergency_{i+1}"
                    })
            except Exception as e:
                print(f"⚠️ No se pudo evaluar fitness en emergencia: {e}")
                # Solo si falla completamente, entonces no incluir fitness
                for i, solution in enumerate(self.best_solutions):
                    emergency_solutions.append({
                        'rank': i + 1,
                        # 'fitness': NO INCLUIR LA CLAVE SI NO LA TENEMOS
                        'archetype': self._classify_solution_archetype(solution),
                        'hyperparameters': solution,
                        'configuration_name': f"emergency_{i+1}",
                        'note': 'fitness_not_evaluated'
                    })
        
        return {
            'best_solutions': emergency_solutions,
            'optimization_summary': self._create_optimization_summary(),
            'execution_time': time.time() - self.start_time if self.start_time else 0,
            'status': 'interrupted'
        }