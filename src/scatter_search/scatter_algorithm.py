"""
Implementación del algoritmo Scatter Search principal
"""
import random
import numpy as np
from typing import Dict, List, Tuple, Any
import time
from datetime import datetime
import yaml
from .hyperparameter_space import HyperparameterSpace
from .evaluation_engine import EvaluationEngine
from .diversity_manager import DiversityManager
from .solution_generator import SolutionGenerator

class ScatterSearchOptimizer:
    """Implementación principal del algoritmo Scatter Search"""
    
    def __init__(self, config_path: str, systems_data: Dict):
        self.load_config(config_path)
        self.systems_data = systems_data
        
        # Initialize components
        self.hyperparameter_space = HyperparameterSpace(
            "configs/hyperparameter_ranges.yaml"
        )
        self.evaluation_engine = EvaluationEngine(config_path, systems_data)
        self.diversity_manager = DiversityManager(self.hyperparameter_space)
        self.solution_generator = SolutionGenerator(self.hyperparameter_space)
        
        # Algorithm state
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
        
    def run_optimization(self) -> Dict[str, Any]:
        """Ejecuta el algoritmo completo de Scatter Search"""
        print("🚀 Iniciando Scatter Search Optimization")
        print(f"  Tiempo máximo: {self.algo_config['max_time_hours']} horas")
        print(f"👥 Población: {self.algo_config['population_size']}")
        print(f"🎯 RefSet: {self.algo_config['ref_set_size']}")
        
        self.start_time = time.time()
        
        try:
            # Fase 1: Generación de población inicial
            print("\n📊 Fase 1: Generando población inicial...")
            self.population = self._generate_initial_population()
            
            # Fase 2: Construcción RefSet inicial
            print("\n🔍 Fase 2: Construyendo Reference Set inicial...")
            self.reference_set = self._build_initial_reference_set()
            
            # Fase 3: Iteraciones principales
            print("\n🔄 Fase 3: Iteraciones principales...")
            self._main_optimization_loop()
            
            # Fase 4: Evaluación final
            print("\n🏆 Fase 4: Evaluación final...")
            final_results = self._final_evaluation()
            
            return final_results
            
        except KeyboardInterrupt:
            print("\n⚠️  Optimización interrumpida por usuario")
            return self._emergency_results()
        except Exception as e:
            print(f"\n❌ Error durante optimización: {e}")
            return self._emergency_results()
            
    def _generate_initial_population(self) -> List[Dict[str, Any]]:
        """Genera población inicial con diversidad garantizada"""
        population = []
        pop_size = self.algo_config['population_size']
        
        # Distribución de arquetipos
        archetype_names = list(self.hyperparameter_space.archetypes.keys())
        individuals_per_archetype = max(2, pop_size // (len(archetype_names) * 3))
        
        # Generar individuos por arquetipo
        for archetype in archetype_names:
            for _ in range(individuals_per_archetype):
                solution = self.hyperparameter_space.generate_archetype_solution(archetype)
                population.append(solution)
                
        # Llenar resto con soluciones aleatorias
        while len(population) < pop_size:
            solution = self.hyperparameter_space.generate_random_solution()
            population.append(solution)
            
        print(f"Población generada: {len(population)} individuos")
        print(f"   - Por arquetipo: {individuals_per_archetype}")
        print(f"   - Aleatorios: {pop_size - len(archetype_names) * individuals_per_archetype}")
        
        return population
        
    def _build_initial_reference_set(self) -> List[Dict[str, Any]]:
        """Construye el Reference Set inicial"""
        print("📈 Evaluando población inicial (nivel fast)...")
        
        # Evaluar toda la población
        fitness_scores = self.evaluation_engine.batch_evaluate(
            self.population, level="fast"
        )
        
        # Combinar soluciones con sus fitness
        evaluated_population = list(zip(self.population, fitness_scores))
        evaluated_population.sort(key=lambda x: x[1], reverse=True)
        
        # Selección balanceada para RefSet
        ref_set_size = self.algo_config['ref_set_size']
        elite_count = self.algo_config['elite_count']
        diverse_count = self.algo_config['diverse_count']
        
        reference_set = []
        
        # b1: Elite solutions (mejores por fitness)
        for i in range(elite_count):
            reference_set.append(evaluated_population[i][0])
            
        # b2: Diverse solutions
        remaining_solutions = [sol for sol, _ in evaluated_population[elite_count:]]
        diverse_solutions = self.diversity_manager.select_diverse_subset(
            remaining_solutions, reference_set, diverse_count
        )
        reference_set.extend(diverse_solutions)
        
        print(f"RefSet construido: {len(reference_set)} soluciones")
        print(f"   - Elite: {elite_count}")
        print(f"   - Diverse: {len(diverse_solutions)}")
        
        # Guardar mejores para tracking
        self.best_solutions = [sol for sol, _ in evaluated_population[:5]]
        
        return reference_set
        
    def _main_optimization_loop(self):
        """Loop principal de optimización"""
        max_iterations = self.algo_config['max_iterations']
        max_time = self.algo_config['max_time_hours'] * 3600
        
        for iteration in range(max_iterations):
            self.current_iteration = iteration
            iteration_start = time.time()
            
            # Check time limit
            elapsed_time = time.time() - self.start_time
            if elapsed_time > max_time:
                print(f"⏰ Límite de tiempo alcanzado ({elapsed_time/3600:.1f}h)")
                break
                
            print(f"\n🔄 Iteración {iteration + 1}/{max_iterations}")
            
            # Combination phase
            print("  🧬 Fase de combinación...")
            new_solutions = self._combination_phase()
            
            # Improvement phase  
            print("  🔧 Fase de mejora...")
            improved_solutions = self._improvement_phase(new_solutions)
            
            # RefSet update
            print("  📊 Actualizando RefSet...")
            self.reference_set = self._update_reference_set(improved_solutions)
            
            # Progress tracking
            iteration_time = time.time() - iteration_start
            self._log_iteration_progress(iteration, iteration_time)
            
            # Save checkpoint
            if iteration % self.output_config.get('save_frequency', 2) == 0:
                self._save_checkpoint(iteration)
                
    def _combination_phase(self) -> List[Dict[str, Any]]:
        """Fase de combinación de soluciones"""
        new_solutions = []
        ref_set = self.reference_set
        
        # Combinaciones por pares
        for i in range(len(ref_set)):
            for j in range(i + 1, len(ref_set)):
                if random.random() < self.algo_config['combination_probability']:
                    combined_solution = self.solution_generator.combine_solutions(
                        ref_set[i], ref_set[j]
                    )
                    new_solutions.append(combined_solution)
                    
        print(f"    Generadas {len(new_solutions)} combinaciones")
        return new_solutions
        
    def _improvement_phase(self, solutions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fase de mejora local"""
        improved_solutions = []
        
        for solution in solutions:
            if random.random() < self.algo_config['improvement_probability']:
                improved = self.solution_generator.local_improvement(solution)
                improved_solutions.append(improved)
            else:
                improved_solutions.append(solution)
                
        print(f"    Mejoradas {len(improved_solutions)} soluciones")
        return improved_solutions
        
    def _update_reference_set(self, new_solutions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Actualiza el Reference Set con nuevas soluciones"""
        # Combine current RefSet + new solutions
        all_candidates = self.reference_set + new_solutions
        
        # Evaluate all candidates (medium level for better precision)
        fitness_scores = self.evaluation_engine.batch_evaluate(
            all_candidates, level="medium"
        )
        
        # Select new RefSet balancing quality and diversity
        evaluated_candidates = list(zip(all_candidates, fitness_scores))
        
        new_ref_set = self.diversity_manager.select_balanced_refset(
            evaluated_candidates, 
            ref_set_size=self.algo_config['ref_set_size'],
            elite_count=self.algo_config['elite_count']
        )
        
        # Update best solutions tracking
        evaluated_candidates.sort(key=lambda x: x[1], reverse=True)
        current_best = [sol for sol, _ in evaluated_candidates[:5]]
        self.best_solutions = current_best
        
        print(f"    RefSet actualizado")
        return new_ref_set
        
    def _final_evaluation(self) -> Dict[str, Any]:
        """Evaluación final completa de los mejores candidatos"""
        print("🔍 Evaluación final completa...")
        
        # Evaluar top candidatos con nivel "full"
        top_candidates = self.best_solutions[:5]
        final_fitness = self.evaluation_engine.batch_evaluate(
            top_candidates, level="full"
        )
        
        # Crear resultados finales
        final_results = []
        for i, (solution, fitness) in enumerate(zip(top_candidates, final_fitness)):
            archetype = self.diversity_manager.classify_solution_archetype(solution)
            
            result = {
                'rank': i + 1,
                'fitness': fitness,
                'archetype': archetype,
                'hyperparameters': solution,
                'configuration_name': f"optimized_{archetype}_{i+1}"
            }
            final_results.append(result)
            
        # Sort by fitness
        final_results.sort(key=lambda x: x['fitness'], reverse=True)
        
        return {
            'best_solutions': final_results,
            'optimization_summary': self._create_optimization_summary(),
            'execution_time': time.time() - self.start_time
        }
        
    def _log_iteration_progress(self, iteration: int, iteration_time: float):
        """Log del progreso de la iteración"""
        elapsed_total = time.time() - self.start_time
        
        # Get current best fitness
        if self.best_solutions:
            current_best_fitness = self.evaluation_engine.evaluate_solution(
                self.best_solutions[0], level="fast"
            )
        else:
            current_best_fitness = 0
            
        print(f"      Tiempo iteración: {iteration_time:.1f}s")
        print(f"      Tiempo total: {elapsed_total/3600:.2f}h")
        print(f"     Mejor fitness actual: {current_best_fitness:.2f}")
        
        # Store iteration data
        self.iteration_history.append({
            'iteration': iteration + 1,
            'time': elapsed_total,
            'best_fitness': current_best_fitness,
            'iteration_duration': iteration_time
        })