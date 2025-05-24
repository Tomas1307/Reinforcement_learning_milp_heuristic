"""
Análisis y exportación de resultados del Scatter Search
"""
import os
import json
import numpy as np
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Any

class ResultsAnalyzer:
    """Analiza y exporta resultados de la optimización"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.ensure_output_dirs()
        
    def ensure_output_dirs(self):
        """Crea directorios de salida necesarios"""
        dirs_to_create = [
            self.output_dir,
            os.path.join(self.output_dir, "configurations"),
            os.path.join(self.output_dir, "visualizations"),
            os.path.join(self.output_dir, "analysis"),
            os.path.join(self.output_dir, "exports")
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
            
    def save_complete_results(self, results: Dict[str, Any]):
        """Guarda resultados completos en formato JSON"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Resultados principales
        results_file = os.path.join(self.output_dir, f"scatter_search_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        # Resumen ejecutivo
        summary_file = os.path.join(self.output_dir, "optimization_summary.json")
        summary = self._create_executive_summary(results)
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        print(f"Resultados guardados en: {results_file}")
        print(f"Resumen guardado en: {summary_file}")
        
    def generate_summary_report(self, results: Dict[str, Any]):
        """Genera reporte de resumen en texto"""
        
        report_path = os.path.join(self.output_dir, "optimization_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("SCATTER SEARCH OPTIMIZATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Información general
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Tiempo total: {results['execution_time']/3600:.2f} horas\n\n")
            
            # Mejores soluciones
            f.write("MEJORES SOLUCIONES ENCONTRADAS:\n")
            f.write("-" * 30 + "\n")
            
            for i, solution in enumerate(results['best_solutions'][:5]):
                f.write(f"\n{i+1}. {solution['archetype'].upper()}\n")
                f.write(f"   Fitness: {solution['fitness']:.2f}\n")
                f.write(f"   Configuración: {solution['configuration_name']}\n")
                
                # Hiperparámetros clave
                hp = solution['hyperparameters']
                f.write(f"   Learning Rate: {hp.get('learning_rate', 'N/A')}\n")
                f.write(f"   Gamma: {hp.get('gamma', 'N/A')}\n")
                f.write(f"   Satisfaction Weight: {hp.get('energy_satisfaction_weight', 'N/A')}\n")
                f.write(f"   Cost Weight: {hp.get('energy_cost_weight', 'N/A')}\n")
                
            # Análisis de arquetipos
            f.write(f"\n\nANÁLISIS DE ARQUETIPOS:\n")
            f.write("-" * 20 + "\n")
            
            archetype_analysis = self._analyze_archetypes(results['best_solutions'])
            for archetype, analysis in archetype_analysis.items():
                f.write(f"\n{archetype.upper()}:\n")
                f.write(f"   Soluciones encontradas: {analysis['count']}\n")
                f.write(f"   Fitness promedio: {analysis['avg_fitness']:.2f}\n")
                f.write(f"   Características distintivas: {analysis['characteristics']}\n")
                
        print(f"Reporte generado en: {report_path}")
        
    def create_configuration_files(self, best_solutions: List[Dict]):
        """Crea archivos de configuración YAML para cada solución"""
        
        config_dir = os.path.join(self.output_dir, "configurations")
        
        for solution in best_solutions:
            archetype = solution['archetype']
            rank = solution['rank']
            fitness = solution['fitness']
            
            # Crear configuración completa
            config = {
                'metadata': {
                    'archetype': archetype,
                    'rank': rank,
                    'fitness': fitness,
                    'generated_date': datetime.now().isoformat(),
                    'optimization_method': 'scatter_search'
                },
                'dqn_hyperparameters': self._extract_dqn_params(solution['hyperparameters']),
                'reward_weights': self._extract_reward_weights(solution['hyperparameters']),
                'description': self._get_archetype_description(archetype),
                'recommended_usage': self._get_usage_recommendations(archetype)
            }
            
            # Guardar configuración
            config_filename = f"{archetype}_rank_{rank}.yaml"
            config_path = os.path.join(config_dir, config_filename)
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
                
        print(f"Configuraciones exportadas a: {config_dir}")
        
    def generate_visualizations(self, results: Dict[str, Any]):
        """Genera visualizaciones de los resultados"""
        
        viz_dir = os.path.join(self.output_dir, "visualizations")
        
        # 1. Distribución de fitness por arquetipo
        self._plot_fitness_by_archetype(results['best_solutions'], viz_dir)
        
        # 2. Heatmap de hiperparámetros
        self._plot_hyperparameter_heatmap(results['best_solutions'], viz_dir)
        
        # 3. Análisis de trade-offs
        self._plot_tradeoff_analysis(results['best_solutions'], viz_dir)
        
        # 4. Convergencia del algoritmo (si hay datos históricos)
        if 'optimization_summary' in results and 'iteration_history' in results['optimization_summary']:
            self._plot_convergence(results['optimization_summary']['iteration_history'], viz_dir)
            
        print(f"Visualizaciones generadas en: {viz_dir}")
        
    def _create_executive_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Crea resumen ejecutivo de los resultados"""
        
        best_solutions = results['best_solutions']
        
        return {
            'optimization_completed': True,
            'execution_time_hours': results['execution_time'] / 3600,
            'best_fitness': best_solutions[0]['fitness'] if best_solutions else 0,
            'total_solutions_found': len(best_solutions),
            'archetypes_discovered': list(set(sol['archetype'] for sol in best_solutions)),
            'top_3_solutions': [
                {
                    'rank': sol['rank'],
                    'archetype': sol['archetype'],
                    'fitness': sol['fitness'],
                    'config_file': f"{sol['archetype']}_rank_{sol['rank']}.yaml"
                }
                for sol in best_solutions[:3]
            ],
            'performance_improvement': self._calculate_improvement_over_baseline(best_solutions),
            'recommendations': self._generate_recommendations(best_solutions)
        }
        
    def _analyze_archetypes(self, solutions: List[Dict]) -> Dict[str, Dict]:
        """Analiza patrones por arquetipo"""
        
        archetype_data = {}
        
        for solution in solutions:
            archetype = solution['archetype']
            
            if archetype not in archetype_data:
                archetype_data[archetype] = {
                    'solutions': [],
                    'fitness_scores': []
                }
                
            archetype_data[archetype]['solutions'].append(solution)
            archetype_data[archetype]['fitness_scores'].append(solution['fitness'])
            
        # Análisis por arquetipo
        analysis = {}
        for archetype, data in archetype_data.items():
            analysis[archetype] = {
                'count': len(data['solutions']),
                'avg_fitness': np.mean(data['fitness_scores']),
                'std_fitness': np.std(data['fitness_scores']),
                'characteristics': self._extract_archetype_characteristics(data['solutions']),
                'best_solution': max(data['solutions'], key=lambda x: x['fitness'])
            }
            
        return analysis
        
    def _extract_dqn_params(self, hyperparameters: Dict) -> Dict:
        """Extrae solo hiperparámetros DQN"""
        dqn_params = [
            'learning_rate', 'gamma', 'epsilon_start', 'epsilon_min',
            'epsilon_decay', 'batch_size', 'target_update_freq', 
            'memory_size', 'dueling_network'
        ]
        
        return {param: hyperparameters[param] for param in dqn_params if param in hyperparameters}
        
    def _extract_reward_weights(self, hyperparameters: Dict) -> Dict:
        """Extrae solo pesos de recompensa"""
        weight_params = [
            'energy_satisfaction_weight', 'energy_cost_weight',
            'penalty_skipped_vehicle', 'reward_assigned_vehicle'
        ]
        
        return {param: hyperparameters[param] for param in weight_params if param in hyperparameters}
        
    def _get_archetype_description(self, archetype: str) -> str:
        """Retorna descripción del arquetipo"""
        descriptions = {
            'cost_minimizer': 'Optimiza costos de energía, acepta menor satisfacción cuando es económicamente justificable',
            'satisfaction_maximizer': 'Prioriza satisfacción completa del cliente, considera costos como factor secundario',
            'balanced_optimizer': 'Equilibra eficientemente costos y satisfacción para uso general',
            'urgency_focused': 'Nunca deja vehículos sin atender, ideal para servicios críticos',
            'efficiency_focused': 'Maximiza throughput y asignaciones, optimiza utilización de recursos'
        }
        return descriptions.get(archetype, 'Arquetipo personalizado optimizado por Scatter Search')
        
    def _get_usage_recommendations(self, archetype: str) -> List[str]:
        """Retorna recomendaciones de uso por arquetipo"""
        recommendations = {
            'cost_minimizer': [
                'Estaciones públicas con presión económica',
                'Operadores con márgenes ajustados',
                'Sistemas con alta variabilidad de precios'
            ],
            'satisfaction_maximizer': [
                'Hoteles y centros comerciales premium',
                'Servicios de alta gama',
                'Flotas corporativas ejecutivas'
            ],
            'balanced_optimizer': [
                'Oficinas corporativas',
                'Centros comerciales estándar',
                'Uso general en entornos mixtos'
            ],
            'urgency_focused': [
                'Hospitales y centros médicos',
                'Aeropuertos y transporte crítico',
                'Servicios de emergencia'
            ],
            'efficiency_focused': [
                'Flotas comerciales de alta rotación',
                'Estaciones de tránsito rápido',
                'Operaciones logísticas'
            ]
        }
        return recommendations.get(archetype, ['Uso especializado según optimización'])
        
    def _plot_fitness_by_archetype(self, solutions: List[Dict], viz_dir: str):
        """Gráfico de fitness por arquetipo"""
        
        df = pd.DataFrame([
            {'Archetype': sol['archetype'], 'Fitness': sol['fitness'], 'Rank': sol['rank']}
            for sol in solutions
        ])
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='Archetype', y='Fitness')
        plt.title('Distribución de Fitness por Arquetipo')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(os.path.join(viz_dir, 'fitness_by_archetype.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_hyperparameter_heatmap(self, solutions: List[Dict], viz_dir: str):
       """Heatmap de hiperparámetros normalizados"""
       
       # Extraer hiperparámetros numéricos
       numeric_params = []
       for sol in solutions:
           hp = sol['hyperparameters']
           numeric_hp = {k: v for k, v in hp.items() 
                        if isinstance(v, (int, float)) and k != 'dueling_network'}
           numeric_hp['archetype'] = sol['archetype']
           numeric_hp['rank'] = sol['rank']
           numeric_params.append(numeric_hp)
           
       df = pd.DataFrame(numeric_params)
       
       # Normalizar valores para mejor visualización
       numeric_cols = [col for col in df.columns if col not in ['archetype', 'rank']]
       df_normalized = df.copy()
       for col in numeric_cols:
           df_normalized[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
           
       # Crear heatmap
       plt.figure(figsize=(14, 8))
       
       # Pivot para tener arquetipos como filas
       pivot_data = df_normalized.groupby('archetype')[numeric_cols].mean()
       
       sns.heatmap(pivot_data, annot=True, cmap='viridis', fmt='.3f', 
                  cbar_kws={'label': 'Valor Normalizado'})
       plt.title('Heatmap de Hiperparámetros por Arquetipo')
       plt.ylabel('Arquetipo')
       plt.xlabel('Hiperparámetros')
       plt.tight_layout()
       
       plt.savefig(os.path.join(viz_dir, 'hyperparameter_heatmap.png'), 
                  dpi=300, bbox_inches='tight')
       plt.close()
       
    def _plot_tradeoff_analysis(self, solutions: List[Dict], viz_dir: str):
       """Análisis de trade-offs costo vs satisfacción"""
       
       # Extraer pesos de recompensa
       tradeoff_data = []
       for sol in solutions:
           hp = sol['hyperparameters']
           tradeoff_data.append({
               'satisfaction_weight': hp.get('energy_satisfaction_weight', 1.0),
               'cost_weight': hp.get('energy_cost_weight', 0.1),
               'fitness': sol['fitness'],
               'archetype': sol['archetype'],
               'rank': sol['rank']
           })
           
       df = pd.DataFrame(tradeoff_data)
       
       # Scatter plot con colores por arquetipo
       plt.figure(figsize=(12, 8))
       
       for archetype in df['archetype'].unique():
           arch_data = df[df['archetype'] == archetype]
           plt.scatter(arch_data['cost_weight'], arch_data['satisfaction_weight'], 
                      s=arch_data['fitness']/10, alpha=0.7, label=archetype)
           
       plt.xlabel('Energy Cost Weight')
       plt.ylabel('Energy Satisfaction Weight')
       plt.title('Trade-off Analysis: Cost vs Satisfaction Weights\n(Tamaño = Fitness)')
       plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
       plt.grid(True, alpha=0.3)
       plt.tight_layout()
       
       plt.savefig(os.path.join(viz_dir, 'tradeoff_analysis.png'), 
                  dpi=300, bbox_inches='tight')
       plt.close()
       
    def _plot_convergence(self, iteration_history: List[Dict], viz_dir: str):
       """Gráfico de convergencia del algoritmo"""
       
       if not iteration_history:
           return
           
       df = pd.DataFrame(iteration_history)
       
       fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
       
       # Plot 1: Best fitness over time
       ax1.plot(df['iteration'], df['best_fitness'], 'b-o', linewidth=2)
       ax1.set_xlabel('Iteración')
       ax1.set_ylabel('Mejor Fitness')
       ax1.set_title('Convergencia del Algoritmo - Mejor Fitness')
       ax1.grid(True, alpha=0.3)
       
       # Plot 2: Time per iteration
       ax2.bar(df['iteration'], df['iteration_duration'], alpha=0.7, color='green')
       ax2.set_xlabel('Iteración')
       ax2.set_ylabel('Tiempo (segundos)')
       ax2.set_title('Tiempo por Iteración')
       ax2.grid(True, alpha=0.3)
       
       plt.tight_layout()
       plt.savefig(os.path.join(viz_dir, 'convergence_analysis.png'), 
                  dpi=300, bbox_inches='tight')
       plt.close()
       
    def _calculate_improvement_over_baseline(self, solutions: List[Dict]) -> Dict[str, float]:
       """Calcula mejora sobre baseline (asumiendo baseline conocido)"""
       
       if not solutions:
           return {'improvement_pct': 0.0, 'baseline_fitness': 0.0}
           
       # Baseline ficticio para ejemplo - en producción usar valores reales
       baseline_fitness = 800.0  # Valor baseline de tu sistema actual
       
       best_fitness = solutions[0]['fitness']
       improvement_pct = ((best_fitness - baseline_fitness) / baseline_fitness) * 100
       
       return {
           'improvement_pct': improvement_pct,
           'baseline_fitness': baseline_fitness,
           'best_fitness': best_fitness,
           'absolute_improvement': best_fitness - baseline_fitness
       }
       
    def _generate_recommendations(self, solutions: List[Dict]) -> List[str]:
       """Genera recomendaciones basadas en los resultados"""
       
       recommendations = []
       
       if not solutions:
           return ["No se encontraron soluciones válidas"]
           
       best_solution = solutions[0]
       
       # Recomendación general
       recommendations.append(
           f"Implementar configuración '{best_solution['archetype']}' como solución principal "
           f"(Fitness: {best_solution['fitness']:.2f})"
       )
       
       # Análisis de arquetipos
       archetypes_found = set(sol['archetype'] for sol in solutions)
       
       if len(archetypes_found) >= 3:
           recommendations.append(
               "Se encontraron múltiples arquetipos viables. Considere implementar "
               "sistema adaptativo que seleccione arquetipo según contexto operativo."
           )
           
       # Recomendaciones específicas por arquetipo dominante
       if best_solution['archetype'] == 'cost_minimizer':
           recommendations.append(
               "Arquetipo cost_minimizer detectado como óptimo. Ideal para operaciones "
               "con márgenes ajustados y alta sensibilidad a precios energéticos."
           )
       elif best_solution['archetype'] == 'satisfaction_maximizer':
           recommendations.append(
               "Arquetipo satisfaction_maximizer óptimo. Recomendado para servicios "
               "premium donde la experiencia del cliente es prioritaria."
           )
           
       # Recomendación de hiperparámetros clave
       hp = best_solution['hyperparameters']
       lr = hp.get('learning_rate', 0)
       if lr < 0.001:
           recommendations.append(
               f"Learning rate optimizado es conservador ({lr:.4f}). "
               "Indicativo de problema complejo que requiere aprendizaje gradual."
           )
       elif lr > 0.003:
           recommendations.append(
               f"Learning rate optimizado es agresivo ({lr:.4f}). "
               "Sugiere potencial para convergencia rápida en este dominio."
           )
           
       return recommendations
       
    def _extract_archetype_characteristics(self, solutions: List[Dict]) -> str:
       """Extrae características distintivas de un arquetipo"""
       
       if not solutions:
           return "Sin datos suficientes"
           
       # Promediar hiperparámetros del arquetipo
       hp_sums = {}
       hp_counts = {}
       
       for sol in solutions:
           for param, value in sol['hyperparameters'].items():
               if isinstance(value, (int, float)):
                   if param not in hp_sums:
                       hp_sums[param] = 0
                       hp_counts[param] = 0
                   hp_sums[param] += value
                   hp_counts[param] += 1
                   
       # Calcular promedios
       avg_params = {param: hp_sums[param] / hp_counts[param] 
                    for param in hp_sums.keys()}
       
       # Identificar características distintivas
       characteristics = []
       
       if 'energy_satisfaction_weight' in avg_params:
           sat_weight = avg_params['energy_satisfaction_weight']
           if sat_weight > 2.0:
               characteristics.append("Alta prioridad en satisfacción")
           elif sat_weight < 0.8:
               characteristics.append("Baja prioridad en satisfacción")
               
       if 'energy_cost_weight' in avg_params:
           cost_weight = avg_params['energy_cost_weight']
           if cost_weight > 0.5:
               characteristics.append("Fuerte enfoque en costos")
           elif cost_weight < 0.2:
               characteristics.append("Costos como factor secundario")
               
       if 'learning_rate' in avg_params:
           lr = avg_params['learning_rate']
           if lr > 0.003:
               characteristics.append("Aprendizaje agresivo")
           elif lr < 0.001:
               characteristics.append("Aprendizaje conservador")
               
       return ", ".join(characteristics) if characteristics else "Características balanceadas"
       
    def export_to_excel(self, results: Dict[str, Any]):
       """Exporta resultados a Excel para análisis adicional"""
       
       excel_path = os.path.join(self.output_dir, "exports", "scatter_search_analysis.xlsx")
       
       with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
           
           # Hoja 1: Mejores soluciones
           solutions_data = []
           for sol in results['best_solutions']:
               row = {
                   'Rank': sol['rank'],
                   'Archetype': sol['archetype'],
                   'Fitness': sol['fitness'],
                   'Config_File': sol['configuration_name']
               }
               row.update(sol['hyperparameters'])
               solutions_data.append(row)
               
           df_solutions = pd.DataFrame(solutions_data)
           df_solutions.to_excel(writer, sheet_name='Best_Solutions', index=False)
           
           # Hoja 2: Análisis por arquetipo
           archetype_analysis = self._analyze_archetypes(results['best_solutions'])
           df_archetypes = pd.DataFrame([
               {
                   'Archetype': arch,
                   'Count': data['count'],
                   'Avg_Fitness': data['avg_fitness'],
                   'Std_Fitness': data['std_fitness'],
                   'Characteristics': data['characteristics']
               }
               for arch, data in archetype_analysis.items()
           ])
           df_archetypes.to_excel(writer, sheet_name='Archetype_Analysis', index=False)
           
           # Hoja 3: Resumen ejecutivo
           if 'optimization_summary' in results:
               summary_data = []
               for key, value in results['optimization_summary'].items():
                   summary_data.append({'Metric': key, 'Value': str(value)})
               df_summary = pd.DataFrame(summary_data)
               df_summary.to_excel(writer, sheet_name='Executive_Summary', index=False)
               
       print(f"Análisis Excel exportado a: {excel_path}")
       
    def create_deployment_package(self, results: Dict[str, Any]):
       """Crea paquete completo para deployment"""
       
       package_dir = os.path.join(self.output_dir, "deployment_package")
       os.makedirs(package_dir, exist_ok=True)
       
       # README con instrucciones
       readme_path = os.path.join(package_dir, "README.md")
       with open(readme_path, 'w') as f:
           f.write("# Scatter Search Optimization Results\n\n")
           f.write("## Deployment Package\n\n")
           f.write("Este paquete contiene las configuraciones optimizadas listas para producción.\n\n")
           
           f.write("### Archivos incluidos:\n")
           for sol in results['best_solutions'][:3]:
               config_name = f"{sol['archetype']}_rank_{sol['rank']}.yaml"
               f.write(f"- `{config_name}`: {sol['archetype']} (Fitness: {sol['fitness']:.2f})\n")
               
           f.write("\n### Uso recomendado:\n")
           f.write("1. Seleccionar configuración según contexto operativo\n")
           f.write("2. Cargar hiperparámetros en el sistema DQN\n")
           f.write("3. Actualizar pesos de recompensa en el environment\n")
           f.write("4. Entrenar agente con nueva configuración\n\n")
           
           f.write("### Configuraciones por contexto:\n")
           for sol in results['best_solutions']:
               archetype = sol['archetype']
               recommendations = self._get_usage_recommendations(archetype)
               f.write(f"\n**{archetype.upper()}**:\n")
               for rec in recommendations:
                   f.write(f"- {rec}\n")
                   
       # Copiar archivos de configuración al paquete
       config_source = os.path.join(self.output_dir, "configurations")
       for sol in results['best_solutions'][:5]:  # Top 5 soluciones
           config_name = f"{sol['archetype']}_rank_{sol['rank']}.yaml"
           source_path = os.path.join(config_source, config_name)
           dest_path = os.path.join(package_dir, config_name)
           
           if os.path.exists(source_path):
               import shutil
               shutil.copy2(source_path, dest_path)
               
       print(f"Paquete de deployment creado en: {package_dir}")