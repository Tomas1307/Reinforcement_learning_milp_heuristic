"""
Advanced Analysis for Scatter Search Optimization Results
Generates comprehensive visualizations from checkpoint history
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from datetime import datetime
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class AdvancedScatterSearchAnalyzer:
    """Análisis avanzado de resultados de Scatter Search"""
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.checkpoints_dir = os.path.join(base_dir, "checkpoints")
        self.analysis_dir = os.path.join(base_dir, "analysis")
        self.optimization_summary_path = os.path.join(base_dir, "optimization_summary.json")
        
        os.makedirs(self.analysis_dir, exist_ok=True)
        
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_checkpoint_history(self) -> List[Dict]:
        """Carga todo el historial de checkpoints ordenado"""
        checkpoint_files = glob.glob(os.path.join(self.checkpoints_dir, "scatter_checkpoint_iter_*.json"))
        
        checkpoints = []
        for file_path in checkpoint_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    iteration_num = int(os.path.basename(file_path).split('_')[-1].split('.')[0])
                    data['checkpoint_iteration'] = iteration_num
                    data['checkpoint_file'] = file_path
                    checkpoints.append(data)
            except Exception as e:
                print(f"Error cargando {file_path}: {e}")
        
        checkpoints.sort(key=lambda x: x['checkpoint_iteration'])
        print(f"Cargados {len(checkpoints)} checkpoints (iteraciones {checkpoints[0]['checkpoint_iteration']}-{checkpoints[-1]['checkpoint_iteration']})")
        
        return checkpoints
    
    def load_optimization_summary(self) -> Optional[Dict]:
        """Carga el resumen de optimización final"""
        try:
            with open(self.optimization_summary_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Archivo de resumen no encontrado: {self.optimization_summary_path}")
            return None
        except Exception as e:
            print(f"Error cargando resumen: {e}")
            return None
    
    def extract_historical_data(self, checkpoints: List[Dict]) -> pd.DataFrame:
        """Extrae datos históricos de todos los checkpoints"""
        historical_data = []
        
        for checkpoint in checkpoints:
            checkpoint_iter = checkpoint['checkpoint_iteration']
            
            if 'reference_set' in checkpoint:
                for i, solution in enumerate(checkpoint['reference_set']):
                    row = {
                        'checkpoint_iteration': checkpoint_iter,
                        'solution_type': 'reference_set',
                        'solution_rank': i + 1,
                        'archetype': self._classify_archetype(solution),
                        'learning_rate': solution.get('learning_rate', np.nan),
                        'gamma': solution.get('gamma', np.nan),
                        'energy_satisfaction_weight': solution.get('energy_satisfaction_weight', np.nan),
                        'energy_cost_weight': solution.get('energy_cost_weight', np.nan),
                        'penalty_skipped_vehicle': solution.get('penalty_skipped_vehicle', np.nan),
                        'reward_assigned_vehicle': solution.get('reward_assigned_vehicle', np.nan),
                        'batch_size': solution.get('batch_size', np.nan),
                        'target_update_freq': solution.get('target_update_freq', np.nan),
                        'memory_size': solution.get('memory_size', np.nan),
                        'dueling_network': solution.get('dueling_network', False)
                    }
                    historical_data.append(row)
            
            if 'best_solutions' in checkpoint:
                for i, solution in enumerate(checkpoint['best_solutions']):
                    row = {
                        'checkpoint_iteration': checkpoint_iter,
                        'solution_type': 'best_solutions',
                        'solution_rank': i + 1,
                        'archetype': self._classify_archetype(solution),
                        'learning_rate': solution.get('learning_rate', np.nan),
                        'gamma': solution.get('gamma', np.nan),
                        'energy_satisfaction_weight': solution.get('energy_satisfaction_weight', np.nan),
                        'energy_cost_weight': solution.get('energy_cost_weight', np.nan),
                        'penalty_skipped_vehicle': solution.get('penalty_skipped_vehicle', np.nan),
                        'reward_assigned_vehicle': solution.get('reward_assigned_vehicle', np.nan),
                        'batch_size': solution.get('batch_size', np.nan),
                        'target_update_freq': solution.get('target_update_freq', np.nan),
                        'memory_size': solution.get('memory_size', np.nan),
                        'dueling_network': solution.get('dueling_network', False)
                    }
                    historical_data.append(row)
        
        return pd.DataFrame(historical_data)
    
    def extract_convergence_data(self, checkpoints: List[Dict]) -> pd.DataFrame:
        """Extrae datos de convergencia del historial de iteraciones"""
        convergence_data = []
        
        for checkpoint in checkpoints:
            if 'iteration_history' in checkpoint:
                for hist_entry in checkpoint['iteration_history']:
                    convergence_data.append({
                        'checkpoint_iteration': checkpoint['checkpoint_iteration'],
                        'iteration': hist_entry.get('iteration', 0),
                        'time_hours': hist_entry.get('time', 0) / 3600,
                        'best_fitness': hist_entry.get('best_fitness', 0),
                        'iteration_duration_min': hist_entry.get('iteration_duration', 0) / 60,
                        'diversity_score': hist_entry.get('diversity_score', 0)
                    })
        
        df = pd.DataFrame(convergence_data)
        
        if not df.empty:
            df = df.drop_duplicates(subset=['iteration'], keep='last')
            df = df.sort_values('iteration').reset_index(drop=True)
        
        return df
    
    def _classify_archetype(self, solution: Dict) -> str:
        """Clasifica el arquetipo de una solución"""
        energy_satisfaction = solution.get('energy_satisfaction_weight', 1.0)
        energy_cost = solution.get('energy_cost_weight', 0.1)
        penalty_skipped = solution.get('penalty_skipped_vehicle', 50.0)
        reward_assigned = solution.get('reward_assigned_vehicle', 20.0)
        
        cost_ratio = energy_cost / max(energy_satisfaction, 0.1)
        
        if energy_cost > 1.0 or cost_ratio > 0.4:
            return "cost_minimizer"
        elif penalty_skipped > 200.0:
            return "urgency_focused"
        elif reward_assigned > 100.0:
            return "efficiency_focused"
        elif energy_satisfaction > 4.0 and energy_cost < 0.5:
            return "satisfaction_maximizer"
        else:
            return "balanced_optimizer"
    
    def plot_convergence_analysis(self, convergence_df: pd.DataFrame):
        """Gráfico de análisis de convergencia completo"""
        if convergence_df.empty:
            print("No hay datos de convergencia para graficar")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análisis de Convergencia del Scatter Search', fontsize=16, fontweight='bold')
        
        axes[0, 0].plot(convergence_df['iteration'], convergence_df['best_fitness'], 
                       'b-o', linewidth=2, markersize=4, alpha=0.8)
        axes[0, 0].set_xlabel('Iteración')
        axes[0, 0].set_ylabel('Mejor Fitness')
        axes[0, 0].set_title('Evolución del Mejor Fitness')
        axes[0, 0].grid(True, alpha=0.3)
        
        if len(convergence_df) > 1:
            z = np.polyfit(convergence_df['iteration'], convergence_df['best_fitness'], 1)
            p = np.poly1d(z)
            axes[0, 0].plot(convergence_df['iteration'], p(convergence_df['iteration']), 
                           "r--", alpha=0.8, label=f'Tendencia (slope={z[0]:.2f})')
            axes[0, 0].legend()
        
        axes[0, 1].plot(convergence_df['iteration'], convergence_df['diversity_score'], 
                       'g-o', linewidth=2, markersize=4, alpha=0.8)
        axes[0, 1].set_xlabel('Iteración')
        axes[0, 1].set_ylabel('Score de Diversidad')
        axes[0, 1].set_title('Evolución de la Diversidad')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(convergence_df['iteration'], convergence_df['time_hours'], 
                       'purple', linewidth=2, alpha=0.8)
        axes[1, 0].set_xlabel('Iteración')
        axes[1, 0].set_ylabel('Tiempo Acumulado (horas)')
        axes[1, 0].set_title('Progreso Temporal')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].bar(convergence_df['iteration'], convergence_df['iteration_duration_min'], 
                      alpha=0.7, color='orange')
        axes[1, 1].set_xlabel('Iteración')
        axes[1, 1].set_ylabel('Duración por Iteración (min)')
        axes[1, 1].set_title('Tiempo por Iteración')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, 'convergence_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Gráfico de convergencia guardado")
    
    def plot_archetype_evolution(self, historical_df: pd.DataFrame):
        """Evolución de arquetipos a lo largo del tiempo"""
        if historical_df.empty:
            print("No hay datos históricos para graficar")
            return
        
        # Filtrar solo RefSet para analizar evolución
        refset_df = historical_df[historical_df['solution_type'] == 'reference_set']
        
        if refset_df.empty:
            print("No hay datos de RefSet para graficar")
            return
        
        archetype_counts = refset_df.groupby(['checkpoint_iteration', 'archetype']).size().unstack(fill_value=0)
        
        archetype_percentages = archetype_counts.div(archetype_counts.sum(axis=1), axis=0) * 100
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('Evolución de Arquetipos en el Reference Set', fontsize=16, fontweight='bold')
        
        archetype_percentages.plot(kind='area', stacked=True, ax=ax1, alpha=0.8)
        ax1.set_xlabel('Iteración del Checkpoint')
        ax1.set_ylabel('Porcentaje del RefSet (%)')
        ax1.set_title('Distribución Porcentual de Arquetipos')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        for archetype in archetype_counts.columns:
            ax2.plot(archetype_counts.index, archetype_counts[archetype], 
                    marker='o', linewidth=2, label=archetype, alpha=0.8)
        
        ax2.set_xlabel('Iteración del Checkpoint')
        ax2.set_ylabel('Número de Soluciones')
        ax2.set_title('Evolución del Número de Soluciones por Arquetipo')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, 'archetype_evolution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Gráfico de evolución de arquetipos guardado")
    
    def plot_hyperparameter_evolution(self, historical_df: pd.DataFrame):
        """Evolución de hiperparámetros clave"""
        if historical_df.empty:
            return
        
        best_df = historical_df[historical_df['solution_type'] == 'best_solutions']
        best_df = best_df[best_df['solution_rank'] == 1]  # Solo el mejor de cada checkpoint
        
        if best_df.empty:
            print("No hay datos de mejores soluciones para graficar")
            return
        
        key_params = ['learning_rate', 'gamma', 'energy_satisfaction_weight', 
                     'energy_cost_weight', 'penalty_skipped_vehicle', 'reward_assigned_vehicle']
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        fig.suptitle('Evolución de Hiperparámetros en la Mejor Solución', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for i, param in enumerate(key_params):
            if param in best_df.columns:
                param_data = best_df[['checkpoint_iteration', param]].dropna()
                
                if not param_data.empty:
                    axes[i].plot(param_data['checkpoint_iteration'], param_data[param], 
                               'o-', linewidth=2, markersize=6, alpha=0.8)
                    axes[i].set_xlabel('Iteración del Checkpoint')
                    axes[i].set_ylabel(param.replace('_', ' ').title())
                    axes[i].set_title(f'Evolución de {param.replace("_", " ").title()}')
                    axes[i].grid(True, alpha=0.3)
                    
                    # Añadir valores en puntos clave
                    for j, row in param_data.iterrows():
                        if j % max(1, len(param_data) // 5) == 0:  # Mostrar cada 5 puntos
                            axes[i].annotate(f'{row[param]:.3f}', 
                                           (row['checkpoint_iteration'], row[param]),
                                           xytext=(5, 5), textcoords='offset points',
                                           fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, 'hyperparameter_evolution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Gráfico de evolución de hiperparámetros guardado")
    
    def plot_diversity_vs_performance(self, convergence_df: pd.DataFrame):
        """Análisis de diversidad vs performance"""
        if convergence_df.empty:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Análisis de Diversidad vs Performance', fontsize=16, fontweight='bold')
        
        scatter = ax1.scatter(convergence_df['diversity_score'], convergence_df['best_fitness'], 
                            c=convergence_df['iteration'], cmap='viridis', alpha=0.7, s=60)
        ax1.set_xlabel('Score de Diversidad')
        ax1.set_ylabel('Mejor Fitness')
        ax1.set_title('Diversidad vs Fitness')
        ax1.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Iteración')
        
        if len(convergence_df) > 1:
            z = np.polyfit(convergence_df['diversity_score'], convergence_df['best_fitness'], 1)
            p = np.poly1d(z)
            ax1.plot(convergence_df['diversity_score'], p(convergence_df['diversity_score']), 
                    "r--", alpha=0.8, label=f'Tendencia (r={np.corrcoef(convergence_df["diversity_score"], convergence_df["best_fitness"])[0,1]:.3f})')
            ax1.legend()
        

        convergence_df['diversity_bin'] = pd.cut(convergence_df['diversity_score'], 
                                               bins=5, labels=['Muy Baja', 'Baja', 'Media', 'Alta', 'Muy Alta'])
        
        convergence_df.boxplot(column='best_fitness', by='diversity_bin', ax=ax2)
        ax2.set_xlabel('Nivel de Diversidad')
        ax2.set_ylabel('Mejor Fitness')
        ax2.set_title('Distribución de Fitness por Nivel de Diversidad')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, 'diversity_vs_performance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Gráfico de diversidad vs performance guardado")
    
    def plot_final_solutions_analysis(self, optimization_summary: Dict):
        """Análisis de las soluciones finales"""
        if not optimization_summary or 'best_solutions' not in optimization_summary:
            print("No hay datos de soluciones finales para graficar")
            return
        
        best_solutions = optimization_summary['best_solutions']
        
        if not best_solutions:
            return
        
        fitness_data = [sol.get('fitness', 0) for sol in best_solutions]
        archetype_data = [sol.get('archetype', 'unknown') for sol in best_solutions]
        
        hp_data = []
        for sol in best_solutions:
            hp = sol.get('hyperparameters', {})
            hp_data.append({
                'fitness': sol.get('fitness', 0),
                'archetype': sol.get('archetype', 'unknown'),
                'learning_rate': hp.get('learning_rate', 0),
                'gamma': hp.get('gamma', 0),
                'energy_satisfaction_weight': hp.get('energy_satisfaction_weight', 0),
                'energy_cost_weight': hp.get('energy_cost_weight', 0),
                'penalty_skipped_vehicle': hp.get('penalty_skipped_vehicle', 0),
                'reward_assigned_vehicle': hp.get('reward_assigned_vehicle', 0)
            })
        
        hp_df = pd.DataFrame(hp_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análisis de las Mejores Soluciones Finales', fontsize=16, fontweight='bold')
        
        axes[0, 0].bar(range(len(fitness_data)), fitness_data, 
                      color=sns.color_palette("viridis", len(fitness_data)))
        axes[0, 0].set_xlabel('Ranking de Solución')
        axes[0, 0].set_ylabel('Fitness')
        axes[0, 0].set_title('Distribución de Fitness en Top Solutions')
        axes[0, 0].grid(True, alpha=0.3)
        
        for i, v in enumerate(fitness_data):
            axes[0, 0].text(i, v + max(fitness_data) * 0.01, f'{v:.0f}', 
                           ha='center', va='bottom', fontsize=10)
        
        archetype_counts = pd.Series(archetype_data).value_counts()
        axes[0, 1].pie(archetype_counts.values, labels=archetype_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Distribución de Arquetipos en Top Solutions')
        
        if len(hp_df) > 1:
            scatter = axes[1, 0].scatter(hp_df['energy_cost_weight'], hp_df['energy_satisfaction_weight'], 
                                       c=hp_df['fitness'], cmap='viridis', s=100, alpha=0.8)
            axes[1, 0].set_xlabel('Energy Cost Weight')
            axes[1, 0].set_ylabel('Energy Satisfaction Weight')
            axes[1, 0].set_title('Trade-off Cost vs Satisfaction')
            axes[1, 0].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[1, 0], label='Fitness')
        
        categories = ['Learning Rate', 'Gamma', 'Satisfaction Weight', 'Cost Weight']
        
        best_sol = hp_df.iloc[0]  
        values = [
            best_sol['learning_rate'] / 0.01,  
            best_sol['gamma'],
            best_sol['energy_satisfaction_weight'] / 5.0,
            best_sol['energy_cost_weight'] / 2.0
        ]
        
        values += values[:1]
        categories += categories[:1]
        
        angles = np.linspace(0, 2 * np.pi, len(categories))
        
        axes[1, 1].plot(angles, values, 'o-', linewidth=2, alpha=0.8)
        axes[1, 1].fill(angles, values, alpha=0.3)
        axes[1, 1].set_xticks(angles[:-1])
        axes[1, 1].set_xticklabels(categories[:-1])
        axes[1, 1].set_title('Perfil de la Mejor Solución')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, 'final_solutions_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Gráfico de análisis de soluciones finales guardado")
    
    def generate_comprehensive_report(self, checkpoints: List[Dict], 
                                    convergence_df: pd.DataFrame, 
                                    historical_df: pd.DataFrame,
                                    optimization_summary: Dict):
        """Genera reporte comprehensivo en texto"""
        
        report_path = os.path.join(self.analysis_dir, 'comprehensive_analysis_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("REPORTE COMPREHENSIVO DE ANÁLISIS SCATTER SEARCH\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("1. INFORMACIÓN GENERAL:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Checkpoints analizados: {len(checkpoints)}\n")
            if not convergence_df.empty:
                f.write(f"Iteraciones completadas: {convergence_df['iteration'].max()}\n")
                f.write(f"Tiempo total: {convergence_df['time_hours'].max():.2f} horas\n")
            f.write(f"Soluciones históricas analizadas: {len(historical_df)}\n\n")
            
            if not convergence_df.empty:
                f.write("2. ANÁLISIS DE CONVERGENCIA:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Fitness inicial: {convergence_df['best_fitness'].iloc[0]:.2f}\n")
                f.write(f"Fitness final: {convergence_df['best_fitness'].iloc[-1]:.2f}\n")
                improvement = convergence_df['best_fitness'].iloc[-1] - convergence_df['best_fitness'].iloc[0]
                f.write(f"Mejora total: {improvement:.2f}\n")
                f.write(f"Mejor fitness alcanzado: {convergence_df['best_fitness'].max():.2f}\n")
                f.write(f"Diversidad promedio: {convergence_df['diversity_score'].mean():.3f}\n\n")
            
            if not historical_df.empty:
                f.write("3. ANÁLISIS DE ARQUETIPOS:\n")
                f.write("-" * 30 + "\n")
                
                refset_df = historical_df[historical_df['solution_type'] == 'reference_set']
                if not refset_df.empty:
                    archetype_dist = refset_df['archetype'].value_counts()
                    f.write("Distribución total en Reference Sets:\n")
                    for archetype, count in archetype_dist.items():
                        percentage = (count / len(refset_df)) * 100
                        f.write(f"  {archetype}: {count} ({percentage:.1f}%)\n")
                f.write("\n")
            
            if optimization_summary and 'best_solutions' in optimization_summary:
                f.write("4. SOLUCIONES FINALES:\n")
                f.write("-" * 25 + "\n")
                
                best_solutions = optimization_summary['best_solutions']
                for i, sol in enumerate(best_solutions[:5]):
                    f.write(f"Rank {i+1}: {sol.get('archetype', 'unknown').upper()}\n")
                    f.write(f"  Fitness: {sol.get('fitness', 0):.2f}\n")
                    
                    hp = sol.get('hyperparameters', {})
                    f.write(f"  Learning Rate: {hp.get('learning_rate', 0):.6f}\n")
                    f.write(f"  Gamma: {hp.get('gamma', 0):.6f}\n")
                    f.write(f"  Satisfaction Weight: {hp.get('energy_satisfaction_weight', 0):.2f}\n")
                    f.write(f"  Cost Weight: {hp.get('energy_cost_weight', 0):.6f}\n\n")
            
            f.write("5. RECOMENDACIONES:\n")
            f.write("-" * 20 + "\n")
            
            if not convergence_df.empty:
                final_diversity = convergence_df['diversity_score'].iloc[-1]
                if final_diversity < 0.3:
                    f.write("• La diversidad final es baja, considera aumentar parámetros de diversidad\n")
                elif final_diversity > 0.4:
                    f.write("• Buena diversidad mantenida durante la optimización\n")
                
                if convergence_df['best_fitness'].iloc[-1] == convergence_df['best_fitness'].max():
                    f.write("• El algoritmo converge en las últimas iteraciones - tiempo bien utilizado\n")
                else:
                    f.write("• El mejor fitness no está al final - posible convergencia temprana\n")
            
            if optimization_summary:
                best_solutions = optimization_summary.get('best_solutions', [])
                archetypes = [sol.get('archetype') for sol in best_solutions]
                unique_archetypes = len(set(archetypes))
                
                if unique_archetypes == 1:
                    f.write("• Todas las mejores soluciones son del mismo arquetipo - considera ajustar diversidad\n")
                elif unique_archetypes >= 3:
                    f.write("• Buena diversidad de arquetipos en las mejores soluciones\n")
        
        print(f"✓ Reporte comprehensivo guardado en: {report_path}")
    
    def run_complete_analysis(self):
        """Ejecuta análisis completo"""
        print("Iniciando análisis avanzado de Scatter Search...")
        print(f"Directorio base: {self.base_dir}")
        
        print("\n1. Cargando historial de checkpoints...")
        checkpoints = self.load_checkpoint_history()
        
        if not checkpoints:
            print(" No se encontraron checkpoints para analizar")
            return
        
        print("2. Cargando resumen de optimización...")
        optimization_summary = self.load_optimization_summary()
        
        print("3. Extrayendo datos históricos...")
        historical_df = self.extract_historical_data(checkpoints)
        convergence_df = self.extract_convergence_data(checkpoints)
        
        print(f"   - Datos históricos: {len(historical_df)} registros")
        print(f"   - Datos de convergencia: {len(convergence_df)} puntos")
        
        print("\n4. Generando visualizaciones avanzadas...")
        
        self.plot_convergence_analysis(convergence_df)
        self.plot_archetype_evolution(historical_df)
        self.plot_hyperparameter_evolution(historical_df)
        self.plot_diversity_vs_performance(convergence_df)
        
        if optimization_summary:
            self.plot_final_solutions_analysis(optimization_summary)
        
        print("5. Generando reporte comprehensivo...")
        self.generate_comprehensive_report(checkpoints, convergence_df, historical_df, optimization_summary)
        
        print("\n" + "="*60)
        print("ANÁLISIS COMPLETADO")
        print("="*60)
        print(f" Archivos generados en: {self.analysis_dir}")
        print(" Visualizaciones:")
        print("   • convergence_analysis.png - Análisis completo de convergencia")
        print("   • archetype_evolution.png - Evolución de arquetipos en el tiempo")
        print("   • hyperparameter_evolution.png - Evolución de hiperparámetros clave")
        print("   • diversity_vs_performance.png - Relación diversidad vs performance")
        if optimization_summary:
            print("   • final_solutions_analysis.png - Análisis de soluciones finales")
        print(" Reportes:")
        print("   • comprehensive_analysis_report.txt - Reporte detallado completo")
        
        if not convergence_df.empty:
            total_time = convergence_df['time_hours'].max()
            total_iterations = convergence_df['iteration'].max()
            best_fitness = convergence_df['best_fitness'].max()
            
            print(f"\nResumen ejecutivo:")
            print(f"   • Tiempo total: {total_time:.2f} horas")
            print(f"   • Iteraciones: {total_iterations}")
            print(f"   • Mejor fitness: {best_fitness:.2f}")
            
            if len(convergence_df) > 1:
                improvement = convergence_df['best_fitness'].iloc[-1] - convergence_df['best_fitness'].iloc[0]
                print(f"   • Mejora total: {improvement:.2f}")


def main():
    """Función principal para ejecutar análisis desde línea de comandos"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Análisis avanzado de resultados Scatter Search")
    parser.add_argument("--base_dir", type=str, default="results/scatter_search",
                       help="Directorio base con resultados de Scatter Search")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.base_dir):
        print(f"❌ Error: Directorio no encontrado: {args.base_dir}")
        return
    
    analyzer = AdvancedScatterSearchAnalyzer(args.base_dir)
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()