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
    """
    Advanced analysis of Scatter Search optimization results.

    This class provides functionalities to load, process, and visualize
    the historical data generated during a Scatter Search optimization
    process, including convergence, archetype evolution, and hyperparameter
    trends.
    """

    def __init__(self, base_dir: str):
        """
        Initializes the analyzer with the base directory of the optimization results.

        Args:
            base_dir (str): The base directory where Scatter Search results are stored.
                            This directory should contain 'checkpoints' and 'optimization_summary.json'.
        """
        self.base_dir = base_dir
        self.checkpoints_dir = os.path.join(base_dir, "checkpoints")
        self.analysis_dir = os.path.join(base_dir, "analysis")
        self.optimization_summary_path = os.path.join(base_dir, "optimization_summary.json")

        os.makedirs(self.analysis_dir, exist_ok=True)

        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def load_checkpoint_history(self) -> List[Dict]:
        """
        Loads all checkpoint history files, sorted by iteration number.

        Returns:
            List[Dict]: A list of dictionaries, where each dictionary represents a checkpoint.
                        Returns an empty list if no checkpoints are found or an error occurs.
        """
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
                print(f"Error loading {file_path}: {e}")

        checkpoints.sort(key=lambda x: x['checkpoint_iteration'])
        if checkpoints:
            print(f"Loaded {len(checkpoints)} checkpoints (iterations {checkpoints[0]['checkpoint_iteration']}-{checkpoints[-1]['checkpoint_iteration']})")
        else:
            print("No checkpoints found.")
        return checkpoints

    def load_optimization_summary(self) -> Optional[Dict]:
        """
        Loads the final optimization summary file.

        Returns:
            Optional[Dict]: A dictionary containing the optimization summary, or None if the file
                            is not found or an error occurs during loading.
        """
        try:
            with open(self.optimization_summary_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Summary file not found: {self.optimization_summary_path}")
            return None
        except Exception as e:
            print(f"Error loading summary: {e}")
            return None

    def extract_historical_data(self, checkpoints: List[Dict]) -> pd.DataFrame:
        """
        Extracts historical data for solutions (reference set and best solutions) from checkpoints.

        Args:
            checkpoints (List[Dict]): A list of checkpoint dictionaries.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the extracted historical data,
                          including solution parameters and archetypes.
        """
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
        """
        Extracts convergence-related data from the iteration history within checkpoints.

        Args:
            checkpoints (List[Dict]): A list of checkpoint dictionaries.

        Returns:
            pd.DataFrame: A pandas DataFrame containing data points for convergence analysis,
                          including iteration number, time, best fitness, iteration duration,
                          and diversity score.
        """
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
        """
        Classifies a solution into a predefined archetype based on its hyperparameter values.

        Args:
            solution (Dict): A dictionary representing a solution with its hyperparameters.

        Returns:
            str: The classified archetype of the solution.
        """
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
        """
        Generates a comprehensive convergence analysis plot.

        This plot includes the evolution of best fitness, diversity score,
        accumulated time, and iteration duration over iterations.

        Args:
            convergence_df (pd.DataFrame): DataFrame containing convergence data.
        """
        if convergence_df.empty:
            print("No convergence data to plot.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Scatter Search Convergence Analysis', fontsize=16, fontweight='bold')

        axes[0, 0].plot(convergence_df['iteration'], convergence_df['best_fitness'],
                         'b-o', linewidth=2, markersize=4, alpha=0.8)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Best Fitness')
        axes[0, 0].set_title('Evolution of Best Fitness')
        axes[0, 0].grid(True, alpha=0.3)

        if len(convergence_df) > 1:
            z = np.polyfit(convergence_df['iteration'], convergence_df['best_fitness'], 1)
            p = np.poly1d(z)
            axes[0, 0].plot(convergence_df['iteration'], p(convergence_df['iteration']),
                             "r--", alpha=0.8, label=f'Trend (slope={z[0]:.2f})')
            axes[0, 0].legend()

        axes[0, 1].plot(convergence_df['iteration'], convergence_df['diversity_score'],
                         'g-o', linewidth=2, markersize=4, alpha=0.8)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Diversity Score')
        axes[0, 1].set_title('Evolution of Diversity')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(convergence_df['iteration'], convergence_df['time_hours'],
                         'purple', linewidth=2, alpha=0.8)
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Accumulated Time (hours)')
        axes[1, 0].set_title('Temporal Progress')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].bar(convergence_df['iteration'], convergence_df['iteration_duration_min'],
                       alpha=0.7, color='orange')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Iteration Duration (min)')
        axes[1, 1].set_title('Time per Iteration')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, 'convergence_analysis.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        print("✓ Convergence plot saved.")

    def plot_archetype_evolution(self, historical_df: pd.DataFrame):
        """
        Generates a plot showing the evolution of solution archetypes in the reference set over time.

        Args:
            historical_df (pd.DataFrame): DataFrame containing historical solution data.
        """
        if historical_df.empty:
            print("No historical data to plot archetype evolution.")
            return

        refset_df = historical_df[historical_df['solution_type'] == 'reference_set']

        if refset_df.empty:
            print("No RefSet data to plot archetype evolution.")
            return

        archetype_counts = refset_df.groupby(['checkpoint_iteration', 'archetype']).size().unstack(fill_value=0)

        archetype_percentages = archetype_counts.div(archetype_counts.sum(axis=1), axis=0) * 100

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('Evolution of Archetypes in the Reference Set', fontsize=16, fontweight='bold')

        archetype_percentages.plot(kind='area', stacked=True, ax=ax1, alpha=0.8)
        ax1.set_xlabel('Checkpoint Iteration')
        ax1.set_ylabel('Percentage of RefSet (%)')
        ax1.set_title('Percentage Distribution of Archetypes')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        for archetype in archetype_counts.columns:
            ax2.plot(archetype_counts.index, archetype_counts[archetype],
                     marker='o', linewidth=2, label=archetype, alpha=0.8)

        ax2.set_xlabel('Checkpoint Iteration')
        ax2.set_ylabel('Number of Solutions')
        ax2.set_title('Evolution of Number of Solutions per Archetype')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, 'archetype_evolution.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        print("✓ Archetype evolution plot saved.")

    def plot_hyperparameter_evolution(self, historical_df: pd.DataFrame):
        """
        Generates plots showing the evolution of key hyperparameters for the best solution over time.

        Args:
            historical_df (pd.DataFrame): DataFrame containing historical solution data.
        """
        if historical_df.empty:
            print("No historical data to plot hyperparameter evolution.")
            return

        best_df = historical_df[historical_df['solution_type'] == 'best_solutions']
        best_df = best_df[best_df['solution_rank'] == 1]

        if best_df.empty:
            print("No best solutions data to plot hyperparameter evolution.")
            return

        key_params = ['learning_rate', 'gamma', 'energy_satisfaction_weight',
                      'energy_cost_weight', 'penalty_skipped_vehicle', 'reward_assigned_vehicle']

        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        fig.suptitle('Evolution of Hyperparameters in the Best Solution', fontsize=16, fontweight='bold')

        axes = axes.flatten()

        for i, param in enumerate(key_params):
            if param in best_df.columns:
                param_data = best_df[['checkpoint_iteration', param]].dropna()

                if not param_data.empty:
                    axes[i].plot(param_data['checkpoint_iteration'], param_data[param],
                                 'o-', linewidth=2, markersize=6, alpha=0.8)
                    axes[i].set_xlabel('Checkpoint Iteration')
                    axes[i].set_ylabel(param.replace('_', ' ').title())
                    axes[i].set_title(f'Evolution of {param.replace("_", " ").title()}')
                    axes[i].grid(True, alpha=0.3)

                    for j, row in param_data.iterrows():
                        if j % max(1, len(param_data) // 5) == 0:
                            axes[i].annotate(f'{row[param]:.3f}',
                                             (row['checkpoint_iteration'], row[param]),
                                             xytext=(5, 5), textcoords='offset points',
                                             fontsize=8, alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, 'hyperparameter_evolution.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        print("✓ Hyperparameter evolution plot saved.")

    def plot_diversity_vs_performance(self, convergence_df: pd.DataFrame):
        """
        Generates plots to analyze the relationship between diversity and performance (fitness).

        Args:
            convergence_df (pd.DataFrame): DataFrame containing convergence data.
        """
        if convergence_df.empty:
            print("No convergence data to plot diversity vs performance.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Diversity vs Performance Analysis', fontsize=16, fontweight='bold')

        scatter = ax1.scatter(convergence_df['diversity_score'], convergence_df['best_fitness'],
                              c=convergence_df['iteration'], cmap='viridis', alpha=0.7, s=60)
        ax1.set_xlabel('Diversity Score')
        ax1.set_ylabel('Best Fitness')
        ax1.set_title('Diversity vs Fitness')
        ax1.grid(True, alpha=0.3)

        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Iteration')

        if len(convergence_df) > 1:
            z = np.polyfit(convergence_df['diversity_score'], convergence_df['best_fitness'], 1)
            p = np.poly1d(z)
            ax1.plot(convergence_df['diversity_score'], p(convergence_df['diversity_score']),
                     "r--", alpha=0.8, label=f'Trend (r={np.corrcoef(convergence_df["diversity_score"], convergence_df["best_fitness"])[0,1]:.3f})')
            ax1.legend()

        convergence_df['diversity_bin'] = pd.cut(convergence_df['diversity_score'],
                                                bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

        convergence_df.boxplot(column='best_fitness', by='diversity_bin', ax=ax2)
        ax2.set_xlabel('Diversity Level')
        ax2.set_ylabel('Best Fitness')
        ax2.set_title('Fitness Distribution by Diversity Level')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, 'diversity_vs_performance.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        print("✓ Diversity vs performance plot saved.")

    def plot_final_solutions_analysis(self, optimization_summary: Dict):
        """
        Generates plots to analyze the final best solutions from the optimization summary.

        This includes distribution of fitness, archetype distribution, and a radar chart
        of hyperparameters for the top solution.

        Args:
            optimization_summary (Dict): Dictionary containing the final optimization summary.
        """
        if not optimization_summary or 'best_solutions' not in optimization_summary:
            print("No final solution data to plot.")
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
        fig.suptitle('Analysis of Final Best Solutions', fontsize=16, fontweight='bold')

        axes[0, 0].bar(range(len(fitness_data)), fitness_data,
                        color=sns.color_palette("viridis", len(fitness_data)))
        axes[0, 0].set_xlabel('Solution Rank')
        axes[0, 0].set_ylabel('Fitness')
        axes[0, 0].set_title('Fitness Distribution in Top Solutions')
        axes[0, 0].grid(True, alpha=0.3)

        for i, v in enumerate(fitness_data):
            axes[0, 0].text(i, v + max(fitness_data) * 0.01, f'{v:.0f}',
                            ha='center', va='bottom', fontsize=10)

        archetype_counts = pd.Series(archetype_data).value_counts()
        axes[0, 1].pie(archetype_counts.values, labels=archetype_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Archetype Distribution in Top Solutions')

        if len(hp_df) > 1:
            scatter = axes[1, 0].scatter(hp_df['energy_cost_weight'], hp_df['energy_satisfaction_weight'],
                                         c=hp_df['fitness'], cmap='viridis', s=100, alpha=0.8)
            axes[1, 0].set_xlabel('Energy Cost Weight')
            axes[1, 0].set_ylabel('Energy Satisfaction Weight')
            axes[1, 0].set_title('Cost vs Satisfaction Trade-off')
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
        axes[1, 1].set_title('Best Solution Profile')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, 'final_solutions_analysis.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        print("✓ Final solutions analysis plot saved.")

    def generate_comprehensive_report(self, checkpoints: List[Dict],
                                      convergence_df: pd.DataFrame,
                                      historical_df: pd.DataFrame,
                                      optimization_summary: Dict):
        """
        Generates a comprehensive text report summarizing the analysis findings.

        Args:
            checkpoints (List[Dict]): List of checkpoint dictionaries.
            convergence_df (pd.DataFrame): DataFrame containing convergence data.
            historical_df (pd.DataFrame): DataFrame containing historical solution data.
            optimization_summary (Dict): Dictionary containing the final optimization summary.
        """

        report_path = os.path.join(self.analysis_dir, 'comprehensive_analysis_report.txt')

        with open(report_path, 'w') as f:
            f.write("SCATTER SEARCH COMPREHENSIVE ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("1. GENERAL INFORMATION:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Checkpoints analyzed: {len(checkpoints)}\n")
            if not convergence_df.empty:
                f.write(f"Iterations completed: {convergence_df['iteration'].max()}\n")
                f.write(f"Total time: {convergence_df['time_hours'].max():.2f} hours\n")
            f.write(f"Historical solutions analyzed: {len(historical_df)}\n\n")

            if not convergence_df.empty:
                f.write("2. CONVERGENCE ANALYSIS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Initial fitness: {convergence_df['best_fitness'].iloc[0]:.2f}\n")
                f.write(f"Final fitness: {convergence_df['best_fitness'].iloc[-1]:.2f}\n")
                improvement = convergence_df['best_fitness'].iloc[-1] - convergence_df['best_fitness'].iloc[0]
                f.write(f"Total improvement: {improvement:.2f}\n")
                f.write(f"Best fitness achieved: {convergence_df['best_fitness'].max():.2f}\n")
                f.write(f"Average diversity: {convergence_df['diversity_score'].mean():.3f}\n\n")

            if not historical_df.empty:
                f.write("3. ARCHETYPE ANALYSIS:\n")
                f.write("-" * 30 + "\n")

                refset_df = historical_df[historical_df['solution_type'] == 'reference_set']
                if not refset_df.empty:
                    archetype_dist = refset_df['archetype'].value_counts()
                    f.write("Overall Distribution in Reference Sets:\n")
                    for archetype, count in archetype_dist.items():
                        percentage = (count / len(refset_df)) * 100
                        f.write(f"   {archetype}: {count} ({percentage:.1f}%)\n")
                f.write("\n")

            if optimization_summary and 'best_solutions' in optimization_summary:
                f.write("4. FINAL SOLUTIONS:\n")
                f.write("-" * 25 + "\n")

                best_solutions = optimization_summary['best_solutions']
                for i, sol in enumerate(best_solutions[:5]):
                    f.write(f"Rank {i+1}: {sol.get('archetype', 'unknown').upper()}\n")
                    f.write(f"   Fitness: {sol.get('fitness', 0):.2f}\n")

                    hp = sol.get('hyperparameters', {})
                    f.write(f"   Learning Rate: {hp.get('learning_rate', 0):.6f}\n")
                    f.write(f"   Gamma: {hp.get('gamma', 0):.6f}\n")
                    f.write(f"   Satisfaction Weight: {hp.get('energy_satisfaction_weight', 0):.2f}\n")
                    f.write(f"   Cost Weight: {hp.get('energy_cost_weight', 0):.6f}\n\n")

            f.write("5. RECOMMENDATIONS:\n")
            f.write("-" * 20 + "\n")

            if not convergence_df.empty:
                final_diversity = convergence_df['diversity_score'].iloc[-1]
                if final_diversity < 0.3:
                    f.write("• Final diversity is low; consider increasing diversity parameters.\n")
                elif final_diversity > 0.4:
                    f.write("• Good diversity maintained during optimization.\n")

                if convergence_df['best_fitness'].iloc[-1] == convergence_df['best_fitness'].max():
                    f.write("• The algorithm converged in the last iterations – time well spent.\n")
                else:
                    f.write("• The best fitness is not at the end – possible early convergence.\n")

            if optimization_summary:
                best_solutions = optimization_summary.get('best_solutions', [])
                archetypes = [sol.get('archetype') for sol in best_solutions]
                unique_archetypes = len(set(archetypes))

                if unique_archetypes == 1:
                    f.write("• All best solutions are of the same archetype – consider adjusting for more diversity.\n")
                elif unique_archetypes >= 3:
                    f.write("• Good diversity of archetypes among the best solutions.\n")

        print(f"✓ Comprehensive report saved to: {report_path}")

    def run_complete_analysis(self):
        """
        Executes the complete Scatter Search analysis workflow.

        This method orchestrates the loading of data, extraction of relevant information,
        generation of all plots, and the final comprehensive report.
        """
        print("Starting advanced Scatter Search analysis...")
        print(f"Base directory: {self.base_dir}")

        print("\n1. Loading checkpoint history...")
        checkpoints = self.load_checkpoint_history()

        if not checkpoints:
            print(" No checkpoints found to analyze.")
            return

        print("2. Loading optimization summary...")
        optimization_summary = self.load_optimization_summary()

        print("3. Extracting historical data...")
        historical_df = self.extract_historical_data(checkpoints)
        convergence_df = self.extract_convergence_data(checkpoints)

        print(f"   - Historical data: {len(historical_df)} records")
        print(f"   - Convergence data: {len(convergence_df)} points")

        print("\n4. Generating advanced visualizations...")

        self.plot_convergence_analysis(convergence_df)
        self.plot_archetype_evolution(historical_df)
        self.plot_hyperparameter_evolution(historical_df)
        self.plot_diversity_vs_performance(convergence_df)

        if optimization_summary:
            self.plot_final_solutions_analysis(optimization_summary)

        print("5. Generating comprehensive report...")
        self.generate_comprehensive_report(checkpoints, convergence_df, historical_df, optimization_summary)

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETED")
        print("=" * 60)
        print(f" Files generated in: {self.analysis_dir}")
        print(" Visualizations:")
        print("    • convergence_analysis.png - Complete convergence analysis")
        print("    • archetype_evolution.png - Evolution of archetypes over time")
        print("    • hyperparameter_evolution.png - Evolution of key hyperparameters")
        print("    • diversity_vs_performance.png - Diversity vs performance relationship")
        if optimization_summary:
            print("    • final_solutions_analysis.png - Analysis of final solutions")
        print(" Reports:")
        print("    • comprehensive_analysis_report.txt - Detailed comprehensive report")

        if not convergence_df.empty:
            total_time = convergence_df['time_hours'].max()
            total_iterations = convergence_df['iteration'].max()
            best_fitness = convergence_df['best_fitness'].max()

            print(f"\nExecutive Summary:")
            print(f"    • Total time: {total_time:.2f} hours")
            print(f"    • Iterations: {total_iterations}")
            print(f"    • Best fitness: {best_fitness:.2f}")

            if len(convergence_df) > 1:
                improvement = convergence_df['best_fitness'].iloc[-1] - convergence_df['best_fitness'].iloc[0]
                print(f"    • Total improvement: {improvement:.2f}")


def main():
    """
    Main function to run the analysis from the command line.

    Parses command-line arguments for the base directory and
    initiates the analysis.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Advanced analysis of Scatter Search results")
    parser.add_argument("--base_dir", type=str, default="results/scatter_search",
                        help="Base directory containing Scatter Search results")

    args = parser.parse_args()

    if not os.path.exists(args.base_dir):
        print(f"Error: Directory not found: {args.base_dir}")
        return

    analyzer = AdvancedScatterSearchAnalyzer(args.base_dir)
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()