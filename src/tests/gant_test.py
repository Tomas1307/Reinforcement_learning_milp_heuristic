import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json
import os
from typing import Dict, List, Tuple

class SimplifiedGanttVisualizer:
    """
    Creates Gantt charts from evaluation results without needing original system configs.
    """
    
    def __init__(self, evaluator_output_dir: str):
        self.output_dir = evaluator_output_dir
        self.plots_dir = os.path.join(evaluator_output_dir, 'gantt_plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
    def create_gantt_from_results(self, max_systems: int = 5):
        """
        Creates Gantt charts using only the evaluation results data.
        """
        print("Creating Gantt charts from evaluation results...")
        
        # Load consolidated results to find best models
        consolidated_path = os.path.join(self.output_dir, 'consolidated_comparison.json')
        if not os.path.exists(consolidated_path):
            print("Error: consolidated_comparison.json not found")
            return
            
        with open(consolidated_path, 'r') as f:
            consolidated = json.load(f)
        
        # Get best models by satisfaction
        best_models = consolidated['performance_ranking']['by_satisfaction'][:3]
        
        success_count = 0
        total_attempts = 0
        
        # Select representative systems
        all_systems = self._get_all_systems_from_results()
        selected_systems = self._select_representative_systems(all_systems, max_systems)
        
        for sys_id in selected_systems:
            for model_info in best_models:
                model_name = model_info['model']
                total_attempts += 1
                try:
                    self._create_schedule_gantt(sys_id, model_name)
                    success_count += 1
                except Exception as e:
                    print(f"Error creating Gantt for System {sys_id}, Model {model_name}: {e}")
        
        print(f"Successfully created {success_count}/{total_attempts} Gantt charts")
        print(f"Gantt charts saved to: {self.plots_dir}")
        
    def _get_all_systems_from_results(self) -> Dict[int, Dict]:
        """Get system information from any model's results."""
        systems_info = {}
        
        # Load from first available model
        model_dirs = [d for d in os.listdir(self.output_dir) 
                     if os.path.isdir(os.path.join(self.output_dir, d)) and 'rank' in d]
        
        if model_dirs:
            first_model_dir = os.path.join(self.output_dir, model_dirs[0])
            results_file = os.path.join(first_model_dir, f"{model_dirs[0]}_results.json")
            
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    model_results = json.load(f)
                    
                for sys_result in model_results['system_results']:
                    if 'error' not in sys_result:
                        sys_id = sys_result['system_id']
                        systems_info[sys_id] = sys_result['system_metrics']
        
        return systems_info
    
    def _select_representative_systems(self, systems_info: Dict, max_systems: int) -> List[int]:
        """Select representative systems of different complexities."""
        if not systems_info:
            return []
            
        # Sort by number of vehicles (complexity)
        sorted_systems = sorted(systems_info.items(), 
                              key=lambda x: x[1]['num_vehicles'])
        
        # Select systems representing different complexity levels
        total_systems = len(sorted_systems)
        if total_systems <= max_systems:
            return [sys_id for sys_id, _ in sorted_systems]
        
        # Select evenly distributed systems
        indices = np.linspace(0, total_systems-1, max_systems, dtype=int)
        return [sorted_systems[i][0] for i in indices]
    
    def _create_schedule_gantt(self, sys_id: int, model_name: str):
        """Create Gantt chart showing only the charging schedule."""
        
        # Load model results
        model_results = self._load_model_results(model_name)
        if not model_results:
            return
            
        # Find system result
        sys_result = None
        for result in model_results['system_results']:
            if result['system_id'] == sys_id and 'error' not in result:
                sys_result = result
                break
                
        if not sys_result:
            return
            
        # Create the schedule visualization
        self._plot_schedule_gantt(sys_id, model_name, sys_result)
    
    def _load_model_results(self, model_name: str) -> Dict:
        """Load results for a specific model."""
        model_dir = os.path.join(self.output_dir, model_name)
        results_file = os.path.join(model_dir, f"{model_name}_results.json")
        
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                return json.load(f)
        return None
    
    def _plot_schedule_gantt(self, sys_id: int, model_name: str, sys_result: Dict):
        """Create the Gantt chart plot showing charging schedule."""
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        # Extract data
        schedule = sys_result['detailed_schedule']
        performance = sys_result['performance']
        system_metrics = sys_result['system_metrics']
        energy_details = sys_result['energy_metrics_detail']
        
        n_spots = system_metrics['num_slots']
        total_vehicles = system_metrics['num_vehicles']
        
        if not schedule:
            ax.text(0.5, 0.5, 'No charging assignments', transform=ax.transAxes,
                   ha='center', va='center', fontsize=16, color='red')
            ax.set_ylim(0, n_spots)
            ax.set_xlim(0, 14)
        else:
            # Plot the charging schedule
            self._plot_charging_blocks(ax, schedule, n_spots, energy_details)
        
        # Add vehicle information text box
        self._add_vehicle_info_box(ax, energy_details, performance)
        
        # Formatting
        max_time_idx = max([entry[1] for entry in schedule]) if schedule else 55
        dt = 0.25  # 15-minute intervals
        max_time = max_time_idx * dt
        
        ax.set_xlim(0, max_time)
        ax.set_ylim(-0.5, n_spots - 0.5)
        
        ax.set_title(f'System {sys_id} - Charging Schedule: {model_name}\n'
                    f'Satisfaction: {performance["energy_satisfaction_pct"]:.1f}%, '
                    f'Assigned: {performance["vehicles_assigned"]}/{total_vehicles} vehicles, '
                    f'Cost: ${performance["total_cost_dollars"]:.2f}', 
                    fontsize=14, fontweight='bold')
        
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Parking Slot')
        ax.set_yticks(range(n_spots))
        ax.set_yticklabels([f'Slot {i}' for i in range(n_spots)])
        ax.grid(True, alpha=0.3)
        
        # Add time markers
        time_markers = np.arange(0, max_time + 1, 2)  # Every 2 hours
        ax.set_xticks(time_markers)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"schedule_gantt_system_{sys_id}_{model_name.replace('_', '-')}.png"
        plot_path = os.path.join(self.plots_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created schedule Gantt: {plot_filename}")
    
    def _plot_charging_blocks(self, ax, schedule: List, n_spots: int, energy_details: Dict):
        """Plot the charging schedule as colored blocks."""
        
        dt = 0.25  # 15-minute intervals
        
        # Create color map for vehicles
        vehicle_ids = sorted(set(entry[0] for entry in schedule))
        colors = plt.cm.Set3(np.linspace(0, 1, len(vehicle_ids)))
        vehicle_colors = {vehicle_ids[i]: colors[i] for i in range(len(vehicle_ids))}
        
        # Track vehicle assignments for legend
        legend_info = {}
        
        for ev_id, time_idx, charger_id, slot, power in schedule:
            start_time = time_idx * dt
            end_time = start_time + dt
            
            color = vehicle_colors[ev_id]
            
            # Draw charging block
            rect = patches.Rectangle((start_time, slot - 0.4), dt, 0.8,
                                   facecolor=color, edgecolor='black', linewidth=0.5,
                                   alpha=0.8)
            ax.add_patch(rect)
            
            # Add vehicle ID and power text
            ax.text(start_time + dt/2, slot, f'V{ev_id}\n{power:.1f}kW',
                   ha='center', va='center', fontsize=6, fontweight='bold')
            
            # Store info for legend
            if ev_id not in legend_info:
                vehicle_energy = energy_details.get(str(ev_id), {})
                required = vehicle_energy.get('required_energy', 0)
                delivered = vehicle_energy.get('delivered_energy', 0)
                satisfaction = vehicle_energy.get('satisfaction', 0)
                
                legend_info[ev_id] = {
                    'color': color,
                    'required': required,
                    'delivered': delivered,
                    'satisfaction': satisfaction
                }
        
        # Create legend
        legend_elements = []
        for ev_id in sorted(legend_info.keys())[:8]:  # Limit to 8 vehicles
            info = legend_info[ev_id]
            label = f'V{ev_id}: {info["satisfaction"]:.1%} ({info["delivered"]:.1f}/{info["required"]:.1f} kWh)'
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=info['color'], 
                                               alpha=0.8, label=label))
        
        if len(legend_info) > 8:
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='gray', alpha=0.5,
                                               label=f'... +{len(legend_info)-8} more vehicles'))
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8, ncol=1)
    
    def _add_vehicle_info_box(self, ax, energy_details: Dict, performance: Dict):
        """Add information box with vehicle statistics."""
        
        total_vehicles = len(energy_details)
        assigned_vehicles = performance['vehicles_assigned']
        unassigned = total_vehicles - assigned_vehicles
        
        # Calculate satisfaction categories
        fully_satisfied = sum(1 for v in energy_details.values() if v.get('satisfaction', 0) >= 0.95)
        partially_satisfied = sum(1 for v in energy_details.values() if 0.1 <= v.get('satisfaction', 0) < 0.95)
        unsatisfied = sum(1 for v in energy_details.values() if v.get('satisfaction', 0) < 0.1)
        
        info_text = f"""Vehicle Status:
• Total: {total_vehicles}
• Assigned: {assigned_vehicles}
• Unassigned: {unassigned}

Satisfaction:
• Fully (≥95%): {fully_satisfied}
• Partial (10-95%): {partially_satisfied}  
• Poor (<10%): {unsatisfied}

Energy: {performance['total_energy_delivered']:.1f}/{performance['total_energy_required']:.1f} kWh
Cost/kWh: ${performance['avg_cost_per_kwh']:.2f}"""
        
        # Add text box
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
               facecolor='lightgray', alpha=0.8))

def create_schedule_gantt_charts(evaluator_output_dir: str, max_systems: int = 5):
    """
    Main function to create schedule Gantt charts.
    
    Args:
        evaluator_output_dir (str): Path to evaluator output directory
        max_systems (int): Maximum number of systems to visualize
    """
    visualizer = SimplifiedGanttVisualizer(evaluator_output_dir)
    visualizer.create_gantt_from_results(max_systems)

# Example usage
if __name__ == "__main__":
    # Update this path to your evaluator output directory
    output_dir = "results/scatter_search/model_solutions"
    create_schedule_gantt_charts(output_dir, max_systems=5)
