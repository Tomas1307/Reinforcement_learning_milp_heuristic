import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from datetime import datetime
import glob
from src.common.config import load_system_config


class ScatterSearchGanttPlotter:
    """
    Generador de gráficos Gantt para visualizar las soluciones de Scatter Search
    Muestra distribución de vehículos por hora arriba y Gantt detallado abajo
    """
    
    def __init__(self, solutions_dir: str, systems_dir: str, output_dir: str = None):
        self.solutions_dir = solutions_dir
        self.systems_dir = systems_dir
        self.output_dir = output_dir or os.path.join(solutions_dir, "gantt_plots")
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"ScatterSearchGanttPlotter initialized")
        print(f"  Solutions: {solutions_dir}")
        print(f"  Systems: {systems_dir}")
        print(f"  Output: {self.output_dir}")
    
    def discover_solution_files(self):
        """Descubre todos los archivos config_*.json disponibles"""
        pattern = os.path.join(self.solutions_dir, "config_*.json")
        files = glob.glob(pattern)
        
        solution_info = []
        for file_path in files:
            filename = os.path.basename(file_path)
            # Extraer número del sistema de config_X.json
            import re
            match = re.match(r'config_(\d+)\.json', filename)
            if match:
                system_id = int(match.group(1))
                solution_info.append({
                    'system_id': system_id,
                    'filename': filename,
                    'path': file_path
                })
        
        solution_info.sort(key=lambda x: x['system_id'])
        
        print(f"Found {len(solution_info)} solution files:")
        for info in solution_info:
            print(f"  - System {info['system_id']}: {info['filename']}")
        
        return solution_info
    
    def load_solution_and_system(self, solution_info):
        """Carga tanto la solución como la configuración del sistema"""
        
        # Cargar solución
        with open(solution_info['path'], 'r') as f:
            solution_data = json.load(f)
        
        # Cargar configuración del sistema
        system_config_path = os.path.join(self.systems_dir, f"test_system_{solution_info['system_id']}.json")
        system_config = load_system_config(system_config_path)
        
        return solution_data, system_config
    
    def extract_schedule_from_solution(self, solution_data):
        """Extrae el schedule de los datos de solución"""
        return solution_data.get('schedule_detail', [])
    
    def create_gantt_plot(self, solution_info, figsize=(16, 12)):
        """
        Crea un gráfico Gantt completo con:
        - Subplot superior: Distribución de vehículos por hora
        - Subplot inferior: Gantt detallado de asignaciones
        """
        
        print(f"\nCreating Gantt plot for System {solution_info['system_id']}...")
        
        # Cargar datos
        solution_data, system_config = self.load_solution_and_system(solution_info)
        schedule = self.extract_schedule_from_solution(solution_data)
        
        if not schedule:
            print(f"No schedule data found for System {solution_info['system_id']}")
            return None
        
        # Crear figura con 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[1, 3])
        fig.suptitle(f"System {solution_info['system_id']} - Vehicle Assignment Analysis\n"
                     f"Model: {solution_data.get('model_info', {}).get('model_name', 'Unknown')}", 
                     fontsize=16, fontweight='bold')
        
        # Datos básicos del sistema
        times = system_config['times']
        dt = system_config.get('dt', 0.25)
        arrivals = system_config['arrivals']
        
        # SUBPLOT 1: Distribución de vehículos por hora
        self._plot_vehicle_distribution(ax1, arrivals, times, schedule)
        
        # SUBPLOT 2: Gantt detallado
        self._plot_detailed_gantt(ax2, schedule, system_config, solution_data)
        
        # Ajustar layout
        plt.tight_layout()
        
        # Guardar
        output_filename = f"gantt_system_{solution_info['system_id']}.png"
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Gantt plot saved: {output_filename}")
        return output_path
    
    def _plot_vehicle_distribution(self, ax, arrivals, times, schedule):
        """Subplot superior: Distribución de vehículos por hora (CORREGIDO)"""
        
        # Contar vehículos presentes y asignados por timestep
        vehicles_present = []
        vehicles_assigned = []
        
        print(f"   📊 Calculando distribución para {len(times)} timesteps...")
        
        for i, current_time in enumerate(times):
            # CORRECCIÓN: Contar vehículos presentes en este momento específico
            present_count = 0
            for arr in arrivals:
                # Un vehículo está presente si: arrival_time <= current_time < departure_time
                if arr['arrival_time'] <= current_time < arr['departure_time']:
                    present_count += 1
            
            # Contar vehículos asignados en este timestep específico
            assigned_count = len([entry for entry in schedule if entry[1] == i])
            
            vehicles_present.append(present_count)
            vehicles_assigned.append(assigned_count)
            
            # Debug para algunos timesteps clave
            if i < 5 or i % 10 == 0:
                print(f"     t={current_time:.2f}h (idx={i}): {present_count} presentes, {assigned_count} asignados")
        
        # Verificación final
        max_present = max(vehicles_present)
        total_vehicles = len(arrivals)
        print(f"   📊 Máximo vehículos presentes simultáneos: {max_present}/{total_vehicles}")
        
        # Gráfico de barras
        x = np.arange(len(times))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, vehicles_present, width, label='Vehicles Present', 
                      color='lightblue', alpha=0.7)
        bars2 = ax.bar(x + width/2, vehicles_assigned, width, label='Vehicles Assigned', 
                      color='darkgreen', alpha=0.8)
        
        # Configuración del subplot
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Number of Vehicles')
        ax.set_title('Vehicle Presence vs Assignment Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Configurar ticks del eje X (mostrar cada hora o cada 4 timesteps)
        step = max(1, len(times) // 20)  # Mostrar máximo 20 ticks
        ax.set_xticks(x[::step])
        ax.set_xticklabels([f"{times[i]:.1f}" for i in range(0, len(times), step)])
        
        # Agregar números en las barras (solo algunas para no saturar)
        # Mostrar números solo en timesteps importantes (cada 5to timestep)
        for i in range(0, len(bars1), max(1, len(bars1) // 15)):
            if i < len(bars1) and vehicles_present[i] > 0:
                ax.text(bars1[i].get_x() + bars1[i].get_width()/2, bars1[i].get_height() + 0.3,
                       f'{vehicles_present[i]}', ha='center', va='bottom', fontsize=8, color='blue')
        
        for i in range(0, len(bars2), max(1, len(bars2) // 15)):
            if i < len(bars2) and vehicles_assigned[i] > 0:
                ax.text(bars2[i].get_x() + bars2[i].get_width()/2, bars2[i].get_height() + 0.3,
                       f'{vehicles_assigned[i]}', ha='center', va='bottom', fontsize=8, color='darkgreen')
        
        # Agregar información de estadísticas en el gráfico
        stats_text = f"Max Simultaneous: {max_present} vehicles\n"
        stats_text += f"Peak Assignment: {max(vehicles_assigned) if vehicles_assigned else 0} per timestep\n"
        stats_text += f"Total Vehicles: {total_vehicles}"
        
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    def _plot_detailed_gantt(self, ax, schedule, system_config, solution_data):
        """Subplot inferior: Gantt detallado por slots (Eje Y = Slots, Eje X = Tiempo)"""
        
        times = system_config['times']
        dt = system_config.get('dt', 0.25)
        arrivals = system_config['arrivals']
        chargers = system_config.get('chargers', [])
        n_spots = system_config.get('n_spots', 10)
        
        # Crear mapeo de slots y sus características
        slot_info = {}
        slot_labels = []
        
        # Identificar qué slots tienen cargadores
        charger_slots = set()
        charger_map = {}  # slot -> charger_info
        
        for i, charger in enumerate(chargers):
            # Asumir que cada cargador ocupa un slot (puede ajustarse según tu lógica)
            slot_id = i if i < n_spots else None
            if slot_id is not None:
                charger_slots.add(slot_id)
                charger_map[slot_id] = {
                    'charger_id': charger.get('charger_id', i),
                    'power': charger.get('power', 7)
                }
        
        # Crear labels para todos los slots
        for slot_id in range(n_spots):
            if slot_id in charger_slots:
                charger_info = charger_map[slot_id]
                label = f"Slot_{slot_id}_Charger_{charger_info['charger_id']}"
                slot_info[slot_id] = {'type': 'charging', 'label': label, 'charger_info': charger_info}
            else:
                label = f"Slot_{slot_id}_Parking"
                slot_info[slot_id] = {'type': 'parking', 'label': label, 'charger_info': None}
            
            slot_labels.append(slot_info[slot_id]['label'])
        
        # Organizar schedule por slot y tiempo
        slot_assignments = {}
        for slot_id in range(n_spots):
            slot_assignments[slot_id] = []
        
        # Procesar todas las asignaciones
        for entry in schedule:
            ev_id, time_idx, charger_id, slot, power = entry
            
            if slot < n_spots:  # Verificar que el slot esté en rango
                assignment = {
                    'ev_id': ev_id,
                    'time_idx': time_idx,
                    'time_start': times[time_idx],
                    'time_end': times[time_idx] + dt,
                    'charger_id': charger_id,
                    'power': power,
                    'is_charging': charger_id is not None and power > 0
                }
                slot_assignments[slot].append(assignment)
        
        # Colores para diferentes tipos de vehículos/asignaciones
        colors = {
            'charging': 'green',
            'parking': 'orange',
            'high_power': 'darkgreen',
            'low_power': 'lightgreen'
        }
        
        # Dibujar Gantt por slots
        for slot_id in range(n_spots):
            y_pos = slot_id
            
            # Dibujar las asignaciones para este slot
            for assignment in slot_assignments[slot_id]:
                ev_id = assignment['ev_id']
                
                # Obtener info del vehículo para el color
                ev_info = next((arr for arr in arrivals if arr['id'] == ev_id), None)
                
                # Determinar color basado en el tipo de asignación
                if assignment['is_charging']:
                    if assignment['power'] >= 7:  # Alta potencia
                        color = colors['high_power']
                    else:
                        color = colors['low_power']
                else:
                    color = colors['parking']
                
                # Dibujar rectángulo para la asignación
                rect = patches.Rectangle((assignment['time_start'], y_pos - 0.4),
                                       dt, 0.8,
                                       linewidth=1, edgecolor='black',
                                       facecolor=color, alpha=0.8)
                ax.add_patch(rect)
                
                # Agregar texto con información del vehículo
                text = f"EV_{ev_id}"
                if assignment['is_charging']:
                    text += f"\n{assignment['power']:.1f}kW"
                
                # Solo mostrar texto si el bloque es lo suficientemente ancho
                if dt > 0.3:
                    ax.text(assignment['time_start'] + dt/2, y_pos,
                           text, ha='center', va='center',
                           fontsize=7, fontweight='bold', color='white')
                
                # Dibujar ventana de tiempo del vehículo como fondo sutil
                if ev_info:
                    arrival_time = ev_info['arrival_time']
                    departure_time = ev_info['departure_time']
                    
                    # Solo mostrar ventana si se superpone con el tiempo de asignación
                    window_start = max(arrival_time, assignment['time_start'])
                    window_end = min(departure_time, assignment['time_end'])
                    
                    if window_start < window_end:
                        window_rect = patches.Rectangle((arrival_time, y_pos - 0.45),
                                                       departure_time - arrival_time, 0.9,
                                                       linewidth=0.5, edgecolor='blue',
                                                       facecolor='none', alpha=0.3, linestyle='--')
                        ax.add_patch(window_rect)
        
        # Configuración del subplot
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Parking Slots')
        ax.set_title('Slot Assignment Gantt Chart (Slots vs Time)')
        
        # Configurar ejes
        ax.set_ylim(-0.5, n_spots - 0.5)
        ax.set_xlim(min(times), max(times))
        
        # Labels del eje Y (slots)
        ax.set_yticks(range(n_spots))
        ax.set_yticklabels(slot_labels, fontsize=8)
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        # Leyenda
        legend_elements = [
            patches.Patch(color=colors['high_power'], label='High Power Charging (≥7kW)'),
            patches.Patch(color=colors['low_power'], label='Low Power Charging (<7kW)'),
            patches.Patch(color=colors['parking'], label='Parking Only'),
            patches.Rectangle((0,0), 1, 1, linewidth=0.5, edgecolor='blue', 
                            facecolor='none', alpha=0.3, linestyle='--', label='Vehicle Time Window')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        # Agregar métricas como texto
        metrics_text = self._format_metrics_text(solution_data)
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Agregar información de utilización de slots
        slot_utilization_text = self._format_slot_utilization_text(slot_assignments, n_spots)
        ax.text(0.98, 0.98, slot_utilization_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    def _format_metrics_text(self, solution_data):
        """Formatea las métricas para mostrar en el gráfico"""
        
        energy_perf = solution_data.get('energy_performance', {})
        vehicle_perf = solution_data.get('vehicle_performance', {})
        resource_util = solution_data.get('resource_utilization', {})
        
        text = f"Performance Metrics:\n"
        text += f"• Energy Satisfaction: {energy_perf.get('overall_satisfaction_pct', 0):.1f}%\n"
        text += f"• Vehicles Fully Satisfied: {vehicle_perf.get('vehicles_fully_satisfied', 0)}/{vehicle_perf.get('vehicles_total', 0)}\n"
        text += f"• Charger Utilization: {resource_util.get('chargers_utilization_pct', 0):.1f}%\n"
        text += f"• Total Energy: {energy_perf.get('total_energy_delivered_kwh', 0):.1f} kWh"
        
    def _format_slot_utilization_text(self, slot_assignments, n_spots):
        """Formatea información de utilización de slots"""
        
        slots_used = len([slot_id for slot_id, assignments in slot_assignments.items() if assignments])
        utilization_pct = (slots_used / n_spots) * 100 if n_spots > 0 else 0
        
        charging_slots = 0
        parking_slots = 0
        
        for slot_id, assignments in slot_assignments.items():
            if assignments:
                # Verificar si alguna asignación en este slot es de carga
                has_charging = any(a['is_charging'] for a in assignments)
                if has_charging:
                    charging_slots += 1
                else:
                    parking_slots += 1
        
        text = f"Slot Utilization:\n"
        text += f"• Used: {slots_used}/{n_spots} ({utilization_pct:.1f}%)\n"
        text += f"• Charging Slots: {charging_slots}\n"
        text += f"• Parking Slots: {parking_slots}"
        
        return text
    
    def generate_all_gantt_plots(self, max_systems=None):
        """Genera gráficos Gantt para todos los sistemas disponibles"""
        
        print(f"\n{'='*60}")
        print("GENERATING GANTT PLOTS FOR ALL SOLUTIONS")
        print(f"{'='*60}")
        
        solution_files = self.discover_solution_files()
        
        if max_systems:
            solution_files = solution_files[:max_systems]
        
        generated_plots = []
        
        for i, solution_info in enumerate(solution_files, 1):
            print(f"\nProcessing {i}/{len(solution_files)}: System {solution_info['system_id']}")
            
            try:
                plot_path = self.create_gantt_plot(solution_info)
                if plot_path:
                    generated_plots.append(plot_path)
                    
            except Exception as e:
                print(f"❌ Error generating plot for System {solution_info['system_id']}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*60}")
        print(f"GANTT GENERATION COMPLETED")
        print(f"{'='*60}")
        print(f"Generated plots: {len(generated_plots)}")
        print(f"Output directory: {self.output_dir}")
        
        if generated_plots:
            print("\nGenerated files:")
            for plot_path in generated_plots:
                print(f"  ✅ {os.path.basename(plot_path)}")
        
        return generated_plots
    
    def generate_single_gantt(self, system_id):
        """Genera gráfico Gantt para un sistema específico"""
        
        solution_files = self.discover_solution_files()
        target_solution = next((s for s in solution_files if s['system_id'] == system_id), None)
        
        if not target_solution:
            print(f"❌ No solution found for System {system_id}")
            available_systems = [s['system_id'] for s in solution_files]
            print(f"Available systems: {available_systems}")
            return None
        
        print(f"Generating Gantt plot for System {system_id}...")
        return self.create_gantt_plot(target_solution)


def main():
    """Función principal para testing"""
    
    solutions_dir = "results/scatter_search/model_solution/efficiency_focused_rank_1"
    systems_dir = "src/configs/system_data"
    
    if not os.path.exists(solutions_dir):
        print(f"Solutions directory not found: {solutions_dir}")
        return
    
    if not os.path.exists(systems_dir):
        print(f"Systems directory not found: {systems_dir}")
        return
    
    plotter = ScatterSearchGanttPlotter(solutions_dir, systems_dir)
    
    # Opción 1: Generar para todos los sistemas
    print("Generating Gantt plots for all available systems...")
    plotter.generate_all_gantt_plots()
    
    # Opción 2: Generar para un sistema específico
    # plotter.generate_single_gantt(system_id=1)


if __name__ == "__main__":
    main()