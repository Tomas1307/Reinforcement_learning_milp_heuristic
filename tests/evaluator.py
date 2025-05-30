import sys
from src.dqn_agent.agent import EnhancedDQNAgentPyTorch
from src.dqn_agent.environment import EVChargingEnv
from src.common.config import load_system_config
import os
import re
from typing import List, Dict, Any
import glob
from time import time
import json
import numpy as np
import torch
from datetime import datetime

class ModelEvaluator:
    """
    Comprehensive evaluator for Scatter Search models.

    This class handles the discovery of trained models and system configurations,
    evaluates each model against all discovered systems, and generates detailed
    and consolidated reports of the evaluation results.
    """
    def __init__(self, models_dir: str, systems_dir: str, output_dir: str):
        """
        Initializes the ModelEvaluator with paths for models, systems, and output.

        Args:
            models_dir (str): Directory containing the trained model files (.pt).
            systems_dir (str): Directory containing the system configuration files (.json).
            output_dir (str): Directory where all evaluation results and reports will be saved.
        """
        self.models_dir = models_dir
        self.systems_dir = systems_dir
        self.output_dir = output_dir
        self.results = {}
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Evaluator initialized")
        print(f"  Models: {models_dir}")
        print(f"  Systems: {systems_dir}")
        print(f"  Output: {output_dir}")

    def discover_models(self) -> List[Dict[str, Any]]:
        """
        Discovers all available model files within the specified models directory.

        Model files are expected to follow the naming convention 'archetype_rank_X.pt'
        or 'archetype_X.pt', from which the archetype and rank are extracted.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                                  contains 'archetype', 'rank', 'filename', and 'path'
                                  for a discovered model. Models are sorted by archetype
                                  and then by rank.
        """
        print("\nDiscovering models...")
        
        model_files = glob.glob(os.path.join(self.models_dir, "*.pt"))
        models = []
        
        for model_path in model_files:
            filename = os.path.basename(model_path)
            
            # Extract archetype and rank from filename
            # Expected formats: archetype_rank_X.pt or archetype_X.pt
            match = re.match(r'(.+)_rank_(\d+)\.pt$', filename)
            if match:
                archetype = match.group(1)
                rank = int(match.group(2))
            else:
                # Alternative format: archetype_X.pt
                match = re.match(r'(.+)_(\d+)\.pt$', filename)
                if match:
                    archetype = match.group(1)
                    rank = int(match.group(2))
                else:
                    print(f"Could not parse: {filename}")
                    continue
            
            models.append({
                'archetype': archetype,
                'rank': rank,
                'filename': filename,
                'path': model_path
            })
        
        models.sort(key=lambda x: (x['archetype'], x['rank']))
        
        print(f"Found {len(models)} models:")
        for model in models:
            print(f"  - {model['archetype']} rank {model['rank']}")
        
        return models

    def _calculate_theoretical_capacity(self, system_config: Dict) -> Dict[str, Any]:
        """
        Calculates the theoretical maximum capacity of the system with improved logic.
        
        Args:
            system_config (Dict): System configuration
            
        Returns:
            Dict: Dictionary with theoretical capacity metrics
        """
        try:
            # Basic system parameters
            times = system_config['times']
            dt = system_config.get('dt', 0.25)
            n_spots = system_config['n_spots']
            
            # Handle different charger formats
            if 'chargers' in system_config:
                chargers = system_config['chargers']
            elif 'parking_config' in system_config and 'chargers' in system_config['parking_config']:
                chargers = system_config['parking_config']['chargers']
            else:
                chargers = []
            
            arrivals = system_config['arrivals']
            
            # Get transformer limit
            transformer_limit = None
            if 'parking_config' in system_config:
                transformer_limit = system_config['parking_config'].get('transformer_limit')
            if transformer_limit is None:
                transformer_limit = system_config.get('transformer_limit', float('inf'))
            
            # Calculate energy constraints
            total_charger_power = sum(c.get('power', 0) for c in chargers)
            total_time_hours = len(times) * dt
            theoretical_max_energy = min(total_charger_power, transformer_limit) * total_time_hours
            
            # Calculate total energy demand
            total_energy_demand = sum(arr['required_energy'] for arr in arrivals)
            
            # Energy-limited capacity
            if theoretical_max_energy >= total_energy_demand:
                energy_limited_vehicles = len(arrivals)
            else:
                energy_limited_vehicles = max(1, int((theoretical_max_energy / total_energy_demand) * len(arrivals)))
            
            # Spot-limited capacity (improved calculation)
            spot_limited_vehicles = self._calculate_spot_limited_capacity_improved(system_config)
            
            # Charger-limited capacity
            charger_limited_vehicles = min(len(chargers), len(arrivals)) if chargers else len(arrivals)
            
            # Time-based capacity (considering overlap)
            time_limited_vehicles = self._calculate_time_based_capacity(system_config)
            
            # The real bottleneck is the minimum of all constraints
            theoretical_max_vehicles = min(
                energy_limited_vehicles,
                spot_limited_vehicles, 
                charger_limited_vehicles,
                time_limited_vehicles,
                len(arrivals)
            )
            
            # Identify bottleneck more precisely
            bottleneck = self._identify_bottleneck_improved(
                energy_limited_vehicles, 
                spot_limited_vehicles, 
                charger_limited_vehicles,
                time_limited_vehicles,
                len(arrivals)
            )
            
            return {
                'total_vehicles': len(arrivals),
                'theoretical_max_vehicles': theoretical_max_vehicles,
                'energy_limited': energy_limited_vehicles,
                'spot_limited': spot_limited_vehicles,
                'charger_limited': charger_limited_vehicles,
                'time_limited': time_limited_vehicles,
                'total_energy_demand': total_energy_demand,
                'theoretical_max_energy': theoretical_max_energy,
                'transformer_limit': transformer_limit,
                'bottleneck': bottleneck
            }
            
        except Exception as e:
            print(f"Error calculating theoretical capacity: {e}")
            return {
                'total_vehicles': len(system_config.get('arrivals', [])),
                'theoretical_max_vehicles': min(len(system_config.get('arrivals', [])), 
                                            system_config.get('n_spots', len(system_config.get('arrivals', [])))),
                'bottleneck': 'calculation_error'
            }
            
    def _calculate_spot_limited_capacity_improved(self, system_config: Dict) -> int:
        """
        Improved calculation of spot-limited capacity considering realistic overlap.
        """
        try:
            times = system_config['times']
            n_spots = system_config['n_spots']
            arrivals = system_config['arrivals']
            dt = system_config.get('dt', 0.25)
            
            # Calculate vehicle presence over time
            occupancy_timeline = []
            
            for time_val in times:
                vehicles_present = 0
                for arr in arrivals:
                    if arr['arrival_time'] <= time_val < arr['departure_time']:
                        vehicles_present += 1
                occupancy_timeline.append(min(vehicles_present, n_spots))
            
            if not occupancy_timeline:
                return len(arrivals)
            
            # Calculate average occupancy
            avg_occupancy = sum(occupancy_timeline) / len(occupancy_timeline)
            max_occupancy = max(occupancy_timeline)
            
            # Calculate average stay duration
            avg_stay_duration = sum(arr['departure_time'] - arr['arrival_time'] for arr in arrivals) / len(arrivals)
            total_simulation_time = max(times) - min(times) if times else 14
            
            # Estimate throughput based on turnover
            if avg_stay_duration > 0:
                turnover_rate = total_simulation_time / avg_stay_duration
                theoretical_throughput = int(avg_occupancy * turnover_rate)
            else:
                theoretical_throughput = len(arrivals)
            
            # Be conservative: use the minimum of calculated values
            spot_capacity = min(
                theoretical_throughput,
                max_occupancy + int(avg_occupancy * 0.5),  # Add buffer for turnover
                len(arrivals)
            )
            
            return max(1, spot_capacity)
            
        except Exception as e:
            print(f"Error in spot capacity calculation: {e}")
            return min(system_config.get('n_spots', 10), len(system_config.get('arrivals', [])))

    def _calculate_time_based_capacity(self, system_config: Dict) -> int:
        """
        Calculates capacity based on temporal constraints and charging time requirements.
        """
        try:
            arrivals = system_config['arrivals']
            chargers = system_config.get('chargers', system_config.get('parking_config', {}).get('chargers', []))
            dt = system_config.get('dt', 0.25)
            
            if not chargers:
                return len(arrivals)
            
            # Estimate average charging power
            avg_charger_power = sum(c.get('power', 7) for c in chargers) / len(chargers)
            
            serviceable_count = 0
            
            for arr in arrivals:
                required_energy = arr['required_energy']
                available_time = arr['departure_time'] - arr['arrival_time']
                
                # Minimum time needed to charge with average power
                min_charging_time = required_energy / avg_charger_power if avg_charger_power > 0 else float('inf')
                
                # If there's enough time, vehicle can potentially be served
                if available_time >= min_charging_time:
                    serviceable_count += 1
            
            return serviceable_count
            
        except Exception as e:
            print(f"Error in time-based capacity calculation: {e}")
            return len(system_config.get('arrivals', []))

    def _identify_bottleneck_improved(self, energy_limited: int, spot_limited: int, 
                                    charger_limited: int, time_limited: int, total_vehicles: int) -> str:
        """Improved bottleneck identification."""
        
        # Find the most restrictive constraint
        constraints = {
            'energy': energy_limited,
            'spots': spot_limited,
            'chargers': charger_limited,
            'time': time_limited,
            'demand': total_vehicles
        }
        
        min_capacity = min(constraints.values())
        
        # Identify which constraint(s) are limiting
        limiting_factors = [name for name, value in constraints.items() if value == min_capacity]
        
        if len(limiting_factors) == 1:
            return limiting_factors[0]
        elif 'energy' in limiting_factors:
            return 'energy'
        elif 'spots' in limiting_factors:
            return 'spots'  
        elif 'chargers' in limiting_factors:
            return 'chargers'
        elif 'time' in limiting_factors:
            return 'time'
        else:
            return 'none'
    
    def _calculate_spot_limited_capacity(self, system_config: Dict) -> int:
        """
        Calculates how many vehicles can be served considering spot availability over time.
        """
        try:
            times = system_config['times']
            n_spots = system_config['n_spots']
            arrivals = system_config['arrivals']
            
            # Create a timeline of vehicle presence
            timeline = {}
            for t_idx, time_val in enumerate(times):
                vehicles_present = 0
                for arr in arrivals:
                    if arr['arrival_time'] <= time_val < arr['departure_time']:
                        vehicles_present += 1
                timeline[t_idx] = min(vehicles_present, n_spots)
            
            # The spot-limited capacity is the maximum number of unique vehicles
            # that could theoretically be served if we had perfect scheduling
            max_concurrent = max(timeline.values()) if timeline else 0
            
            # Estimate based on average occupancy and turnover
            total_time = len(times)
            if total_time > 0:
                avg_occupancy = sum(timeline.values()) / total_time
                # Rough estimate: if vehicles stay for half the time on average
                avg_stay_ratio = 0.5  # This could be calculated more precisely
                estimated_throughput = int(avg_occupancy / avg_stay_ratio) if avg_stay_ratio > 0 else len(arrivals)
                return min(estimated_throughput, len(arrivals))
            
            return min(max_concurrent, len(arrivals))
            
        except Exception:
            return len(system_config.get('arrivals', []))
    
    def _identify_bottleneck(self, energy_limited: int, spot_limited: int, charger_limited: int, total_vehicles: int) -> str:
        """Identifies the main bottleneck in the system."""
        min_capacity = min(energy_limited, spot_limited, charger_limited, total_vehicles)
        
        if min_capacity == energy_limited and min_capacity < total_vehicles:
            return 'energy'
        elif min_capacity == spot_limited and min_capacity < total_vehicles:
            return 'spots'
        elif min_capacity == charger_limited and min_capacity < total_vehicles:
            return 'chargers'
        else:
            return 'none'
        
    def discover_systems(self) -> List[Dict[str, Any]]:    
        """
        Discovers all system configuration files within the specified systems directory.

        Each JSON file is loaded as a system configuration, and relevant metrics
        such as number of vehicles, slots, chargers, and simulation hours are extracted.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                                  contains 'system_id', 'filename', 'path', 'config',
                                  and extracted system metrics for a discovered system.
                                  Systems are sorted by system_id.
        """
        print("\nDiscovering systems...")
        
        json_files = glob.glob(os.path.join(self.systems_dir, "*.json"))
        systems = []
        
        for json_path in json_files:
            try:
                config = load_system_config(json_path)
                
                system_info = {
                    'system_id': config.get('test_number', len(systems) + 1),
                    'filename': os.path.basename(json_path),
                    'path': json_path,
                    'config': config,
                    'num_vehicles': len(config['arrivals']),
                    'num_slots': config['n_spots'],
                    'num_chargers': len(config['chargers']),
                    'simulation_hours': max(config['times']) if config['times'] else 0
                }
                systems.append(system_info)
                
            except Exception as e:
                print(f"Error loading {json_path}: {e}")
        
        systems.sort(key=lambda x: x['system_id'])
        
        print(f"Found {len(systems)} systems:")
        for system in systems:
            print(f"  - System {system['system_id']}: {system['num_vehicles']} vehicles, "
                  f"{system['num_slots']} slots, {system['num_chargers']} chargers")
        
        return systems

    def _detect_model_architecture(self, model_path: str) -> bool:
        """
        Detects whether a model uses Dueling DQN architecture by examining its state dict.
        
        Args:
            model_path (str): Path to the model file
            
        Returns:
            bool: True if Dueling architecture, False if simple DQN
        """
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dict = checkpoint.get('q_network_state_dict', checkpoint)
            
            # Check for Dueling DQN specific layers
            dueling_keys = ['value_stream.weight', 'advantage_stream.weight']
            simple_keys = ['output.weight']
            
            has_dueling = any(key in state_dict for key in dueling_keys)
            has_simple = any(key in state_dict for key in simple_keys)
            
            if has_dueling:
                return True
            elif has_simple:
                return False
            else:
                # Default to simple if unclear
                return False
                
        except Exception as e:
            print(f"Error detecting architecture for {model_path}: {e}")
            return False

    def evaluate_model_on_system(self, model_info: Dict, system_info: Dict) -> Dict[str, Any]:
        """
        Evaluates a specific model on a specific system configuration.

        This involves setting up the EV charging environment, loading the trained
        DQN agent, running a simulation, and calculating key performance metrics
        such as energy satisfaction, total cost, and execution time.

        Args:
            model_info (Dict): A dictionary containing information about the model
                                (archetype, rank, path).
            system_info (Dict): A dictionary containing information about the system
                                (system_id, config, and extracted metrics).

        Returns:
            Dict[str, Any]: A dictionary containing detailed evaluation results for
                            the model on the given system, including performance metrics
                            and a detailed schedule. Returns an error message if the
                            evaluation fails.
        """
        
        model_name = f"{model_info['archetype']}_rank_{model_info['rank']}"
        system_id = system_info['system_id']
        
        print(f"Evaluating {model_name} on System {system_id}...", end=" ")
        
        try:
            start_time = time()
            
            # Create environment
            env = EVChargingEnv(system_info['config'])
            
            # Determine state and action dimensions
            temp_state = env.reset()
            if temp_state is None:
                raise ValueError("Could not get initial state.")
            
            # Use fixed dimensions that match training
            state_size = 40
            action_size = 60
            
            # Detect model architecture and create agent accordingly
            is_dueling = self._detect_model_architecture(model_info['path'])
            
            # Create agent with the correct architecture
            agent = EnhancedDQNAgentPyTorch(
                state_size, 
                action_size,
                dueling_network=is_dueling  # Use detected architecture
            )
            
            model_loaded = agent.load(model_info['path'])
            if not model_loaded:
                raise ValueError(f"Could not load model {model_info['path']}")
            
            # Configure for inference (no exploration)
            original_epsilon = agent.epsilon
            agent.epsilon = 0.0  # No random exploration
            
            # Run simulation
            state = env.reset()
            done = False
            step_count = 0
            
            while not done and step_count < 2000:  # Safety limit
                possible_actions = env._get_possible_actions(state)
                
                if not possible_actions:
                    break
                
                action = agent.act(state, possible_actions)
                if action == -1:
                    break
                
                next_state, reward, done = env.step(action)
                state = next_state
                step_count += 1
            
            # Restore original epsilon
            agent.epsilon = original_epsilon
            
            # Get final metrics
            execution_time = time() - start_time
            schedule = env.get_schedule()
            energy_metrics = env.get_energy_satisfaction_metrics()
            
            # Calculate total cost
            total_cost = 0.0
            for ev_id, time_idx, charger_id, slot, power in schedule:
                if time_idx < len(system_info['config']['prices']):
                    energy_delivered = power * system_info['config'].get('dt', 0.25)
                    cost = energy_delivered * system_info['config']['prices'][time_idx]
                    total_cost += cost
            
            # Get theoretical capacity info (with safety checks)
            theoretical_capacity = system_info.get('theoretical_capacity', {})
            max_serviceable = theoretical_capacity.get('theoretical_max_vehicles', system_info['num_vehicles'])
            bottleneck = theoretical_capacity.get('bottleneck', 'unknown')
            
            # Calculate efficiency ratio
            vehicles_assigned = len(set(entry[0] for entry in schedule))
            efficiency_ratio = (vehicles_assigned / max_serviceable * 100) if max_serviceable > 0 else 0
            
            # Build result
            result = {
                'system_id': system_id,
                'system_metrics': {
                    'num_vehicles': system_info['num_vehicles'],
                    'num_slots': system_info['num_slots'],
                    'num_chargers': system_info['num_chargers'],
                    'simulation_hours': system_info['simulation_hours'],
                    'theoretical_max_vehicles': max_serviceable,
                    'bottleneck': bottleneck
                },
                'performance': {
                    'execution_time_seconds': round(execution_time, 3),
                    'energy_satisfaction_pct': round(energy_metrics['total_satisfaction_pct'], 2),
                    'total_cost_dollars': round(total_cost, 2),
                    'vehicles_assigned': vehicles_assigned,
                    'vehicles_total': system_info['num_vehicles'],
                    'vehicles_max_serviceable': max_serviceable,
                    'assignment_ratio_pct': round(vehicles_assigned / system_info['num_vehicles'] * 100, 1),
                    'efficiency_ratio_pct': round(efficiency_ratio, 1),
                    'total_energy_delivered': round(energy_metrics['total_delivered_energy'], 2),
                    'total_energy_required': round(energy_metrics['total_required_energy'], 2),
                    'avg_cost_per_kwh': round(total_cost / energy_metrics['total_delivered_energy'], 4) if energy_metrics['total_delivered_energy'] > 0 else 0
                },
                'detailed_schedule': [[int(x[0]), int(x[1]), int(x[2]), int(x[3]), float(x[4])] for x in schedule],
                'energy_metrics_detail': energy_metrics.get('ev_metrics', {}),
                'model_architecture': 'Dueling DQN' if is_dueling else 'Simple DQN'
            }
            
            print(f"OK ({execution_time:.1f}s, {energy_metrics['total_satisfaction_pct']:.1f}%)")
            
            # Clear memory
            del agent
            del env
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return result
            
        except Exception as e:
            print(f"Error: {str(e)[:50]}...")
            
            # Safe fallback for error cases
            theoretical_capacity = system_info.get('theoretical_capacity', {})
            max_serviceable = theoretical_capacity.get('theoretical_max_vehicles', system_info['num_vehicles'])
            bottleneck = theoretical_capacity.get('bottleneck', 'unknown')
            
            return {
                'system_id': system_id,
                'error': str(e),
                'system_metrics': {
                    'num_vehicles': system_info['num_vehicles'],
                    'num_slots': system_info['num_slots'],
                    'num_chargers': system_info['num_chargers'],
                    'simulation_hours': system_info['simulation_hours'],
                    'theoretical_max_vehicles': max_serviceable,
                    'bottleneck': bottleneck
                }
            }
            
    def check_model_differentiation(self, all_results: List[Dict]):
        """
        Checks if models are actually producing different results.
        
        Args:
            all_results (List[Dict]): All model evaluation results
        """
        
        differentiation_path = os.path.join(self.output_dir, 'model_differentiation_analysis.txt')
        
        with open(differentiation_path, 'w', encoding='utf-8') as f:
            f.write("MODEL DIFFERENTIATION ANALYSIS\n")
            f.write("=" * 60 + "\n\n")
            
            # Get all systems
            if not all_results or not all_results[0]['system_results']:
                f.write("No results to analyze.\n")
                return
                
            systems_ids = [r['system_id'] for r in all_results[0]['system_results'] if 'error' not in r]
            
            for sys_id in systems_ids:
                f.write(f"SYSTEM {sys_id} ANALYSIS:\n")
                f.write("-" * 30 + "\n")
                
                # Collect results for this system from all models
                system_results = []
                for model_result in all_results:
                    model_info = model_result['model_info']
                    model_name = f"{model_info['archetype']}_rank_{model_info['rank']}"
                    
                    # Find this system's result
                    sys_result = None
                    for sys_res in model_result['system_results']:
                        if sys_res['system_id'] == sys_id and 'error' not in sys_res:
                            sys_result = sys_res
                            break
                    
                    if sys_result:
                        perf = sys_result['performance']
                        system_results.append({
                            'model': model_name,
                            'satisfaction': perf['energy_satisfaction_pct'],
                            'cost': perf['total_cost_dollars'],
                            'assigned': perf['vehicles_assigned'],
                            'energy': perf['total_energy_delivered']
                        })
                
                if not system_results:
                    f.write("  No valid results for this system.\n\n")
                    continue
                
                # Check for identical results
                satisfaction_values = [r['satisfaction'] for r in system_results]
                cost_values = [r['cost'] for r in system_results]
                assigned_values = [r['assigned'] for r in system_results]
                energy_values = [r['energy'] for r in system_results]
                
                # Calculate uniqueness
                unique_satisfactions = len(set(satisfaction_values))
                unique_costs = len(set(cost_values))
                unique_assigned = len(set(assigned_values))
                unique_energies = len(set(energy_values))
                
                total_models = len(system_results)
                
                f.write(f"  Total models evaluated: {total_models}\n")
                f.write(f"  Unique satisfaction values: {unique_satisfactions}/{total_models}\n")
                f.write(f"  Unique cost values: {unique_costs}/{total_models}\n")
                f.write(f"  Unique assignment values: {unique_assigned}/{total_models}\n")
                f.write(f"  Unique energy values: {unique_energies}/{total_models}\n")
                
                # Differentiation score (0 = all identical, 1 = all different)
                avg_uniqueness = (unique_satisfactions + unique_costs + unique_assigned + unique_energies) / (4 * total_models)
                f.write(f"  Differentiation score: {avg_uniqueness:.3f}\n")
                
                if avg_uniqueness < 0.2:
                    f.write("  ⚠️  WARNING: Models are producing very similar results!\n")
                elif avg_uniqueness < 0.5:
                    f.write("  ⚠️  CAUTION: Limited model differentiation detected.\n")
                else:
                    f.write("  ✅ Good model differentiation.\n")
                
                # Show the actual values
                f.write(f"\n  Detailed Results:\n")
                for result in system_results:
                    f.write(f"    {result['model']}: Sat={result['satisfaction']:.1f}%, "
                        f"Cost=${result['cost']:.2f}, Assigned={result['assigned']}\n")
                
                f.write(f"\n")
            
            # Overall analysis
            f.write("OVERALL DIFFERENTIATION SUMMARY:\n")
            f.write("=" * 40 + "\n")
            
            total_comparisons = 0
            problematic_systems = 0
            
            for sys_id in systems_ids:
                system_results = []
                for model_result in all_results:
                    for sys_res in model_result['system_results']:
                        if sys_res['system_id'] == sys_id and 'error' not in sys_res:
                            system_results.append(sys_res['performance'])
                
                if len(system_results) > 1:
                    total_comparisons += 1
                    
                    # Check if all results are identical
                    first_result = system_results[0]
                    all_identical = all(
                        r['energy_satisfaction_pct'] == first_result['energy_satisfaction_pct'] and
                        r['total_cost_dollars'] == first_result['total_cost_dollars'] and
                        r['vehicles_assigned'] == first_result['vehicles_assigned']
                        for r in system_results[1:]
                    )
                    
                    if all_identical:
                        problematic_systems += 1
            
            f.write(f"Systems analyzed: {total_comparisons}\n")
            f.write(f"Systems with identical results: {problematic_systems}\n")
            f.write(f"Problematic ratio: {problematic_systems/total_comparisons*100:.1f}%\n\n")
            
            if problematic_systems > total_comparisons * 0.5:
                f.write("🚨 CRITICAL ISSUE: Most systems show identical results across models!\n")
                f.write("   Possible causes:\n")
                f.write("   - Models not loading correctly\n")
                f.write("   - Models converged to same solution\n")
                f.write("   - Evaluation environment issues\n")
                f.write("   - Same random seed being used\n")
            elif problematic_systems > 0:
                f.write("⚠️  Some systems show identical results. This may indicate:\n")
                f.write("   - Simple systems with obvious optimal solutions\n")
                f.write("   - Models performing similarly on easy cases\n")
            else:
                f.write("✅ Good differentiation across all systems.\n")
        
        print(f"Model differentiation analysis saved to: {differentiation_path}")

    def generate_enhanced_system_analysis(self, all_results: List[Dict]):
        """
        Generates enhanced analysis showing system characteristics vs performance.
        """
        
        analysis_path = os.path.join(self.output_dir, 'enhanced_system_analysis.txt')
        
        with open(analysis_path, 'w', encoding='utf-8') as f:
            f.write("ENHANCED SYSTEM ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            
            # Get system information
            if not all_results:
                return
                
            # Analyze each system's characteristics vs performance
            systems_analysis = []
            
            for model_result in all_results[:1]:  # Just use first model to get system info
                for sys_result in model_result['system_results']:
                    if 'error' in sys_result:
                        continue
                        
                    sys_metrics = sys_result['system_metrics']
                    sys_id = sys_result['system_id']
                    
                    # Calculate performance across all models for this system
                    all_performances = []
                    for model_res in all_results:
                        for sys_res in model_res['system_results']:
                            if sys_res['system_id'] == sys_id and 'error' not in sys_res:
                                all_performances.append(sys_res['performance'])
                    
                    if all_performances:
                        avg_satisfaction = sum(p['energy_satisfaction_pct'] for p in all_performances) / len(all_performances)
                        avg_efficiency = sum(p.get('efficiency_ratio_pct', 0) for p in all_performances) / len(all_performances)
                        avg_assigned = sum(p['vehicles_assigned'] for p in all_performances) / len(all_performances)
                        
                        # Calculate system complexity metrics
                        vehicle_to_slot_ratio = sys_metrics['num_vehicles'] / sys_metrics['num_slots']
                        vehicle_to_charger_ratio = sys_metrics['num_vehicles'] / sys_metrics['num_chargers']
                        charger_to_slot_ratio = sys_metrics['num_chargers'] / sys_metrics['num_slots']
                        
                        systems_analysis.append({
                            'system_id': sys_id,
                            'num_vehicles': sys_metrics['num_vehicles'],
                            'num_slots': sys_metrics['num_slots'],
                            'num_chargers': sys_metrics['num_chargers'],
                            'max_serviceable': sys_metrics.get('theoretical_max_vehicles', sys_metrics['num_vehicles']),
                            'bottleneck': sys_metrics.get('bottleneck', 'unknown'),
                            'vehicle_slot_ratio': vehicle_to_slot_ratio,
                            'vehicle_charger_ratio': vehicle_to_charger_ratio,
                            'charger_slot_ratio': charger_to_slot_ratio,
                            'avg_satisfaction': avg_satisfaction,
                            'avg_efficiency': avg_efficiency,
                            'avg_assigned': avg_assigned
                        })
            
            # Sort by system complexity (vehicle to resource ratio)
            systems_analysis.sort(key=lambda x: x['vehicle_slot_ratio'])
            
            f.write("SYSTEM CHARACTERISTICS vs PERFORMANCE:\n")
            f.write("-" * 50 + "\n\n")
            
            f.write("| Sys | Vehicles | Slots | Chargers | V/S Ratio | Bottleneck | Satisfaction | Efficiency | Assigned |\n")
            f.write("|-----|----------|-------|----------|-----------|------------|--------------|------------|----------|\n")
            
            for sys in systems_analysis:
                f.write(f"| {sys['system_id']:>2} | {sys['num_vehicles']:>7} | {sys['num_slots']:>4} | "
                    f"{sys['num_chargers']:>7} | {sys['vehicle_slot_ratio']:>7.1f} | "
                    f"{sys['bottleneck']:>9} | {sys['avg_satisfaction']:>10.1f}% | "
                    f"{sys['avg_efficiency']:>8.1f}% | {sys['avg_assigned']:>6.1f} |\n")
            
            f.write(f"\n\nSYSTEM CATEGORIZATION:\n")
            f.write("-" * 25 + "\n")
            
            # Categorize systems
            easy_systems = [s for s in systems_analysis if s['vehicle_slot_ratio'] <= 2.0]
            medium_systems = [s for s in systems_analysis if 2.0 < s['vehicle_slot_ratio'] <= 10.0]
            hard_systems = [s for s in systems_analysis if s['vehicle_slot_ratio'] > 10.0]
            
            f.write(f"EASY Systems (V/S ≤ 2.0): {len(easy_systems)} systems\n")
            if easy_systems:
                avg_perf = sum(s['avg_satisfaction'] for s in easy_systems) / len(easy_systems)
                f.write(f"  Average satisfaction: {avg_perf:.1f}%\n")
                f.write(f"  Systems: {', '.join(str(s['system_id']) for s in easy_systems)}\n")
            
            f.write(f"\nMEDIUM Systems (2.0 < V/S ≤ 10.0): {len(medium_systems)} systems\n")
            if medium_systems:
                avg_perf = sum(s['avg_satisfaction'] for s in medium_systems) / len(medium_systems)
                f.write(f"  Average satisfaction: {avg_perf:.1f}%\n")
                f.write(f"  Systems: {', '.join(str(s['system_id']) for s in medium_systems)}\n")
            
            f.write(f"\nHARD Systems (V/S > 10.0): {len(hard_systems)} systems\n")
            if hard_systems:
                avg_perf = sum(s['avg_satisfaction'] for s in hard_systems) / len(hard_systems)
                f.write(f"  Average satisfaction: {avg_perf:.1f}%\n")
                f.write(f"  Systems: {', '.join(str(s['system_id']) for s in hard_systems)}\n")
            
            # Performance insights
            f.write(f"\n\nPERFORMANCE INSIGHTS:\n")
            f.write("-" * 20 + "\n")
            
            if systems_analysis:
                best_system = max(systems_analysis, key=lambda x: x['avg_satisfaction'])
                worst_system = min(systems_analysis, key=lambda x: x['avg_satisfaction'])
                
                f.write(f"Best performing system: {best_system['system_id']} ({best_system['avg_satisfaction']:.1f}% satisfaction)\n")
                f.write(f"  Characteristics: {best_system['num_vehicles']} vehicles, {best_system['num_slots']} slots, "
                    f"V/S ratio = {best_system['vehicle_slot_ratio']:.1f}\n")
                
                f.write(f"\nWorst performing system: {worst_system['system_id']} ({worst_system['avg_satisfaction']:.1f}% satisfaction)\n")
                f.write(f"  Characteristics: {worst_system['num_vehicles']} vehicles, {worst_system['num_slots']} slots, "
                    f"V/S ratio = {worst_system['vehicle_slot_ratio']:.1f}\n")
        
        print(f"Enhanced system analysis saved to: {analysis_path}")

    def evaluate_model(self, model_info: Dict, systems: List[Dict]) -> Dict[str, Any]:
        """
        Evaluates a single model against all provided system configurations.

        Aggregates the results from individual system evaluations and calculates
        summary statistics for the model's overall performance.

        Args:
            model_info (Dict): A dictionary containing information about the model.
            systems (List[Dict]): A list of dictionaries, each containing system information.

        Returns:
            Dict[str, Any]: A dictionary containing the model's information,
                            results for each system it was evaluated on, and
                            a summary of its performance across all systems.
        """
        
        model_name = f"{model_info['archetype']}_rank_{model_info['rank']}"
        print(f"\nEvaluating model: {model_name}")
        print(f"  Archetype: {model_info['archetype']}")
        print(f"  Rank: {model_info['rank']}")
        print(f"  File: {model_info['filename']}")
        
        system_results = []
        valid_results = []
        
        for system_info in systems:
            result = self.evaluate_model_on_system(model_info, system_info)
            system_results.append(result)
            
            # Only include valid results for summary
            if 'error' not in result:
                valid_results.append(result)
        
        # Calculate model summary
        if valid_results:
            model_summary = {
                'avg_satisfaction_pct': round(np.mean([r['performance']['energy_satisfaction_pct'] for r in valid_results]), 2),
                'avg_cost_dollars': round(np.mean([r['performance']['total_cost_dollars'] for r in valid_results]), 2),
                'avg_execution_time_seconds': round(np.mean([r['performance']['execution_time_seconds'] for r in valid_results]), 3),
                'avg_assignment_ratio_pct': round(np.mean([r['performance']['assignment_ratio_pct'] for r in valid_results]), 1),
                'best_system_id': max(valid_results, key=lambda x: x['performance']['energy_satisfaction_pct'])['system_id'],
                'worst_system_id': min(valid_results, key=lambda x: x['performance']['energy_satisfaction_pct'])['system_id'],
                'systems_evaluated': len(valid_results),
                'systems_failed': len(systems) - len(valid_results)
            }
        else:
            model_summary = {
                'avg_satisfaction_pct': 0,
                'avg_cost_dollars': 0,
                'avg_execution_time_seconds': 0,
                'avg_assignment_ratio_pct': 0,
                'systems_evaluated': 0,
                'systems_failed': len(systems)
            }
        
        # Build complete model result
        model_result = {
            'model_info': {
                'archetype': model_info['archetype'],
                'rank': model_info['rank'],
                'model_path': model_info['filename']
            },
            'system_results': system_results,
            'model_summary': model_summary,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return model_result

    def save_model_results(self, model_info: Dict, model_result: Dict):
        """
        Saves the evaluation results for an individual model.

        Creates a dedicated directory for the model's results and saves a JSON
        file containing all detailed results, as well as a human-readable text report.

        Args:
            model_info (Dict): Information about the model being evaluated.
            model_result (Dict): The complete evaluation results for the model.
        """
        
        model_name = f"{model_info['archetype']}_rank_{model_info['rank']}"
        model_dir = os.path.join(self.output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save complete JSON
        json_path = os.path.join(model_dir, f"{model_name}_results.json")
        with open(json_path, 'w') as f:
            json.dump(model_result, f, indent=2)
        
        # Generate text report
        self.generate_model_report(model_info, model_result, model_dir)
        
        print(f"Results saved to: {model_dir}")

    def generate_model_report(self, model_info: Dict, model_result: Dict, output_dir: str):
        """
        Generates a text report summarizing a model's performance.

        The report includes general model information, a performance summary table
        with average, best, and worst metrics, and a detailed breakdown of results
        for each evaluated system.

        Args:
            model_info (Dict): Information about the model.
            model_result (Dict): The complete evaluation results for the model.
            output_dir (str): The directory where the report will be saved.
        """
        
        model_name = f"{model_info['archetype']}_rank_{model_info['rank']}"
        report_path = os.path.join(output_dir, f"{model_name}_summary.txt")
        
        summary = model_result['model_summary']
        system_results = [r for r in model_result['system_results'] if 'error' not in r]
        
        # Use UTF-8 encoding to handle any special characters
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"MODEL EVALUATION REPORT - {model_info['archetype'].upper()} RANK {model_info['rank']}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Evaluation Date: {model_result['timestamp']}\n")
            f.write(f"Model File: {model_info['filename']}\n\n")
            
            f.write("MODEL CHARACTERISTICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Archetype: {model_info['archetype']}\n")
            f.write(f"Rank: {model_info['rank']}\n")
            f.write(f"Approach: {self.get_archetype_description(model_info['archetype'])}\n\n")
            
            f.write("PERFORMANCE SUMMARY:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Successfully evaluated systems: {summary['systems_evaluated']}\n")
            if summary['systems_failed'] > 0:
                f.write(f"Systems with errors: {summary['systems_failed']}\n")
            f.write(f"\n")
            
            # Use ASCII characters instead of Unicode box drawing
            f.write("+---------------------------------------------------------------+\n")
            f.write("| Metric                  | Average    | Best       | Worst     |\n")
            f.write("+-------------------------+------------+------------+-----------+\n")
            
            if system_results:
                satisfactions = [r['performance']['energy_satisfaction_pct'] for r in system_results]
                costs = [r['performance']['total_cost_dollars'] for r in system_results]
                times = [r['performance']['execution_time_seconds'] for r in system_results]
                assignments = [r['performance']['assignment_ratio_pct'] for r in system_results]
                
                f.write(f"| Satisfaction (%)        | {np.mean(satisfactions):8.1f}   | {np.max(satisfactions):8.1f}  | {np.min(satisfactions):8.1f} |\n")
                f.write(f"| Cost ($)                | {np.mean(costs):8.2f}   | {np.min(costs):8.2f}  | {np.max(costs):8.2f} |\n")
                f.write(f"| Execution Time (s)      | {np.mean(times):8.2f}   | {np.min(times):8.2f}  | {np.max(times):8.2f} |\n")
                f.write(f"| Assignment Ratio (%)    | {np.mean(assignments):8.1f}   | {np.max(assignments):8.1f}  | {np.min(assignments):8.1f} |\n")
            
            f.write("+---------------------------------------------------------------+\n\n")
            
            f.write("DETAIL BY SYSTEM:\n")
            f.write("-" * 20 + "\n")
            
            for result in system_results:
                perf = result['performance']
                metrics = result['system_metrics']
                f.write(f"\nSystem {result['system_id']}:\n")
                f.write(f"  Configuration: {metrics['num_vehicles']} vehicles, {metrics['num_slots']} slots, {metrics['num_chargers']} chargers\n")
                f.write(f"  Satisfaction: {perf['energy_satisfaction_pct']:.1f}%\n")
                f.write(f"  Total Cost: ${perf['total_cost_dollars']:.2f}\n")
                f.write(f"  Time: {perf['execution_time_seconds']:.2f}s\n")
                f.write(f"  Assignment: {perf['vehicles_assigned']}/{perf['vehicles_total']} ({perf['assignment_ratio_pct']:.1f}%)\n")
            
            # Errors if any
            error_results = [r for r in model_result['system_results'] if 'error' in r]
            if error_results:
                f.write(f"\nSYSTEMS WITH ERRORS:\n")
                f.write("-" * 20 + "\n")
                for result in error_results:
                    f.write(f"System {result['system_id']}: {result['error']}\n")

    def get_archetype_description(self, archetype: str) -> str:
        """
        Returns a human-readable description for a given model archetype.

        Args:
            archetype (str): The archetype identifier (e.g., 'cost_minimizer').

        Returns:
            str: A descriptive string for the archetype, or a default message
                 if the archetype is not recognized.
        """
        descriptions = {
            'cost_minimizer': 'Minimizes energy costs, accepts lower satisfaction when economically justifiable',
            'satisfaction_maximizer': 'Maximizes customer satisfaction, considers costs as a secondary factor',
            'balanced_optimizer': 'Efficiently balances costs and satisfaction for general use',
            'urgency_focused': 'Never leaves vehicles unattended, ideal for critical services',
            'efficiency_focused': 'Maximizes throughput and assignments, optimizes resource utilization'
        }
        return descriptions.get(archetype, f'Archetype {archetype} optimized by Scatter Search')

    def generate_consolidated_comparison(self, all_results: List[Dict]):
        """
        Generates a consolidated comparison of all evaluated models.

        This includes a summary of all models, performance rankings by various
        metrics (satisfaction, cost, speed, assignment), and a combined table
        of key performance indicators.

        Args:
            all_results (List[Dict]): A list of complete evaluation results for all models.
        """
        
        print(f"\nGenerating consolidated comparison...")
        
        # Prepare data for comparison
        models_summary = []
        for result in all_results:
            if result['model_summary']['systems_evaluated'] > 0:
                model_info = result['model_info']
                summary = result['model_summary']
                
                models_summary.append({
                    'model': f"{model_info['archetype']}_rank_{model_info['rank']}",
                    'archetype': model_info['archetype'],
                    'rank': model_info['rank'],
                    'avg_satisfaction': summary['avg_satisfaction_pct'],
                    'avg_cost': summary['avg_cost_dollars'],
                    'avg_time': summary['avg_execution_time_seconds'],
                    'avg_assignment': summary['avg_assignment_ratio_pct'],
                    'systems_ok': summary['systems_evaluated']
                })
        
        # Create rankings
        by_satisfaction = sorted(models_summary, key=lambda x: x['avg_satisfaction'], reverse=True)
        by_cost = sorted(models_summary, key=lambda x: x['avg_cost'])
        by_speed = sorted(models_summary, key=lambda x: x['avg_time'])
        by_assignment = sorted(models_summary, key=lambda x: x['avg_assignment'], reverse=True)
        
        # Create consolidated comparison
        consolidated = {
            'comparison_summary': {
                'models_evaluated': [m['model'] for m in models_summary],
                'systems_evaluated': len(all_results[0]['system_results']) if all_results else 0,
                'evaluation_date': datetime.now().strftime('%Y-%m-%d'),
                'total_evaluations': len(models_summary) * (len(all_results[0]['system_results']) if all_results else 0)
            },
            'performance_ranking': {
                'by_satisfaction': [{'model': m['model'], 'avg_satisfaction': m['avg_satisfaction']} for m in by_satisfaction],
                'by_cost': [{'model': m['model'], 'avg_cost': m['avg_cost']} for m in by_cost],
                'by_speed': [{'model': m['model'], 'avg_time': m['avg_time']} for m in by_speed],
                'by_assignment': [{'model': m['model'], 'avg_assignment': m['avg_assignment']} for m in by_assignment]
            },
            'performance_table': models_summary,
            'cross_analysis': {
                'best_overall': by_satisfaction[0]['model'] if by_satisfaction else None,
                'most_cost_effective': by_cost[0]['model'] if by_cost else None,
                'fastest': by_speed[0]['model'] if by_speed else None,
                'best_assignment_rate': by_assignment[0]['model'] if by_assignment else None
            }
        }
        
        # Save consolidated comparison
        consolidated_path = os.path.join(self.output_dir, 'consolidated_comparison.json')
        with open(consolidated_path, 'w') as f:
            json.dump(consolidated, f, indent=2)
        
        # Generate summary table
        self.generate_summary_table(models_summary)
        
        print(f"Consolidated comparison saved to: {consolidated_path}")
        
        return consolidated

    def generate_detailed_results_by_instance(self, all_results: List[Dict]):
        """
        Generates detailed results showing each model's performance on each system instance.
        
        Args:
            all_results (List[Dict]): A list of complete evaluation results for all models.
        """
        
        results_path = os.path.join(self.output_dir, 'detailed_results_by_instance.txt')
        
        # Get all systems evaluated
        if not all_results or not all_results[0]['system_results']:
            return
            
        systems_info = {}
        for result in all_results[0]['system_results']:
            if 'error' not in result:
                sys_id = result['system_id']
                systems_info[sys_id] = result['system_metrics']
        
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write("DETAILED RESULTS BY INSTANCE - ALL MODELS\n")
            f.write("=" * 120 + "\n\n")
            
            # Create table for each system
            for sys_id in sorted(systems_info.keys()):
                sys_metrics = systems_info[sys_id]
                f.write(f"\nSYSTEM {sys_id} ({sys_metrics['num_vehicles']} vehicles, {sys_metrics['num_slots']} slots, {sys_metrics['num_chargers']} chargers)")
                if 'theoretical_max_vehicles' in sys_metrics:
                    f.write(f" - Max Serviceable: {sys_metrics['theoretical_max_vehicles']}, Bottleneck: {sys_metrics.get('bottleneck', 'unknown')}")
                f.write(f"\n")
                f.write("-" * 120 + "\n")
                
                # Table header
                f.write("| Model                      | Satisfaction | Cost ($)  | Time (s) | Assigned | MaxServ | Attended% | Efficiency% | Energy Delivered |\n")
                f.write("|----------------------------|-------------|-----------|----------|----------|---------|-----------|-------------|------------------|\n")
                
                # Get results for this system from all models
                system_results = []
                for model_result in all_results:
                    model_info = model_result['model_info']
                    model_name = f"{model_info['archetype']}_rank_{model_info['rank']}"
                    
                    # Find this system's result for this model
                    sys_result = None
                    for sys_res in model_result['system_results']:
                        if sys_res['system_id'] == sys_id and 'error' not in sys_res:
                            sys_result = sys_res
                            break
                    
                    if sys_result:
                        perf = sys_result['performance']
                        system_results.append({
                            'model': model_name,
                            'satisfaction': perf['energy_satisfaction_pct'],
                            'cost': perf['total_cost_dollars'],
                            'time': perf['execution_time_seconds'],
                            'assigned': perf['vehicles_assigned'],
                            'max_serviceable': perf.get('vehicles_max_serviceable', perf['vehicles_total']),
                            'attended_pct': perf['assignment_ratio_pct'],
                            'efficiency_pct': perf.get('efficiency_ratio_pct', 0),
                            'energy': perf['total_energy_delivered']
                        })
                    else:
                        # Model failed on this system
                        max_serviceable = sys_metrics.get('theoretical_max_vehicles', sys_metrics['num_vehicles'])
                        system_results.append({
                            'model': model_name,
                            'satisfaction': 0,
                            'cost': 0,
                            'time': 0,
                            'assigned': 0,
                            'max_serviceable': max_serviceable,
                            'attended_pct': 0,
                            'efficiency_pct': 0,
                            'energy': 0
                        })
                
                # Sort by satisfaction descending
                system_results.sort(key=lambda x: x['satisfaction'], reverse=True)
                
                # Print results for this system
                for res in system_results:
                    f.write(f"| {res['model']:<26} | {res['satisfaction']:>9.1f}% | {res['cost']:>7.2f} | {res['time']:>6.2f} | {res['assigned']:>6} | {res['max_serviceable']:>5} | {res['attended_pct']:>7.1f}% | {res['efficiency_pct']:>9.1f}% | {res['energy']:>14.2f} |\n")
                
                f.write("\n")
        
        print(f"Detailed results by instance saved to: {results_path}")

    def generate_system_comparison_matrix(self, all_results: List[Dict]):
        """
        Generates a matrix showing model performance across all systems.
        
        Args:
            all_results (List[Dict]): A list of complete evaluation results for all models.
        """
        
        matrix_path = os.path.join(self.output_dir, 'system_comparison_matrix.txt')
        
        if not all_results:
            return
            
        # Get model names
        model_names = []
        for result in all_results:
            if result['model_summary']['systems_evaluated'] > 0:
                model_info = result['model_info']
                model_names.append(f"{model_info['archetype']}_rank_{model_info['rank']}")
        
        # Get system IDs
        system_ids = []
        if all_results and all_results[0]['system_results']:
            system_ids = [r['system_id'] for r in all_results[0]['system_results'] if 'error' not in r]
            system_ids.sort()
        
        with open(matrix_path, 'w', encoding='utf-8') as f:
            f.write("SATISFACTION MATRIX - MODELS vs SYSTEMS\n")
            f.write("=" * 120 + "\n\n")
            f.write("Values show Energy Satisfaction % for each Model-System combination\n\n")
            
            # Header
            f.write("| Model                      |")
            for sys_id in system_ids:
                f.write(f" Sys{sys_id:>2} |")
            f.write(" Average |\n")
            
            f.write("|" + "-" * 28 + "|")
            for _ in system_ids:
                f.write("------|")
            f.write("---------|\n")
            
            # Data rows
            for model_result in all_results:
                if model_result['model_summary']['systems_evaluated'] == 0:
                    continue
                    
                model_info = model_result['model_info']
                model_name = f"{model_info['archetype']}_rank_{model_info['rank']}"
                
                f.write(f"| {model_name:<26} |")
                
                satisfactions = []
                for sys_id in system_ids:
                    # Find result for this system
                    satisfaction = 0
                    for sys_result in model_result['system_results']:
                        if sys_result['system_id'] == sys_id and 'error' not in sys_result:
                            satisfaction = sys_result['performance']['energy_satisfaction_pct']
                            break
                    
                    f.write(f" {satisfaction:>4.1f} |")
                    satisfactions.append(satisfaction)
                
                avg_satisfaction = np.mean(satisfactions) if satisfactions else 0
                f.write(f" {avg_satisfaction:>6.1f}% |\n")
            
            f.write("ATTENDED VEHICLES MATRIX - MODELS vs SYSTEMS\n")
            f.write("=" * 120 + "\n\n")
            f.write("Values show % of vehicles attended for each Model-System combination\n\n")
            
            # Header for attended matrix
            f.write("| Model                      |")
            for sys_id in system_ids:
                f.write(f" Sys{sys_id:>2} |")
            f.write(" Average |\n")
            
            f.write("|" + "-" * 28 + "|")
            for _ in system_ids:
                f.write("------|")
            f.write("---------|\n")
            
            # Attended data rows
            for model_result in all_results:
                if model_result['model_summary']['systems_evaluated'] == 0:
                    continue
                    
                model_info = model_result['model_info']
                model_name = f"{model_info['archetype']}_rank_{model_info['rank']}"
                
                f.write(f"| {model_name:<26} |")
                
                attended_ratios = []
                for sys_id in system_ids:
                    # Find result for this system
                    attended_ratio = 0
                    for sys_result in model_result['system_results']:
                        if sys_result['system_id'] == sys_id and 'error' not in sys_result:
                            attended_ratio = sys_result['performance']['assignment_ratio_pct']
                            break
                    
                    f.write(f" {attended_ratio:>4.1f} |")
                    attended_ratios.append(attended_ratio)
                
                avg_attended = np.mean(attended_ratios) if attended_ratios else 0
                f.write(f" {avg_attended:>6.1f}% |\n")
            
            f.write("\n\n")
            f.write("=" * 120 + "\n\n")
            f.write("Values show Total Cost ($) for each Model-System combination\n\n")
            
            # Header for cost matrix
            f.write("| Model                      |")
            for sys_id in system_ids:
                f.write(f" Sys{sys_id:>2} |")
            f.write(" Average |\n")
            
            f.write("|" + "-" * 28 + "|")
            for _ in system_ids:
                f.write("------|")
            f.write("---------|\n")
            
            # Cost data rows
            for model_result in all_results:
                if model_result['model_summary']['systems_evaluated'] == 0:
                    continue
                    
                model_info = model_result['model_info']
                model_name = f"{model_info['archetype']}_rank_{model_info['rank']}"
                
                f.write(f"| {model_name:<26} |")
                
                costs = []
                for sys_id in system_ids:
                    # Find result for this system
                    cost = 0
                    for sys_result in model_result['system_results']:
                        if sys_result['system_id'] == sys_id and 'error' not in sys_result:
                            cost = sys_result['performance']['total_cost_dollars']
                            break
                    
                    if cost >= 1000:
                        f.write(f" {cost/1000:>3.1f}k|")
                    else:
                        f.write(f" {cost:>4.0f} |")
                    costs.append(cost)
                
                avg_cost = np.mean(costs) if costs else 0
                if avg_cost >= 1000:
                    f.write(f" {avg_cost/1000:>5.1f}k$ |\n")
                else:
                    f.write(f" {avg_cost:>6.0f}$ |\n")
        
        print(f"System comparison matrix saved to: {matrix_path}")

    def generate_summary_table(self, models_summary: List[Dict]):
        """
        Generates a text-based summary table of performance across all models.

        This table provides an overview of average performance metrics for each
        model, followed by rankings for satisfaction, cost, and execution time.

        Args:
            models_summary (List[Dict]): A list of dictionaries, each summarizing
                                         the performance of a single model.
        """
        
        table_path = os.path.join(self.output_dir, 'performance_summary_table.txt')
        
        # Use UTF-8 encoding
        with open(table_path, 'w', encoding='utf-8') as f:
            f.write("PERFORMANCE SUMMARY TABLE - ALL MODELS\n")
            f.write("=" * 120 + "\n\n")
            
            # Table header - using ASCII characters
            f.write("+---------------------------------+---------------+-----------+-------+------------+-----------------+--------------+\n")
            f.write("| Model                           | Avg_Time(s)   | Vehicles  | Slots | Chargers   | Satisfaction(%) | Avg_Cost($)  |\n")
            f.write("+---------------------------------+---------------+-----------+-------+------------+-----------------+--------------+\n")
            
            # Table data
            for model in models_summary:
                f.write(f"| {model['model']:<31} | {model['avg_time']:>11.1f}   | {'-':>7}   | {'-':>5} | {'-':>8}   | {model['avg_satisfaction']:>13.1f}   | {model['avg_cost']:>10.2f}   |\n")
            
            f.write("+---------------------------------+---------------+-----------+-------+------------+-----------------+--------------+\n\n")
            
            # Rankings
            f.write("RANKINGS:\n")
            f.write("-" * 10 + "\n\n")
            
            f.write("BEST SATISFACTION:\n")
            sorted_by_sat = sorted(models_summary, key=lambda x: x['avg_satisfaction'], reverse=True)
            for i, model in enumerate(sorted_by_sat[:5], 1):
                f.write(f"  {i}. {model['model']} - {model['avg_satisfaction']:.1f}%\n")
            
            f.write(f"\nBEST COST (lowest):\n")
            sorted_by_cost = sorted(models_summary, key=lambda x: x['avg_cost'])
            for i, model in enumerate(sorted_by_cost[:5], 1):
                f.write(f"  {i}. {model['model']} - ${model['avg_cost']:.2f}\n")
            
            f.write(f"\nFASTEST:\n")
            sorted_by_time = sorted(models_summary, key=lambda x: x['avg_time'])
            for i, model in enumerate(sorted_by_time[:5], 1):
                f.write(f"  {i}. {model['model']} - {model['avg_time']:.2f}s\n")
        
        print(f"Summary table saved to: {table_path}")

    def run_complete_evaluation(self):
        """
        Executes the complete evaluation workflow for all models.

        This method orchestrates the discovery of models and systems,
        the evaluation of each model on all systems, saving individual
        model results, and finally generating a consolidated comparison report.
        """
        
        print("INITIATING COMPLETE MODEL EVALUATION")
        print("=" * 60)
        
        start_time = time()
        
        # Discover models and systems
        models = self.discover_models()
        systems = self.discover_systems()
        
        if not models:
            print("No models found for evaluation")
            return
        
        if not systems:
            print("No systems found for evaluation")
            return
        
        print(f"\nEvaluation Plan:")
        print(f"  - {len(models)} models")
        print(f"  - {len(systems)} systems")
        print(f"  - {len(models) * len(systems)} total evaluations")
        
        # Evaluate each model
        all_results = []
        
        for i, model_info in enumerate(models, 1):
            print(f"\n{'='*60}")
            print(f"MODEL {i}/{len(models)}")
            print(f"{'='*60}")
            
            model_result = self.evaluate_model(model_info, systems)
            self.save_model_results(model_info, model_result)
            all_results.append(model_result)
        
        # Generate consolidated comparison
        print(f"\n{'='*60}")
        print(f"CONSOLIDATING RESULTS")
        print(f"{'='*60}")
        
        self.generate_consolidated_comparison(all_results)
        
        # Agregar después de generate_consolidated_comparison()
        print(f"Checking model differentiation...")
        self.check_model_differentiation(all_results)
        self.generate_enhanced_system_analysis(all_results)
        
        # Generate detailed reports by instance
        print(f"Generating detailed instance-by-instance results...")
        self.generate_detailed_results_by_instance(all_results)
        self.generate_system_comparison_matrix(all_results)
        
        # Final summary
        total_time = time() - start_time
        
        print(f"\nEVALUATION COMPLETED")
        print(f"{'='*60}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Results in: {self.output_dir}")
        print(f"Successful evaluations: {sum(1 for r in all_results if r['model_summary']['systems_evaluated'] > 0)}/{len(all_results)}")
        
        # Show best models
        valid_results = [r for r in all_results if r['model_summary']['systems_evaluated'] > 0]
        if valid_results:
            best_satisfaction = max(valid_results, key=lambda x: x['model_summary']['avg_satisfaction_pct'])
            best_cost = min(valid_results, key=lambda x: x['model_summary']['avg_cost_dollars'])
            
            print(f"\nBEST MODELS:")
            print(f"  - Best satisfaction: {best_satisfaction['model_info']['archetype']}_rank_{best_satisfaction['model_info']['rank']} ({best_satisfaction['model_summary']['avg_satisfaction_pct']:.1f}%)")
            print(f"  - Best cost: {best_cost['model_info']['archetype']}_rank_{best_cost['model_info']['rank']} (${best_cost['model_summary']['avg_cost_dollars']:.2f})")
            
            
def main():
    # Configure paths
    models_dir = "results/scatter_search/trained_models"
    systems_dir = "src/configs/system_data"
    output_dir = "results/scatter_search/model_solutions"

    # Verify directories exist
    if not os.path.exists(models_dir):
        print(f"Model directory not found: {models_dir}")
        return

    if not os.path.exists(systems_dir):
        print(f"System directory not found: {systems_dir}")
        return

    # Create evaluator and run
    evaluator = ModelEvaluator(models_dir, systems_dir, output_dir)
    evaluator.run_complete_evaluation()
    
if __name__ == "__main__":
    main()