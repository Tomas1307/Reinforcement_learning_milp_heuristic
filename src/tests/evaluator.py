import sys
import os
import re
import glob
import json
import numpy as np
import torch
import psutil
import gc
from time import time
from datetime import datetime
from typing import List, Dict, Any

# Fix imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dqn_agent.agent import EnhancedDQNAgentPyTorch
from src.common.config import load_system_config


class EfficientSlotBySlotSimulator:
    """
    Efficient simulator that predicts slot by slot in an optimized way.
    CORRECTED: Handles slots without chargers by "tricking" the model.
    """
    
    def __init__(self, models_dir: str, systems_dir: str, output_dir: str):
        """
        Initializes the EfficientSlotBySlotSimulator.

        Args:
            models_dir (str): Directory where the trained models are located.
            systems_dir (str): Directory where the system configuration files (JSON) are located.
            output_dir (str): Directory to save the simulation results.
        """
        self.models_dir = models_dir
        self.systems_dir = systems_dir
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"EfficientSlotBySlotSimulator initialized")
        print(f" Models: {models_dir}")
        print(f" Systems: {systems_dir}")
        print(f" Output: {output_dir}")
    
    def discover_models(self) -> List[Dict[str, Any]]:
        """
        Discovers available models in the specified models directory.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing information
                                  about a discovered model (archetype, rank, filename, path).
        """
        print("\nDiscovering models...")
        
        model_files = glob.glob(os.path.join(self.models_dir, "*.pt"))
        models = []
        
        for model_path in model_files:
            filename = os.path.basename(model_path)
            
            # Extract archetype and rank
            match = re.match(r'(.+)_rank_(\d+)\.pt$', filename)
            if match:
                archetype = match.group(1)
                rank = int(match.group(2))
            else:
                match = re.match(r'(.+)_(\d+)\.pt$', filename)
                if match:
                    archetype = match.group(1)
                    rank = int(match.group(2))
                else:
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
            print(f" - {model['archetype']} rank {model['rank']}")
        
        return models
    
    def discover_systems(self) -> List[Dict[str, Any]]:
        """
        Discovers available system configurations in the specified systems directory.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing information
                                  about a discovered system (system_id, filename, path, config,
                                  num_vehicles, num_slots, num_chargers).
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
                }
                systems.append(system_info)
                
            except Exception as e:
                print(f"Error loading {json_path}: {e}")
        
        systems.sort(key=lambda x: x['system_id'])
        
        print(f"Found {len(systems)} systems:")
        for system in systems:
            print(f" - System {system['system_id']}: {system['num_vehicles']} vehicles, "
                  f"{system['num_slots']} slots, {system['num_chargers']} chargers")
        
        return systems
    
    def load_agent_once(self, model_path: str):
        """
        Loads the DQN agent from the specified model path.

        Args:
            model_path (str): The file path to the trained agent model.

        Returns:
            EnhancedDQNAgentPyTorch: The loaded and configured DQN agent.

        Raises:
            ValueError: If the model cannot be loaded.
        """
        print(f"Loading agent from: {os.path.basename(model_path)}")
        
        state_size = 40
        action_size = 60
        
        # Use Simple DQN (not Dueling) for compatibility
        agent = EnhancedDQNAgentPyTorch(state_size, action_size, dueling_network=False)
        
        if agent.load(model_path):
            agent.epsilon = 0.0  # No exploration
            print("Agent loaded successfully")
            return agent
        else:
            raise ValueError(f"Could not load model: {model_path}")
    
    def check_memory(self, stage=""):
        """
        Monitors and prints current memory usage (RAM and GPU).

        Args:
            stage (str): A string indicating the current stage of the process for logging.

        Returns:
            float: Current RAM usage in MB.
        """
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
        print(f" {stage} - RAM: {memory_mb:.1f}MB, GPU: {gpu_memory:.1f}MB")
        return memory_mb

    def simulate_system_efficient(self, agent, system_config: Dict):
        """
        Simulates the entire time horizon for a given system configuration.
        It iterates through all timesteps where vehicles are available.

        Args:
            agent: The DQN agent used for making assignment decisions.
            system_config (Dict): The configuration dictionary for the system to simulate.

        Returns:
            List[Tuple]: A list of tuples, where each tuple represents a schedule entry
                         (vehicle_id, time_idx, charger_id, slot, power).
        """
        print(f"\nEFFICIENT SIMULATION - System with {len(system_config['arrivals'])} vehicles")
        self.check_memory("Simulation start")
        
        all_schedule_entries = []  # List for ALL temporal assignments
        vehicle_assignment_history = set()  # For tracking already assigned vehicles
        
        total_timesteps = len(system_config["times"])
        print(f" Total available timesteps: {total_timesteps}")
        
        # ITERATE THROUGH ALL TIMESTEPS (WITHOUT ARTIFICIAL LIMITS)
        timesteps_processed = 0
        for timestep_idx, current_time in enumerate(system_config["times"]):
            
            # Get available vehicles for THIS timestep
            available_vehicles = []
            for arrival in system_config["arrivals"]:
                if (arrival["arrival_time"] <= current_time < arrival["departure_time"] and 
                    arrival["id"] not in vehicle_assignment_history):
                    available_vehicles.append(arrival["id"])
            
            if not available_vehicles:
                continue  # Skip timesteps with no new vehicles
                
            timesteps_processed += 1
            print(f" Timestep {timestep_idx} (t={current_time:.2f}h): {len(available_vehicles)} vehicles available")
            
            # Log progress every 10 timesteps
            if timesteps_processed % 10 == 0:
                self.check_memory(f"Timestep {timestep_idx}")
            
            # SLOT BY SLOT PREDICTION FOR THIS TIMESTEP
            timestep_assignments = self._predict_all_slots_for_timestep(
                agent, system_config, timestep_idx, available_vehicles
            )
            
            # Convert assignments to schedule format and mark vehicles as assigned
            for assignment in timestep_assignments:
                schedule_entry = (
                    assignment['vehicle_id'],     # ev_id
                    timestep_idx,                 # time_idx
                    assignment['charger'],        # charger_id (can be None)
                    assignment['slot'],           # slot
                    assignment['power']           # power
                )
                all_schedule_entries.append(schedule_entry)
                
                # Mark vehicle as assigned so it's not reassigned
                vehicle_assignment_history.add(assignment['vehicle_id'])
            
            print(f"      Assignments in timestep {timestep_idx}: {len(timestep_assignments)}")
            print(f"      Total vehicles assigned so far: {len(vehicle_assignment_history)}")
            
            # Periodic cleanup every 20 timesteps
            if timesteps_processed % 20 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # FINAL SUMMARY
        unique_vehicles = len(vehicle_assignment_history)
        total_vehicles = len(system_config["arrivals"])
        
        print(f"\nTEMPORAL SIMULATION COMPLETED:")
        print(f" Total schedule entries: {len(all_schedule_entries)}")
        print(f" Unique vehicles assigned: {unique_vehicles}/{total_vehicles} ({unique_vehicles/total_vehicles*100:.1f}%)")
        print(f" Timesteps processed: {timesteps_processed}/{total_timesteps}")
        self.check_memory("Simulation end")
        
        return all_schedule_entries
    
    def _predict_all_slots_for_timestep(self, agent, system_config, timestep_idx, available_vehicles):
        """
        Predicts vehicle assignments for all slots available at a given timestep.
        This includes "tricking" the model for slots without chargers to consider them for parking.

        Args:
            agent: The DQN agent.
            system_config (Dict): The system configuration.
            timestep_idx (int): The current timestep index.
            available_vehicles (List[int]): A list of vehicle IDs available for assignment.

        Returns:
            List[Dict]: A list of assignment dictionaries for the current timestep.
        """
        
        if not available_vehicles:
            return []
        
        # IDENTIFY SLOTS WITH AND WITHOUT CHARGERS
        slots_with_chargers = []
        slots_without_chargers = []
        
        # Assume each charger can go in any slot
        for i, charger in enumerate(system_config["chargers"]):
            if i < system_config["n_spots"]:  # Only if there are enough slots
                slots_with_chargers.append({
                    'slot': i,
                    'charger': charger.get("charger_id", i),
                    'power': charger.get("power", 7)
                })
        
        # Remaining slots are without chargers
        slots_used_for_chargers = len(slots_with_chargers)
        for slot_id in range(slots_used_for_chargers, system_config["n_spots"]):
            slots_without_chargers.append(slot_id)
        
        print(f"    Slots with charger: {len(slots_with_chargers)}")
        print(f"    Slots without charger: {len(slots_without_chargers)}")
        
        # EFFICIENT SLOT BY SLOT PREDICTION
        assignments = []
        working_vehicle_list = available_vehicles.copy()  # WORKING COPY
        
        # FIRST: Slots with chargers (most valuable)
        print(f"    Predicting slots WITH charger...")
        for i, slot_info in enumerate(slots_with_chargers):
            
            if not working_vehicle_list:
                break
                
            print(f"      Slot {slot_info['slot']} + Charger {slot_info['charger']} ({i+1}/{len(slots_with_chargers)})")
            
            # Predict for this specific slot
            selected_vehicle = self._predict_for_single_slot(
                agent, system_config, timestep_idx, 
                slot_info['slot'], working_vehicle_list, 
                charger_info=slot_info
            )
            
            if selected_vehicle is not None:
                # Create assignment
                assignment = {
                    'slot': slot_info['slot'],
                    'vehicle_id': selected_vehicle,
                    'charger': slot_info['charger'],
                    'power': slot_info['power'],
                    'timestep': timestep_idx,
                    'type': 'charging'
                }
                assignments.append(assignment)
                
                # REMOVE vehicle from working list
                working_vehicle_list.remove(selected_vehicle)
                print(f"        Assigned: EV_{selected_vehicle}")
                print(f"        Remaining vehicles: {len(working_vehicle_list)}")
            else:
                print(f"        No vehicle selected")
        
        # SECOND: Slots without chargers (WITH MODEL TRICKING)
        print(f"    Predicting slots WITHOUT charger (using model trick)...")
        for i, slot_id in enumerate(slots_without_chargers):
            
            if not working_vehicle_list:
                break
                
            print(f"      Slot {slot_id} (parking with trick) ({i+1}/{len(slots_without_chargers)})")
            
            # TRICK: Create a fictitious charger_info to trick the model
            fake_charger_info = {
                'slot': slot_id,
                'charger': 999,  # Fictitious ID
                'power': 0.1     # Minimum power to trick
            }
            
            # Predict for this slot using the trick
            selected_vehicle = self._predict_for_single_slot(
                agent, system_config, timestep_idx,
                slot_id, working_vehicle_list,
                charger_info=fake_charger_info,
                is_fake_charger=True  # New flag
            )
            
            if selected_vehicle is not None:
                # Create REAL assignment (no charger)
                assignment = {
                    'slot': slot_id,
                    'vehicle_id': selected_vehicle,
                    'charger': None,  # REAL: No charger
                    'power': 0,       # REAL: No power
                    'timestep': timestep_idx,
                    'type': 'parking'
                }
                assignments.append(assignment)
                
                # REMOVE vehicle from working list
                working_vehicle_list.remove(selected_vehicle)
                print(f"        Assigned: EV_{selected_vehicle}")
                print(f"        CONVERSION: EV_{selected_vehicle} tricked for parking in slot {slot_id}")
                print(f"        Remaining vehicles: {len(working_vehicle_list)}")
            else:
                print(f"        No vehicle selected")
        
        return assignments
    
    def _predict_for_single_slot(self, agent, system_config, timestep_idx, 
                                 slot_id, available_vehicles, charger_info=None,
                                 is_fake_charger=False):
        """
        Predicts which vehicle to assign to a SPECIFIC SLOT.
        This includes handling the "trick" for slots without chargers.

        Args:
            agent: The DQN agent.
            system_config (Dict): The system configuration.
            timestep_idx (int): The current timestep index.
            slot_id (int): The ID of the slot for which to make a prediction.
            available_vehicles (List[int]): A list of vehicle IDs available for assignment.
            charger_info (Dict, optional): Information about the charger at this slot. Defaults to None.
            is_fake_charger (bool, optional): True if the charger information is fictitious for a parking slot. Defaults to False.

        Returns:
            int or None: The ID of the selected vehicle, or None if no vehicle is selected.
        """
        
        if not available_vehicles:
            return None
        
        # Create state for this prediction
        state = self._build_slot_state(
            system_config, timestep_idx, available_vehicles, 
            slot_id, charger_info, is_fake_charger
        )
        
        # Generate possible actions for this slot
        possible_actions = self._generate_slot_actions(
            slot_id, available_vehicles, charger_info, is_fake_charger
        )
        
        print(f"        DEBUG - Vehicles: {available_vehicles}")
        print(f"        DEBUG - Actions: {len(possible_actions)}")
        print(f"        DEBUG - Fake charger: {is_fake_charger}")
        
        if len(possible_actions) <= 1:  # Only no_action
            print(f"        DEBUG - Only no_action available")
            return None
        
        # AGENT PREDICTION
        print(f"        BEFORE agent.act():")
        print(f"          - state type: {type(state)}")
        print(f"          - possible_actions length: {len(possible_actions)}")
        print(f"          - agent epsilon: {agent.epsilon}")

        action_idx = agent.act(state, possible_actions, verbose=False)
        
        print(f"        AFTER agent.act(): {action_idx}")
        
        # DEBUG: What was chosen
        print(f"        DEBUG - Chosen action: {action_idx}/{len(possible_actions)-1}")
        if action_idx >= 0 and action_idx < len(possible_actions):
            print(f"        DEBUG - Action detail: {possible_actions[action_idx]}")
            
        if action_idx == -1 or action_idx >= len(possible_actions):
            return None
        
        selected_action = possible_actions[action_idx]
        
        # Check if it's no_action
        if selected_action.get("type") == "no_action":
            return None
        
        print(f"        DEBUG - selected_action: {selected_action}")
        print(f"        DEBUG - vehicle_id: {selected_action.get('vehicle_id')}")

        # Return the selected vehicle
        vehicle_id = selected_action.get("vehicle_id")
        print(f"        DEBUG - returning: {vehicle_id}")
        
        return vehicle_id
    
    def _build_slot_state(self, system_config, timestep_idx, available_vehicles, 
                          slot_id, charger_info, is_fake_charger=False):
        """
        Constructs the state for a specific slot prediction.
        This includes handling the "trick" for slots without chargers.

        Args:
            system_config (Dict): The system configuration.
            timestep_idx (int): The current timestep index.
            available_vehicles (List[int]): A list of vehicle IDs currently available.
            slot_id (int): The ID of the slot for which the state is being built.
            charger_info (Dict, optional): Information about the charger at this slot. Defaults to None.
            is_fake_charger (bool, optional): True if the charger information is fictitious for a parking slot. Defaults to False.

        Returns:
            Dict: A dictionary representing the state for the agent's prediction.
        """
        
        current_time = system_config["times"][timestep_idx]
        
        # Select most urgent vehicle as representative
        representative_vehicle = self._select_most_urgent_vehicle(
            available_vehicles, system_config, current_time
        )
        
        state = {
            "evs_present": available_vehicles.copy(),
            "available_spots": [slot_id],  # Only this slot
            "current_time_idx": timestep_idx,
            "current_time_normalized": timestep_idx / len(system_config["times"]),
            "representative_ev": representative_vehicle
        }
        
        # CORRECTED LOGIC FOR THE TRICK
        if charger_info is None or (is_fake_charger and charger_info['charger'] == 999):
            # REAL CASE: No charger (but we can trick the model)
            if is_fake_charger:
                # TRICK: Make it believe there is a charger
                state["available_chargers"] = [999]  # Fictitious ID
                state["avg_available_chargers"] = 0.1  # Small but not zero
                print(f"        TRICK: Simulating fictitious charger for slot {slot_id}")
            else:
                # NO TRICK: Tell the truth (no charger)
                state["available_chargers"] = []
                state["avg_available_chargers"] = 0.0
                print(f"        TRUTH: No charger for slot {slot_id}")
        else:
            # NORMAL CASE: There is a real charger
            state["available_chargers"] = [charger_info['charger']]
            state["avg_available_chargers"] = 1.0 / len(system_config["chargers"])
        
        # Features of the representative vehicle
        if representative_vehicle:
            arrival_info = next(arr for arr in system_config["arrivals"] 
                                 if arr["id"] == representative_vehicle)
            
            energy_norm = arrival_info["required_energy"] / 100.0
            time_remaining_norm = (arrival_info["departure_time"] - current_time) / max(system_config["times"])
            arrival_norm = arrival_info["arrival_time"] / max(system_config["times"])
            departure_norm = arrival_info["departure_time"] / max(system_config["times"])
            
            ev_features = [
                arrival_norm,
                departure_norm,
                energy_norm,
                time_remaining_norm,
                0.0,  # energy_delivered_ratio
                timestep_idx / len(system_config["times"]),
                time_remaining_norm
            ]
            
            state["ev_features"] = ev_features
        
        # Basic system metrics
        state.update({
            "avg_available_spots": 1.0 / system_config["n_spots"],
            "min_price": 0.5,
            "avg_price": 0.5,
            "system_demand_ratio": len(available_vehicles) / len(system_config["arrivals"]),
            "competition_pressure": min(1.0, len(available_vehicles) / system_config["n_spots"])
        })
        
        print(f"        STATE DEBUG:")
        print(f"          - evs_present: {state.get('evs_present')}")
        print(f"          - available_spots: {state.get('available_spots')}")  
        print(f"          - available_chargers: {state.get('available_chargers')}")
        print(f"          - avg_available_chargers: {state.get('avg_available_chargers')}")
        print(f"          - representative_ev: {state.get('representative_ev')}")
        print(f"          - ev_features length: {len(state.get('ev_features', []))}")
                
        return state
    
    def _select_most_urgent_vehicle(self, available_vehicles, system_config, current_time):
        """
        Selects the most urgent vehicle from the available vehicles.
        Urgency is determined by the ratio of required energy to remaining time until departure.

        Args:
            available_vehicles (List[int]): A list of vehicle IDs available for assignment.
            system_config (Dict): The system configuration, containing vehicle arrival information.
            current_time (float): The current simulation time.

        Returns:
            int or None: The ID of the most urgent vehicle, or None if no vehicles are available.
        """
        
        if not available_vehicles:
            return None
        
        urgencies = []
        for vehicle_id in available_vehicles:
            arrival_info = next(arr for arr in system_config["arrivals"] 
                                 if arr["id"] == vehicle_id)
            
            energy_needed = arrival_info["required_energy"]
            time_remaining = max(1e-6, arrival_info["departure_time"] - current_time)
            urgency = energy_needed / time_remaining
            
            urgencies.append((vehicle_id, urgency))
        
        return max(urgencies, key=lambda x: x[1])[0]
    
    def _generate_slot_actions(self, slot_id, available_vehicles, charger_info, is_fake_charger=False):
        """
        Generates possible actions for a SPECIFIC slot.
        This includes handling the "trick" for slots without chargers.

        Args:
            slot_id (int): The ID of the slot for which to generate actions.
            available_vehicles (List[int]): A list of vehicle IDs available for assignment.
            charger_info (Dict, optional): Information about the charger at this slot. Defaults to None.
            is_fake_charger (bool, optional): True if the charger information is fictitious for a parking slot. Defaults to False.

        Returns:
            List[Dict]: A list of dictionaries, where each dictionary represents a possible action.
        """
        
        actions = [{"type": "no_action"}]
        
        for vehicle_id in available_vehicles:
            
            if charger_info and not (is_fake_charger and charger_info['charger'] == 999):
                # NORMAL CASE: There is a real charger
                actions.append({
                    "type": "assign_charging",
                    "vehicle_id": vehicle_id,
                    "slot": slot_id,
                    "charger": charger_info['charger'],
                    "power": charger_info['power']
                })
            elif is_fake_charger:
                # TRICK CASE: Make it believe there is a charger
                actions.append({
                    "type": "assign_charging",  # LIE: The model thinks it's charging
                    "vehicle_id": vehicle_id,
                    "slot": slot_id,
                    "charger": 999,  # Fictitious ID
                    "power": 0.1     # Fictitious power
                })
            else:
                # HONEST CASE: Admit it's only parking
                actions.append({
                    "type": "assign_parking",
                    "vehicle_id": vehicle_id,
                    "slot": slot_id,
                    "charger": None,
                    "power": 0
                })
        
        print(f"        ACTIONS DEBUG:")
        for i, action in enumerate(actions):
            print(f"          [{i}]: {action}")
            
        return actions
    
    def evaluate_model_on_system(self, model_info: Dict, system_info: Dict):
        """
        Evaluates a single model on a single system using the efficient temporal method.

        Args:
            model_info (Dict): Dictionary containing information about the model to evaluate.
            system_info (Dict): Dictionary containing information about the system to simulate.

        Returns:
            Dict: A dictionary containing the evaluation results and metrics.
        """
        
        model_name = f"{model_info['archetype']}_rank_{model_info['rank']}"
        system_id = system_info['system_id']
        
        print(f"\n{'='*50}")
        print(f"EVALUATING: {model_name} on System {system_id}")
        print(f"{'='*50}")
        
        self.check_memory("Evaluation start")
        
        try:
            start_time = time()
            
            # LOAD AGENT ONCE
            agent = self.load_agent_once(model_info['path'])
            self.check_memory("Agent loaded")
            
            # SIMULATE USING EFFICIENT TEMPORAL METHOD
            schedule_entries = self.simulate_system_efficient(agent, system_info['config'])
            self.check_memory("Simulation completed")
            
            # CALCULATE METRICS
            metrics = self._calculate_metrics_from_schedule(schedule_entries, system_info['config'])
                
            execution_time = time() - start_time
            
            result = {
                'system_id': system_id,
                'model_info': {
                    'archetype': model_info['archetype'],
                    'rank': model_info['rank'],
                    'filename': model_info['filename']
                },
                'system_metrics': {
                    'num_vehicles': system_info['num_vehicles'],
                    'num_slots': system_info['num_slots'],
                    'num_chargers': system_info['num_chargers'],
                    'system_context': metrics['system_context']
                },
                'execution_performance': {
                    'execution_time_seconds': round(execution_time, 3),
                    'schedule_entries': len(schedule_entries),
                    'timesteps_used': metrics['timesteps_used'],
                    'time_coverage_pct': metrics['time_coverage_pct']
                },
                'vehicle_performance': {
                    'vehicles_assigned': metrics['vehicles_assigned'],
                    'vehicles_total': metrics['total_vehicles'],
                    'assignment_ratio_pct': round(metrics['assignment_ratio'] * 100, 1),
                    'vehicles_fully_satisfied': metrics['vehicles_fully_satisfied'],
                    'vehicles_partially_satisfied': metrics['vehicles_partially_satisfied'],
                    'vehicles_not_satisfied': metrics['vehicles_not_satisfied'],
                    'charging_assignments': metrics['charging_assignments'],
                    'parking_assignments': metrics['parking_assignments']
                },
                'energy_performance': {
                    'total_energy_required_kwh': round(metrics['total_energy_required'], 2),
                    'total_energy_delivered_kwh': round(metrics['total_energy_delivered'], 2),
                    'overall_satisfaction_pct': round(metrics['overall_satisfaction_pct'], 1),
                    'energy_deficit_kwh': round(metrics['energy_deficit'], 2),
                    'satisfaction_distribution': metrics['satisfaction_distribution']
                },
                'economic_performance': {
                    'total_energy_cost': round(metrics['total_energy_cost'], 2),
                    'avg_energy_price_per_kwh': round(metrics['avg_energy_price'], 3),
                    'cost_efficiency': round(metrics['cost_per_kwh_delivered'], 3)
                },
                'resource_utilization': {
                    'capacity_utilization_pct': round(metrics['capacity_utilization_pct'], 1),
                    'spots_utilization_pct': round(metrics['spots_utilization_pct'], 1),
                    'chargers_utilization_pct': round(metrics['chargers_utilization_pct'], 1)
                },
                'priority_analysis': metrics['satisfaction_by_priority'],
                'detailed_ev_metrics': metrics['detailed_ev_metrics'],
                'priority_group_metrics': metrics['priority_group_metrics'],
                'schedule_detail': schedule_entries  # For Gantt compatibility
            }
            
            # Enhanced summary log
            print(f"\nFULL RESULT:")
            print(f"  Overall satisfaction: {metrics['overall_satisfaction_pct']:.1f}%")
            print(f"  Fully satisfied vehicles: {metrics['vehicles_fully_satisfied']}/{metrics['total_vehicles']}")
            print(f"  Energy delivered: {metrics['total_energy_delivered']:.1f}/{metrics['total_energy_required']:.1f} kWh")
            print(f"  Total cost: ${metrics['total_energy_cost']:.2f}")
            print(f"  Capacity utilization: {metrics['capacity_utilization_pct']:.1f}%")
            print(f"  Charger utilization: {metrics['chargers_utilization_pct']:.1f}%")
            print(f"  Execution time: {execution_time:.1f}s")
            
            if metrics['satisfaction_by_priority']:
                print(f"  Analysis by priority:")
                for priority, data in metrics['satisfaction_by_priority'].items():
                    print(f"    {priority}: {data['satisfaction_pct']:.1f}% ({data['vehicles_count']} vehicles)")
            
            return result
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'system_id': system_id,
                'error': str(e),
                'model_info': model_info,
                'system_metrics': {
                    'num_vehicles': system_info['num_vehicles'],
                    'num_slots': system_info['num_slots'],
                    'num_chargers': system_info['num_chargers']
                }
            }
        finally:
            # FORCED CLEANUP
            if 'agent' in locals():
                del agent
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.check_memory("Final cleanup")
        

    def _calculate_metrics_from_schedule(self, schedule_entries, system_config):
        """Calculates corrected metrics from the schedule.

        Args:
            schedule_entries (list): A list of schedule entries, where each entry is a tuple
                                    (ev_id, time_idx, charger_id, slot, power).
            system_config (dict): A dictionary containing system configuration details,
                                including 'arrivals', 'dt', 'prices', 'chargers',
                                'n_spots', 'parking_config', 'times', and 'station_limit'.

        Returns:
            dict: A dictionary containing various calculated metrics for the schedule,
                such as energy delivered, satisfaction, resource utilization, and costs.
        """

        print(f"Calculating metrics for {len(schedule_entries)} schedule entries...")

        unique_vehicles = len(set(entry[0] for entry in schedule_entries))
        total_vehicles = len(system_config['arrivals'])
        unique_timesteps = len(set(entry[1] for entry in schedule_entries))

        charging_entries = len([entry for entry in schedule_entries if entry[2] is not None and entry[4] > 0])
        parking_entries = len([entry for entry in schedule_entries if entry[2] is None or entry[4] == 0])

        print(f"Unique vehicles: {unique_vehicles}/{total_vehicles}")
        print(f"Charging entries: {charging_entries}, Parking entries: {parking_entries}")

        ev_energy_delivered = {}
        ev_required_energy = {}

        for arrival in system_config['arrivals']:
            ev_id = arrival['id']
            ev_required_energy[ev_id] = arrival['required_energy']
            ev_energy_delivered[ev_id] = 0.0

        dt = system_config.get('dt', 0.25)
        print(f"System dt: {dt} hours")

        total_energy_cost = 0

        for entry in schedule_entries:
            ev_id, time_idx, charger_id, slot, power = entry

            if charger_id is not None and power > 0:
                energy_this_slot = power * dt
                print(f"EV {ev_id}: {power}kW x {dt}h = {energy_this_slot:.3f} kWh")

                ev_energy_delivered[ev_id] += energy_this_slot

                if time_idx < len(system_config.get("prices", [])):
                    price = system_config["prices"][time_idx]
                    total_energy_cost += energy_this_slot * price
                else:
                    total_energy_cost += energy_this_slot * 0.5

        total_energy_required = sum(ev_required_energy.values())
        total_energy_delivered = sum(ev_energy_delivered.values())
        overall_satisfaction_pct = (total_energy_delivered / total_energy_required * 100) if total_energy_required > 0 else 0

        print(f"Total energy required: {total_energy_required:.1f} kWh")
        print(f"Total energy delivered: {total_energy_delivered:.1f} kWh")
        print(f"Overall satisfaction: {overall_satisfaction_pct:.1f}%")

        vehicles_fully_satisfied = 0
        vehicles_partially_satisfied = 0
        vehicles_not_satisfied = 0

        satisfaction_ratios = []

        for ev_id in ev_required_energy.keys():
            delivered = ev_energy_delivered[ev_id]
            required = ev_required_energy[ev_id]
            satisfaction_ratio = delivered / required if required > 0 else 0
            satisfaction_ratios.append(satisfaction_ratio)

            if satisfaction_ratio >= 0.99:
                vehicles_fully_satisfied += 1
            elif satisfaction_ratio > 0.01:
                vehicles_partially_satisfied += 1
            else:
                vehicles_not_satisfied += 1

        if satisfaction_ratios:
            satisfaction_distribution = {
                'min_satisfaction': min(satisfaction_ratios),
                'max_satisfaction': max(satisfaction_ratios),
                'avg_satisfaction': sum(satisfaction_ratios) / len(satisfaction_ratios),
                'std_satisfaction': float(np.std(satisfaction_ratios))
            }
        else:
            satisfaction_distribution = {
                'min_satisfaction': 0,
                'max_satisfaction': 0,
                'avg_satisfaction': 0,
                'std_satisfaction': 0
            }

        total_chargers = len(system_config.get("chargers", []))
        total_spots = system_config.get("n_spots", system_config.get("parking_config", {}).get("n_spots", 10))
        total_timesteps = len(system_config.get("times", []))

        max_charger_power = sum(c.get("power", 7) for c in system_config.get("chargers", []))
        theoretical_max_energy = max_charger_power * dt * total_timesteps
        capacity_utilization = (total_energy_delivered / theoretical_max_energy * 100) if theoretical_max_energy > 0 else 0

        unique_slot_time_pairs = len(set((entry[1], entry[3]) for entry in schedule_entries))
        total_slot_opportunities = total_spots * total_timesteps
        spots_utilization = (unique_slot_time_pairs / total_slot_opportunities * 100) if total_slot_opportunities > 0 else 0

        unique_charger_time_pairs = len(set((entry[1], entry[2]) for entry in schedule_entries if entry[2] is not None))
        total_charger_opportunities = total_chargers * total_timesteps
        chargers_utilization = (unique_charger_time_pairs / total_charger_opportunities * 100) if total_charger_opportunities > 0 else 0

        print(f"Capacity utilization: {capacity_utilization:.1f}%")
        print(f"Spots utilization: {spots_utilization:.1f}%")
        print(f"Chargers utilization: {chargers_utilization:.1f}%")

        detailed_ev_metrics = {}
        for ev_id in ev_required_energy.keys():
            delivered = ev_energy_delivered[ev_id]
            required = ev_required_energy[ev_id]
            satisfaction = delivered / required if required > 0 else 0

            detailed_ev_metrics[str(ev_id)] = {
                "required_energy": required,
                "delivered_energy": delivered,
                "satisfaction": satisfaction,
                "priority": 1,
                "willingness": 1.0
            }

        avg_energy_price = total_energy_cost / total_energy_delivered if total_energy_delivered > 0 else 0

        return {
            'vehicles_assigned': unique_vehicles,
            'total_vehicles': total_vehicles,
            'assignment_ratio': unique_vehicles / total_vehicles if total_vehicles > 0 else 0,
            'timesteps_used': unique_timesteps,
            'charging_assignments': charging_entries,
            'parking_assignments': parking_entries,

            'total_energy_required': total_energy_required,
            'total_energy_delivered': total_energy_delivered,
            'overall_satisfaction_pct': overall_satisfaction_pct,
            'energy_deficit': total_energy_required - total_energy_delivered,

            'total_energy_cost': total_energy_cost,
            'avg_energy_price': avg_energy_price,
            'cost_per_kwh_delivered': avg_energy_price,

            'capacity_utilization_pct': capacity_utilization,
            'spots_utilization_pct': spots_utilization,
            'chargers_utilization_pct': chargers_utilization,

            'vehicles_fully_satisfied': vehicles_fully_satisfied,
            'vehicles_partially_satisfied': vehicles_partially_satisfied,
            'vehicles_not_satisfied': vehicles_not_satisfied,

            'satisfaction_by_priority': {},

            'detailed_ev_metrics': detailed_ev_metrics,

            'priority_group_metrics': {},

            'satisfaction_distribution': satisfaction_distribution,

            'avg_time_per_assignment': unique_timesteps / unique_vehicles if unique_vehicles > 0 else 0,
            'time_coverage_pct': unique_timesteps / total_timesteps * 100 if total_timesteps > 0 else 0,

            'system_context': {
                'total_spots': total_spots,
                'total_chargers': total_chargers,
                'total_timesteps': total_timesteps,
                'transformer_limit': system_config.get("station_limit", system_config.get("parking_config", {}).get("transformer_limit", 100)),
                'simulation_duration_hours': max(system_config.get("times", [0])) if system_config.get("times") else 0,
                'dt_hours': dt,
                'max_charger_power_total': max_charger_power,
                'theoretical_max_energy': theoretical_max_energy,
                'has_priority_system': False,
                'has_willingness_to_pay': False,
                'has_brand_compatibility': False
            }
        }


    def save_results(self, model_info: Dict, all_results: List[Dict]):
        """Saves evaluation results including comprehensive metrics.

        Args:
            model_info (Dict): Dictionary containing information about the model being evaluated.
            all_results (List[Dict]): A list of dictionaries, where each dictionary
                                    represents the evaluation results for a single system.
        """

        model_name = f"{model_info['archetype']}_rank_{model_info['rank']}"
        output_file = os.path.join(self.output_dir, f"{model_name}_corrected_temporal_results.json")

        successful_results = [r for r in all_results if 'error' not in r]
        failed_results = [r for r in all_results if 'error' in r]

        aggregated_stats = {}
        if successful_results:
            aggregated_stats = {
                'overall_performance': {
                    'avg_satisfaction_pct': round(np.mean([r['energy_performance']['overall_satisfaction_pct'] for r in successful_results]), 1),
                    'min_satisfaction_pct': round(min([r['energy_performance']['overall_satisfaction_pct'] for r in successful_results]), 1),
                    'max_satisfaction_pct': round(max([r['energy_performance']['overall_satisfaction_pct'] for r in successful_results]), 1),
                    'avg_assignment_ratio_pct': round(np.mean([r['vehicle_performance']['assignment_ratio_pct'] for r in successful_results]), 1),
                    'avg_execution_time_seconds': round(np.mean([r['execution_performance']['execution_time_seconds'] for r in successful_results]), 2)
                },
                'energy_statistics': {
                    'total_energy_required_all_systems': round(sum([r['energy_performance']['total_energy_required_kwh'] for r in successful_results]), 2),
                    'total_energy_delivered_all_systems': round(sum([r['energy_performance']['total_energy_delivered_kwh'] for r in successful_results]), 2),
                    'total_energy_deficit_all_systems': round(sum([r['energy_performance']['energy_deficit_kwh'] for r in successful_results]), 2),
                    'avg_cost_per_kwh': round(np.mean([r['economic_performance']['cost_efficiency'] for r in successful_results if r['economic_performance']['cost_efficiency'] > 0]), 3)
                },
                'resource_efficiency': {
                    'avg_capacity_utilization_pct': round(np.mean([r['resource_utilization']['capacity_utilization_pct'] for r in successful_results]), 1),
                    'avg_spots_utilization_pct': round(np.mean([r['resource_utilization']['spots_utilization_pct'] for r in successful_results]), 1),
                    'avg_chargers_utilization_pct': round(np.mean([r['resource_utilization']['chargers_utilization_pct'] for r in successful_results]), 1)
                },
                'vehicle_satisfaction_summary': {
                    'total_vehicles_all_systems': sum([r['vehicle_performance']['vehicles_total'] for r in successful_results]),
                    'total_vehicles_fully_satisfied': sum([r['vehicle_performance']['vehicles_fully_satisfied'] for r in successful_results]),
                    'total_vehicles_partially_satisfied': sum([r['vehicle_performance']['vehicles_partially_satisfied'] for r in successful_results]),
                    'total_vehicles_not_satisfied': sum([r['vehicle_performance']['vehicles_not_satisfied'] for r in successful_results]),
                    'overall_full_satisfaction_rate_pct': 0
                }
            }

            total_vehicles = aggregated_stats['vehicle_satisfaction_summary']['total_vehicles_all_systems']
            total_fully_satisfied = aggregated_stats['vehicle_satisfaction_summary']['total_vehicles_fully_satisfied']
            if total_vehicles > 0:
                aggregated_stats['vehicle_satisfaction_summary']['overall_full_satisfaction_rate_pct'] = round((total_fully_satisfied / total_vehicles) * 100, 1)

        summary = {
            'model_info': model_info,
            'evaluation_metadata': {
                'evaluation_method': 'corrected_slot_by_slot_temporal_with_deception',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'systems_evaluated': len(successful_results),
                'systems_failed': len(failed_results),
                'total_systems': len(all_results)
            },
            'aggregated_performance': aggregated_stats,
            'system_results': all_results,
            'failed_systems': [{'system_id': r['system_id'], 'error': r['error']} for r in failed_results] if failed_results else []
        }

        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Results saved: {output_file}")

        if successful_results:
            print(f"SAVED SUMMARY:")
            print(f"Average satisfaction: {aggregated_stats['overall_performance']['avg_satisfaction_pct']:.1f}%")
            print(f"Satisfaction range: {aggregated_stats['overall_performance']['min_satisfaction_pct']:.1f}% - {aggregated_stats['overall_performance']['max_satisfaction_pct']:.1f}%")
            print(f"Total energy delivered: {aggregated_stats['energy_statistics']['total_energy_delivered_all_systems']:,.1f} kWh")
            print(f"Vehicles fully satisfied: {aggregated_stats['vehicle_satisfaction_summary']['total_vehicles_fully_satisfied']}/{aggregated_stats['vehicle_satisfaction_summary']['total_vehicles_all_systems']} ({aggregated_stats['vehicle_satisfaction_summary']['overall_full_satisfaction_rate_pct']:.1f}%)")
            print(f"Average capacity utilization: {aggregated_stats['resource_efficiency']['avg_capacity_utilization_pct']:.1f}%")

    def run_evaluation(self, max_models: int = None, max_systems: int = None):
        """Executes a complete evaluation with correction for slots without chargers.

        Args:
            max_models (int, optional): Maximum number of models to evaluate. Defaults to None (all models).
            max_systems (int, optional): Maximum number of systems to evaluate per model. Defaults to None (all systems).
        """

        print("="*60)
        print("CORRECTED SLOT-BY-SLOT TEMPORAL EVALUATION")
        print("With 'deception' to the model for slots without chargers")
        print("="*60)

        models = self.discover_models()
        systems = self.discover_systems()

        if max_models:
            models = models[:max_models]
        if max_systems:
            systems = systems[:max_systems]

        print(f"\nEvaluating {len(models)} models on {len(systems)} systems...")

        for i, model_info in enumerate(models, 1):
            print(f"\n{'-'*20}")
            print(f"MODEL {i}/{len(models)}: {model_info['archetype']}_rank_{model_info['rank']}")
            print(f"{'-'*20}")

            model_results = []

            for system_info in systems:
                result = self.evaluate_model_on_system(model_info, system_info)
                model_results.append(result)

            self.save_results(model_info, model_results)

            successful_results = [r for r in model_results if 'error' not in r]
            if successful_results:
                avg_assignment = np.mean([r['performance']['assignment_ratio_pct'] for r in successful_results])
                avg_time = np.mean([r['performance']['execution_time_seconds'] for r in successful_results])
                avg_schedule_entries = np.mean([r['performance']['schedule_entries'] for r in successful_results])
                avg_charging = np.mean([r['performance']['charging_assignments'] for r in successful_results])
                avg_parking = np.mean([r['performance']['parking_assignments'] for r in successful_results])

                print(f"\nMODEL SUMMARY:")
                print(f"Assignment ratio: {avg_assignment:.1f}%")
                print(f"Average time: {avg_time:.1f}s")
                print(f"Schedule entries: {avg_schedule_entries:.0f}")
                print(f"Charging assignments: {avg_charging:.0f}")
                print(f"Parking assignments: {avg_parking:.0f}")

        print(f"\n{'-'*20}")
        print("CORRECTED TEMPORAL EVALUATION COMPLETED")
        print(f"Results in: {self.output_dir}")
        print(f"{'-'*20}")


def main():
    """Main function to run the evaluation."""

    models_dir = "results/scatter_search/trained_models"
    systems_dir = "src/configs/system_data"
    output_dir = "results/scatter_search/corrected_temporal_evaluation"

    if not os.path.exists(models_dir):
        print(f"Models directory not found: {models_dir}")
        return

    if not os.path.exists(systems_dir):
        print(f"Systems directory not found: {systems_dir}")
        return

    simulator = EfficientSlotBySlotSimulator(models_dir, systems_dir, output_dir)

    print("DEBUG MODE WITH DECEPTION: 1 model x Specific System")

    systems = simulator.discover_systems()

    target_system_id = 3

    target_system = next((s for s in systems if s['system_id'] == target_system_id), None)

    if target_system:
        print(f"EVALUATING SYSTEM {target_system_id}:")
        print(f" {target_system['num_vehicles']} vehicles")
        print(f" {target_system['num_slots']} slots")
        print(f" {target_system['num_chargers']} chargers")
        print(f" Vehicle/charger ratio: {target_system['num_vehicles']/target_system['num_chargers']:.1f}")
        print(f" HIGH DEMAND - Should force usage of slots without chargers")

        models = simulator.discover_models()
        if models:
            model_info = models[0]
            print(f"Evaluating model: {model_info['archetype']}_rank_{model_info['rank']}")

            result = simulator.evaluate_model_on_system(model_info, target_system)

            print(f"\nSPECIFIC RESULT:")
            if 'error' not in result:
                print(f"Vehicles assigned: {result['vehicle_performance']['vehicles_assigned']}/{result['vehicle_performance']['vehicles_total']}")
                print(f"Overall satisfaction: {result['energy_performance']['overall_satisfaction_pct']:.1f}%")
                print(f"Vehicles fully satisfied: {result['vehicle_performance']['vehicles_fully_satisfied']}")
                print(f"Charging assignments: {result['vehicle_performance']['charging_assignments']}")
                print(f"Parking assignments: {result['vehicle_performance']['parking_assignments']}")
                print(f"Energy delivered: {result['energy_performance']['total_energy_delivered_kwh']:.1f} kWh")
                print(f"Total cost: ${result['economic_performance']['total_energy_cost']:.2f}")
                print(f"Charger utilization: {result['resource_utilization']['chargers_utilization_pct']:.1f}%")
                print(f"SUCCESS: Parking > 0 means deception worked!")
            else:
                print(f"ERROR: {result['error']}")

            print(f"\nSAVING RESULT...")
            simulator.save_results(model_info, [result])
            print(f"JSON saved successfully")

        else:
            print("No models found")
    else:
        print(f"System {target_system_id} not found")
        print("Available systems:")
        for s in systems:
            ratio = s['num_vehicles'] / s['num_chargers']
            print(f"System {s['system_id']}: {s['num_vehicles']}v, {s['num_slots']}s, {s['num_chargers']}c (ratio: {ratio:.1f})")


if __name__ == "__main__":
    main()