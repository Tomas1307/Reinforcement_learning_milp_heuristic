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
    Simulador eficiente que predice slot por slot de forma optimizada
    AHORA ITERA POR TODOS LOS TIMESTEPS
    """
    
    def __init__(self, models_dir: str, systems_dir: str, output_dir: str):
        self.models_dir = models_dir
        self.systems_dir = systems_dir
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"EfficientSlotBySlotSimulator initialized")
        print(f"  Models: {models_dir}")
        print(f"  Systems: {systems_dir}")
        print(f"  Output: {output_dir}")
    
    def discover_models(self) -> List[Dict[str, Any]]:
        """Descubre modelos disponibles"""
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
            print(f"  - {model['archetype']} rank {model['rank']}")
        
        return models
    
    def discover_systems(self) -> List[Dict[str, Any]]:
        """Descubre sistemas disponibles"""
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
            print(f"  - System {system['system_id']}: {system['num_vehicles']} vehicles, "
                  f"{system['num_slots']} slots, {system['num_chargers']} chargers")
        
        return systems
    
    def load_agent_once(self, model_path: str):
        """Carga el agente UNA SOLA VEZ"""
        print(f"Loading agent from: {os.path.basename(model_path)}")
        
        state_size = 40
        action_size = 60
        
        # Usar Simple DQN (no Dueling) para compatibilidad
        agent = EnhancedDQNAgentPyTorch(state_size, action_size, dueling_network=False)
        
        if agent.load(model_path):
            agent.epsilon = 0.0  # Sin exploración
            print("✅ Agent loaded successfully")
            return agent
        else:
            raise ValueError(f"No se pudo cargar modelo: {model_path}")
    
    def check_memory(self, stage=""):
        """Monitor de memoria"""
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
        print(f"   💾 {stage} - RAM: {memory_mb:.1f}MB, GPU: {gpu_memory:.1f}MB")
        return memory_mb

    def simulate_system_efficient(self, agent, system_config: Dict):
        """
        NUEVA VERSIÓN: Simula TODO EL HORIZONTE TEMPORAL
        Itera por todos los timesteps donde hay vehículos disponibles
        """
        print(f"\n🚀 SIMULACIÓN EFICIENTE - Sistema con {len(system_config['arrivals'])} vehículos")
        self.check_memory("Inicio simulación")
        
        # LÍMITE DE SEGURIDAD
        max_timesteps_to_process = 20  # Limitar a 20 timesteps máximo
        max_schedule_entries = 1000    # Limitar entradas de schedule
        
        all_schedule_entries = []  # Lista para TODAS las asignaciones temporales
        vehicle_assignment_history = set()  # Para tracking de vehículos ya asignados
        
        total_timesteps = len(system_config["times"])
        print(f"   Total timesteps disponibles: {total_timesteps}")
        
        # ITERAR POR TODOS LOS TIMESTEPS (CON LÍMITE)
        timesteps_processed = 0
        for timestep_idx, current_time in enumerate(system_config["times"]):
            
            # LÍMITES DE SEGURIDAD
            if timesteps_processed >= max_timesteps_to_process:
                print(f"   ⚠️ LÍMITE: Procesados {max_timesteps_to_process} timesteps, deteniendo")
                break
                
            if len(all_schedule_entries) >= max_schedule_entries:
                print(f"   ⚠️ LÍMITE: {max_schedule_entries} entradas de schedule, deteniendo")
                break
            
            # Obtener vehículos disponibles para ESTE timestep
            available_vehicles = []
            for arrival in system_config["arrivals"]:
                if (arrival["arrival_time"] <= current_time < arrival["departure_time"] and 
                    arrival["id"] not in vehicle_assignment_history):
                    available_vehicles.append(arrival["id"])
            
            if not available_vehicles:
                continue  # Skip timesteps sin vehículos nuevos
                
            timesteps_processed += 1
            print(f"   Timestep {timestep_idx} (t={current_time:.2f}h): {len(available_vehicles)} vehículos disponibles")
            self.check_memory(f"Timestep {timestep_idx}")
            
            # PREDICCIÓN SLOT POR SLOT PARA ESTE TIMESTEP
            timestep_assignments = self._predict_all_slots_for_timestep(
                agent, system_config, timestep_idx, available_vehicles
            )
            
            # Convertir asignaciones a formato schedule y marcar vehículos como asignados
            for assignment in timestep_assignments:
                schedule_entry = (
                    assignment['vehicle_id'],    # ev_id
                    timestep_idx,               # time_idx
                    assignment['charger'],       # charger_id (puede ser None)
                    assignment['slot'],         # slot
                    assignment['power']         # power
                )
                all_schedule_entries.append(schedule_entry)
                
                # Marcar vehículo como asignado para que no se reasigne
                vehicle_assignment_history.add(assignment['vehicle_id'])
            
            print(f"       ✅ Asignaciones en timestep {timestep_idx}: {len(timestep_assignments)}")
            print(f"       📊 Total vehículos asignados hasta ahora: {len(vehicle_assignment_history)}")
            
            # Limpieza periódica
            if timesteps_processed % 5 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # RESUMEN FINAL
        unique_vehicles = len(vehicle_assignment_history)
        total_vehicles = len(system_config["arrivals"])
        
        print(f"\n✅ SIMULACIÓN TEMPORAL COMPLETADA:")
        print(f"   Total entradas de schedule: {len(all_schedule_entries)}")
        print(f"   Vehículos únicos asignados: {unique_vehicles}/{total_vehicles} ({unique_vehicles/total_vehicles*100:.1f}%)")
        print(f"   Timesteps procesados: {timesteps_processed}/{total_timesteps}")
        self.check_memory("Fin simulación")
        
        return all_schedule_entries
    
    def _predict_all_slots_for_timestep(self, agent, system_config, timestep_idx, available_vehicles):
        """
        LÓGICA ORIGINAL DE PREDICCIÓN SLOT POR SLOT, pero para UN timestep específico
        """
        
        if not available_vehicles:
            return []
        
        # IDENTIFICAR SLOTS CON CARGADORES Y SIN CARGADORES
        slots_with_chargers = []
        slots_without_chargers = []
        
        # Asumir que cada cargador puede ir en cualquier slot
        for i, charger in enumerate(system_config["chargers"]):
            if i < system_config["n_spots"]:  # Solo si hay slots suficientes
                slots_with_chargers.append({
                    'slot': i,
                    'charger': charger.get("charger_id", i),
                    'power': charger.get("power", 7)
                })
        
        # Los slots restantes son sin cargador
        slots_used_for_chargers = len(slots_with_chargers)
        for slot_id in range(slots_used_for_chargers, system_config["n_spots"]):
            slots_without_chargers.append(slot_id)
        
        print(f"     Slots con cargador: {len(slots_with_chargers)}")
        print(f"     Slots sin cargador: {len(slots_without_chargers)}")
        
        # PREDICCIÓN EFICIENTE SLOT POR SLOT
        assignments = []
        working_vehicle_list = available_vehicles.copy()  # COPIA DE TRABAJO
        
        # PRIMERO: Slots con cargador (más valiosos)
        print(f"     🔌 Prediciendo slots CON cargador...")
        for i, slot_info in enumerate(slots_with_chargers):
            
            if not working_vehicle_list:
                break
                
            print(f"       Slot {slot_info['slot']} + Cargador {slot_info['charger']} ({i+1}/{len(slots_with_chargers)})")
            
            # Predecir para este slot específico
            selected_vehicle = self._predict_for_single_slot(
                agent, system_config, timestep_idx, 
                slot_info['slot'], working_vehicle_list, 
                charger_info=slot_info
            )
            
            if selected_vehicle is not None:
                # Crear asignación
                assignment = {
                    'slot': slot_info['slot'],
                    'vehicle_id': selected_vehicle,
                    'charger': slot_info['charger'],
                    'power': slot_info['power'],
                    'timestep': timestep_idx,
                    'type': 'charging'
                }
                assignments.append(assignment)
                
                # BORRAR vehículo de la lista de trabajo
                working_vehicle_list.remove(selected_vehicle)
                print(f"         ✅ Asignado: EV_{selected_vehicle}")
                print(f"         📋 Vehículos restantes: {len(working_vehicle_list)}")
            else:
                print(f"         ❌ No se seleccionó vehículo")
        
        # SEGUNDO: Slots sin cargador
        print(f"     🅿️ Prediciendo slots SIN cargador...")
        for i, slot_id in enumerate(slots_without_chargers):
            
            if not working_vehicle_list:
                break
                
            print(f"       Slot {slot_id} (solo parking) ({i+1}/{len(slots_without_chargers)})")
            
            # Predecir para este slot sin cargador
            selected_vehicle = self._predict_for_single_slot(
                agent, system_config, timestep_idx,
                slot_id, working_vehicle_list,
                charger_info=None
            )
            
            if selected_vehicle is not None:
                # Crear asignación
                assignment = {
                    'slot': slot_id,
                    'vehicle_id': selected_vehicle,
                    'charger': None,
                    'power': 0,
                    'timestep': timestep_idx,
                    'type': 'parking'
                }
                assignments.append(assignment)
                
                # BORRAR vehículo de la lista de trabajo
                working_vehicle_list.remove(selected_vehicle)
                print(f"         ✅ Asignado: EV_{selected_vehicle}")
                print(f"         📋 Vehículos restantes: {len(working_vehicle_list)}")
            else:
                print(f"         ❌ No se seleccionó vehículo")
        
        return assignments
    
    def _predict_for_single_slot(self, agent, system_config, timestep_idx, 
                                slot_id, available_vehicles, charger_info=None):
        """
        Predice qué vehículo asignar a UN SLOT ESPECÍFICO
        """
        
        if not available_vehicles:
            return None
        
        # Crear estado para esta predicción
        state = self._build_slot_state(
            system_config, timestep_idx, available_vehicles, slot_id, charger_info
        )
        
        # Generar acciones posibles para este slot
        possible_actions = self._generate_slot_actions(
            slot_id, available_vehicles, charger_info
        )
        
        print(f"         🔍 DEBUG - Vehículos: {available_vehicles}")
        print(f"         🔍 DEBUG - Acciones: {len(possible_actions)}")
        
        if len(possible_actions) <= 1:  # Solo no_action
            print(f"         🔍 DEBUG - Solo no_action disponible")
            return None
        
        # PREDICCIÓN DEL AGENTE
        
        print(f"         🔍 ANTES de agent.act():")
        print(f"           - state type: {type(state)}")
        print(f"           - possible_actions length: {len(possible_actions)}")
        print(f"           - agent epsilon: {agent.epsilon}")

        action_idx = agent.act(state, possible_actions, verbose=False)
        
        print(f"         🔍 DESPUÉS de agent.act(): {action_idx}")
        
        # DEBUG: Qué eligió
        print(f"         🔍 DEBUG - Acción elegida: {action_idx}/{len(possible_actions)-1}")
        if action_idx >= 0 and action_idx < len(possible_actions):
            print(f"         🔍 DEBUG - Acción detalle: {possible_actions[action_idx]}")
        
            
        
        if action_idx == -1 or action_idx >= len(possible_actions):
            return None
        
        selected_action = possible_actions[action_idx]
        
        # Verificar si es no_action
        if selected_action.get("type") == "no_action":
            return None
        
        print(f"         🔍 DEBUG - selected_action: {selected_action}")
        print(f"         🔍 DEBUG - vehicle_id: {selected_action.get('vehicle_id')}")

        # Retornar el vehículo seleccionado
        vehicle_id = selected_action.get("vehicle_id")
        print(f"         🔍 DEBUG - returning: {vehicle_id}")
        
        # Retornar el vehículo seleccionado
        return selected_action.get("vehicle_id")
    
    def _build_slot_state(self, system_config, timestep_idx, available_vehicles, slot_id, charger_info):
        """Construye estado para predicción de un slot específico"""
        
        current_time = system_config["times"][timestep_idx]
        
        # Seleccionar vehículo más urgente como representante
        representative_vehicle = self._select_most_urgent_vehicle(
            available_vehicles, system_config, current_time
        )
        
        state = {
            "evs_present": available_vehicles.copy(),
            "available_spots": [slot_id],  # Solo este slot
            "available_chargers": [charger_info['charger']] if charger_info else [],
            "current_time_idx": timestep_idx,
            "current_time_normalized": timestep_idx / len(system_config["times"]),
            "representative_ev": representative_vehicle
        }
        
        # Features del vehículo representante
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
        
        # Métricas básicas del sistema
        state.update({
            "avg_available_spots": 1.0 / system_config["n_spots"],
            "avg_available_chargers": (1 if charger_info else 0) / len(system_config["chargers"]),
            "min_price": 0.5,
            "avg_price": 0.5,
            "system_demand_ratio": len(available_vehicles) / len(system_config["arrivals"]),
            "competition_pressure": min(1.0, len(available_vehicles) / system_config["n_spots"])
        })
        
        print(f"         🔍 STATE DEBUG:")
        print(f"           - evs_present: {state.get('evs_present')}")
        print(f"           - available_spots: {state.get('available_spots')}")  
        print(f"           - available_chargers: {state.get('available_chargers')}")
        print(f"           - representative_ev: {state.get('representative_ev')}")
        print(f"           - ev_features length: {len(state.get('ev_features', []))}")

            
        return state
    
    def _select_most_urgent_vehicle(self, available_vehicles, system_config, current_time):
        """Selecciona vehículo más urgente"""
        
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
    
    def _generate_slot_actions(self, slot_id, available_vehicles, charger_info):
        """Genera acciones posibles para UN slot específico"""
        
        actions = [{"type": "no_action"}]
        
        for vehicle_id in available_vehicles:
            
            if charger_info:
                # Acción con carga
                actions.append({
                    "type": "assign_charging",
                    "vehicle_id": vehicle_id,
                    "slot": slot_id,
                    "charger": charger_info['charger'],
                    "power": charger_info['power']
                })
            else:
                # Solo parking
                actions.append({
                    "type": "assign_parking",
                    "vehicle_id": vehicle_id,
                    "slot": slot_id
                })
        print(f"         🔍 ACTIONS DEBUG:")
        for i, action in enumerate(actions):
            print(f"           [{i}]: {action}")
            
        return actions
    
    def evaluate_model_on_system(self, model_info: Dict, system_info: Dict):
        """Evalúa UN modelo en UN sistema usando el método eficiente TEMPORAL"""
        
        model_name = f"{model_info['archetype']}_rank_{model_info['rank']}"
        system_id = system_info['system_id']
        
        print(f"\n{'='*50}")
        print(f"EVALUATING: {model_name} on System {system_id}")
        print(f"{'='*50}")
        
        self.check_memory("Inicio evaluación")
        
        try:
            start_time = time()
            
            # CARGAR AGENTE UNA SOLA VEZ
            agent = self.load_agent_once(model_info['path'])
            self.check_memory("Agente cargado")
            
            # SIMULAR USANDO MÉTODO EFICIENTE TEMPORAL
            schedule_entries = self.simulate_system_efficient(agent, system_info['config'])
            self.check_memory("Simulación completada")
            
            # CALCULAR MÉTRICAS
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
                    'num_chargers': system_info['num_chargers']
                },
                'performance': {
                    'execution_time_seconds': round(execution_time, 3),
                    'vehicles_assigned': metrics['vehicles_assigned'],
                    'vehicles_total': metrics['total_vehicles'],
                    'assignment_ratio_pct': round(metrics['assignment_ratio'] * 100, 1),
                    'schedule_entries': len(schedule_entries),
                    'timesteps_used': metrics['timesteps_used'],
                    'charging_assignments': metrics['charging_assignments'],
                    'parking_assignments': metrics['parking_assignments']
                },
                'schedule_detail': schedule_entries  # Para compatibilidad con Gantt
            }
            
            print(f"\n✅ RESULTADO: {metrics['vehicles_assigned']}/{metrics['total_vehicles']} vehículos "
                  f"({metrics['assignment_ratio']*100:.1f}%) en {execution_time:.1f}s")
            print(f"   Schedule entries: {len(schedule_entries)} across {metrics['timesteps_used']} timesteps")
            
            return result
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
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
            # LIMPIEZA FORZADA
            if 'agent' in locals():
                del agent
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.check_memory("Limpieza final")
    
    def _calculate_metrics_from_schedule(self, schedule_entries, system_config):
        """Calcula métricas del schedule completo"""
        
        unique_vehicles = len(set(entry[0] for entry in schedule_entries))  # entry[0] = ev_id
        total_vehicles = len(system_config['arrivals'])
        unique_timesteps = len(set(entry[1] for entry in schedule_entries))  # entry[1] = time_idx
        
        charging_entries = len([entry for entry in schedule_entries if entry[2] is not None])  # entry[2] = charger_id
        parking_entries = len([entry for entry in schedule_entries if entry[2] is None])
        
        return {
            'vehicles_assigned': unique_vehicles,
            'total_vehicles': total_vehicles,
            'assignment_ratio': unique_vehicles / total_vehicles if total_vehicles > 0 else 0,
            'timesteps_used': unique_timesteps,
            'charging_assignments': charging_entries,
            'parking_assignments': parking_entries
        }
    
    def save_results(self, model_info: Dict, all_results: List[Dict]):
        """Guarda resultados"""
        
        model_name = f"{model_info['archetype']}_rank_{model_info['rank']}"
        output_file = os.path.join(self.output_dir, f"{model_name}_efficient_temporal_results.json")
        
        summary = {
            'model_info': model_info,
            'evaluation_method': 'efficient_slot_by_slot_temporal',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'systems_evaluated': len([r for r in all_results if 'error' not in r]),
            'systems_failed': len([r for r in all_results if 'error' in r]),
            'results': all_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved: {output_file}")
    
    def run_evaluation(self, max_models: int = None, max_systems: int = None):
        """Ejecuta evaluación completa"""
        
        print("="*60)
        print("EFFICIENT SLOT-BY-SLOT TEMPORAL EVALUATION")
        print("="*60)
        
        models = self.discover_models()
        systems = self.discover_systems()
        
        if max_models:
            models = models[:max_models]
        if max_systems:
            systems = systems[:max_systems]
        
        print(f"\nEvaluating {len(models)} models on {len(systems)} systems...")
        
        for i, model_info in enumerate(models, 1):
            print(f"\n{'🤖'*20}")
            print(f"MODEL {i}/{len(models)}: {model_info['archetype']}_rank_{model_info['rank']}")
            print(f"{'🤖'*20}")
            
            model_results = []
            
            for system_info in systems:
                result = self.evaluate_model_on_system(model_info, system_info)
                model_results.append(result)
            
            self.save_results(model_info, model_results)
            
            # Resumen del modelo
            successful_results = [r for r in model_results if 'error' not in r]
            if successful_results:
                avg_assignment = np.mean([r['performance']['assignment_ratio_pct'] for r in successful_results])
                avg_time = np.mean([r['performance']['execution_time_seconds'] for r in successful_results])
                avg_schedule_entries = np.mean([r['performance']['schedule_entries'] for r in successful_results])
                print(f"\n📊 MODEL SUMMARY: {avg_assignment:.1f}% avg assignment, {avg_time:.1f}s avg time, {avg_schedule_entries:.0f} avg schedule entries")
        
        print(f"\n{'✅'*20}")
        print("TEMPORAL EVALUATION COMPLETED")
        print(f"Results in: {self.output_dir}")
        print(f"{'✅'*20}")


def main():
    """Función principal"""
    
    models_dir = "results/scatter_search/trained_models"
    systems_dir = "src/configs/system_data"
    output_dir = "results/scatter_search/efficient_temporal_evaluation"
    
    if not os.path.exists(models_dir):
        print(f"Models directory not found: {models_dir}")
        return
    
    if not os.path.exists(systems_dir):
        print(f"Systems directory not found: {systems_dir}")
        return
    
    simulator = EfficientSlotBySlotSimulator(models_dir, systems_dir, output_dir)
    
    # EMPEZAR MUY CONSERVADOR
    print("🔬 MODO DEBUG: 1 modelo × 1 sistema pequeño")
    simulator.run_evaluation(max_models=1, max_systems=1)


if __name__ == "__main__":
    main()