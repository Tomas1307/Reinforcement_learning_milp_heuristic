import argparse
import json
import os
import numpy as np
import time
import sys
import re
from datetime import datetime
from collections import defaultdict

# Add the project_root to the Python path to allow absolute imports from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# Import modules from the new structure
from src.common.config import load_data, load_all_test_systems, load_system_config, load_hyperparameters
from src.dqn_agent.agent import EnhancedDQNAgentPyTorch as EnhancedDQNAgent
from src.dqn_agent.environment import EVChargingEnv
from src.dqn_agent.training import train_agent, train_dqn_agent
from src.milp_optimizer.optimizer import EVChargingMILP
from src.common.logger import SimpleEpisodeLogger
from src.common.visualize import plot_dqn_learning_progress, visualize_solution, visualize_milp_solution, visualize_rl_solution
from src.common.utils import ensure_directory_exists, get_timestamp_filepath


def map_hyperparameters(hyperparams):
    """
    Maps hyperparameter names from the YAML file to match the 
    EnhancedDQNAgentPyTorch constructor parameters.
    """
    mapped = {}
    
    # Direct and alternative mappings
    param_mappings = {
        'learning_rate': 'learning_rate',
        'gamma': 'gamma',
        'epsilon_start': 'epsilon',  # Map epsilon_start to epsilon
        'epsilon_min': 'epsilon_min',
        'epsilon_decay': 'epsilon_decay',
        'memory_size': 'memory_size',
        'batch_size': 'batch_size',
        'target_update_freq': 'target_update_freq',
        'target_update_frequency': 'target_update_freq',  # Alternative name
        'dueling_network': 'dueling_network',
        'dueling': 'dueling_network',  # Alternative name
        # Handle epsilon_end as alternative to epsilon_min
        'epsilon_end': 'epsilon_min'
    }
    
    for yaml_key, agent_key in param_mappings.items():
        if yaml_key in hyperparams:
            mapped[agent_key] = hyperparams[yaml_key]
    
    # Set default values for any missing required parameters
    defaults = {
        'learning_rate': 0.0005,
        'gamma': 0.95,
        'epsilon': 1.0,  # Use 1.0 as default to match common practice
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'memory_size': 5000,
        'batch_size': 32,
        'target_update_freq': 50,
        'dueling_network': True
    }
    
    for key, default_value in defaults.items():
        if key not in mapped:
            mapped[key] = default_value
    
    return mapped


def get_all_system_ids(data_dir: str) -> list[int]:
    """
    Escanea el directorio de datos para encontrar todos los system_ids disponibles
    basados en el patrón 'test_system_X.json'.
    """
    system_ids = []
    try:
        for filename in os.listdir(data_dir):
            match = re.match(r"test_system_(\d+)\.json", filename)
            if match:
                system_ids.append(int(match.group(1)))
        system_ids.sort()  # Ordenar los IDs para un procesamiento consistente
    except FileNotFoundError:
        print(f"Warning: Directory {data_dir} not found.")
    return system_ids

def generate_solution(config, agent=None, model_path="./ev_scheduler_model_pytorch.pt"):
    """
    Genera una solución usando el agente RL.
    """
    print("Generando solución RL...")

    # Cargar agente si no se proporciona
    if agent is None:
        print(f"Intentando cargar modelo desde {model_path}...")
        
        # Crear entorno temporal para obtener dimensiones
        temp_env = EVChargingEnv(config)
        state_size = 40  # Valor fijo basado en el main anterior
        action_size = 60  # Valor fijo basado en el main anterior

        agent = EnhancedDQNAgent(
            state_size=state_size, 
            action_size=action_size,
            dueling_network=True
        )

        try:
            model_loaded = agent.load(model_path)
            if not model_loaded:
                raise ValueError("El modelo no se pudo cargar correctamente.")
        except Exception as e:
            raise RuntimeError(f"❌ Error crítico al cargar el modelo desde {model_path}: {e}")

    # Crear entorno
    env = EVChargingEnv(config)
    state = env.reset()
    done = False

    # Configurar agente en modo evaluación
    original_epsilon = agent.epsilon
    agent.epsilon = 0.05  # Poca exploración para evaluación

    # Ejecutar episodio
    while not done:
        possible_actions = env._get_possible_actions(state)
        action = agent.act(state, possible_actions)

        if action == -1 or action >= len(possible_actions):
            break

        state, _, done = env.step(action)

    # Restaurar epsilon
    agent.epsilon = original_epsilon

    # Obtener schedule y métricas
    schedule_rl = env.get_schedule()
    energy_metrics = env.get_energy_satisfaction_metrics()

    # Calcular costo total
    total_cost = sum(
        config["prices"][time_idx] * config["dt"] * power
        for (_, time_idx, _, _, power) in schedule_rl
    )

    metrics = {
        "energy_metrics": energy_metrics,
        "total_cost": total_cost,
        "cost_per_kwh": total_cost / energy_metrics["total_delivered_energy"] if energy_metrics["total_delivered_energy"] > 0 else 0,
        "schedule_size": len(schedule_rl),
        "evs_served": len(set(entry[0] for entry in schedule_rl)),
        "total_evs": len(config["arrivals"])
    }

    print(f"Solución RL generada: {len(schedule_rl)} asignaciones para {metrics['evs_served']}/{metrics['total_evs']} EVs ({metrics['evs_served']/metrics['total_evs']*100:.2f}%)")
    print(f"Satisfacción energética: {energy_metrics['total_satisfaction_pct']:.2f}%")
    print(f"Costo total: ${total_cost:.2f}")

    return schedule_rl, metrics

def optimize_solution(config, rl_schedule, alpha_cost=0.6, alpha_satisfaction=0.4, time_limit=900):
    """
    Optimiza la solución RL usando MILP multiobjetivo.
    """
    print("\nOptimizando solución con MILP multiobjetivo...")
    print(f"Pesos: Costo ({alpha_cost:.2f}), Satisfacción ({alpha_satisfaction:.2f})")
    
    # Crear optimizador
    optimizer = EVChargingMILP(config)
    
    # Resolver con MILP
    start_time = time.time()
    model, schedule, rejected_details, obj_values = optimizer.solve(
        penalty_unmet=1000.0,
        rl_schedule=rl_schedule,
        time_limit=time_limit,
        epsilon_satisfaction=0.6,
        return_infeasible=True
    )

    solve_time = time.time() - start_time
    
    # Verificar si se encontró solución
    if model is None:
        print("No se pudo optimizar. Devolviendo solución RL original.")
        return None, {"error": "Optimización fallida"}
    
    # Convertir schedule a formato estándar lista de tuplas
    schedule_milp = []
    for ev_id, entries in schedule.items():
        for (start_time, end_time, charger_id, slot, power) in entries:
            # Encontrar índice de tiempo
            time_idx = None
            for idx, t in enumerate(config["times"]):
                if abs(t - start_time) < 1e-5:
                    time_idx = idx
                    break
            
            if time_idx is not None:
                schedule_milp.append((ev_id, time_idx, charger_id, slot, power))
    
    # Métricas de la solución MILP
    metrics = {
        "obj_values": obj_values,
        "rejected_details": rejected_details,
        "solve_time": solve_time,
        "schedule_size": len(schedule_milp),
        "evs_served": len(set(entry[0] for entry in schedule_milp)),
        "total_evs": len(config["arrivals"]),
        "milp_status": model.status if model else None,
        "milp_objective": model.objective.value() if model else None
    }
    
    # Imprimir resumen
    print(f"Solución MILP generada en {solve_time:.2f} segundos")
    print(f"Energía satisfecha: {obj_values['energy_satisfaction_pct']:.2f}%")
    print(f"Costo total: ${obj_values['energy_cost']:.2f}")
    print(f"Satisfacción ponderada: {obj_values['weighted_satisfaction']:.4f}")
    
    if rejected_details:
        print(f"Vehículos con energía no satisfecha: {len(rejected_details)}")
    
    return schedule_milp, metrics

def save_solution(schedule, metrics, filename):
    """
    Guarda la solución y métricas en un archivo JSON.
    """
    # Convertir a formato serializable
    serializable_schedule = []
    for entry in schedule:
        serializable_schedule.append(list(entry))
    
    # Crear objeto a guardar
    data = {
        "schedule": serializable_schedule,
        "metrics": metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Guardar en archivo
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Solución guardada en {filename}")

def compare_solutions(rl_metrics, milp_metrics):
    """
    Compara las soluciones RL y MILP.
    """
    print("\n=== COMPARACIÓN DE SOLUCIONES ===")
    
    # Satisfacción energética
    rl_satisfaction = rl_metrics["energy_metrics"]["total_satisfaction_pct"]
    milp_satisfaction = milp_metrics["obj_values"]["energy_satisfaction_pct"]
    satisfaction_diff = milp_satisfaction - rl_satisfaction
    
    print(f"Satisfacción energética:")
    print(f"  RL:   {rl_satisfaction:.2f}%")
    print(f"  MILP: {milp_satisfaction:.2f}% ({'+' if satisfaction_diff >= 0 else ''}{satisfaction_diff:.2f}%)")
    
    # Costo
    rl_cost = rl_metrics["total_cost"]
    milp_cost = milp_metrics["obj_values"]["energy_cost"]
    cost_diff_pct = ((milp_cost - rl_cost) / rl_cost) * 100 if rl_cost > 0 else 0
    
    print(f"Costo total:")
    print(f"  RL:   ${rl_cost:.2f}")
    print(f"  MILP: ${milp_cost:.2f} ({'+' if cost_diff_pct >= 0 else ''}{cost_diff_pct:.2f}%)")
    
    # Vehículos atendidos
    rl_evs = rl_metrics["evs_served"]
    milp_evs = milp_metrics["evs_served"]
    total_evs = rl_metrics["total_evs"]
    
    print(f"Vehículos atendidos:")
    print(f"  RL:   {rl_evs}/{total_evs} ({rl_evs/total_evs*100:.1f}%)")
    print(f"  MILP: {milp_evs}/{total_evs} ({milp_evs/total_evs*100:.1f}%)")

def run_scatter_search_optimization(args):
    """
    Ejecuta la optimización de hiperparámetros usando Scatter Search.
    """
    print("=== MODO SCATTER SEARCH OPTIMIZATION ===")
    
    # Importar módulos de scatter search
    try:
        from src.scatter_search.scatter_algorithm import ScatterSearchOptimizer
        from src.scatter_search.results_analyzer import ResultsAnalyzer
    except ImportError as e:
        print(f"Error al importar módulos de Scatter Search: {e}")
        print("Asegúrate de que todos los archivos de scatter_search están en src/scatter_search/")
        return
    
    # Configurar directorios
    scatter_output_dir = args.output_base_dir
    ensure_directory_exists(scatter_output_dir)
    
    # Cargar sistemas de datos
    print("Cargando sistemas de datos...")
    systems_data = load_all_test_systems(args.data_dir)
    
    if not systems_data:
        print("No se encontraron sistemas de datos para optimización.")
        return
    
    print(f"Sistemas cargados: {list(systems_data.keys())}")
    
    # Configuración del algoritmo
    scatter_config_path = args.scatter_config or "src/configs/scatter_search_config.yaml"
    
    if not os.path.exists(scatter_config_path):
        print(f"Archivo de configuración no encontrado: {scatter_config_path}")
        print("Creando configuración por defecto...")
        create_default_scatter_config(scatter_config_path)
    
    # Inicializar optimizador
    print(f"Inicializando Scatter Search con configuración: {scatter_config_path}")
    optimizer = ScatterSearchOptimizer(scatter_config_path, systems_data)
    
    # Ejecutar optimización
    print("Iniciando optimización de hiperparámetros...")
    start_time = time.time()
    
    try:
        results = optimizer.run_optimization()
        execution_time = time.time() - start_time
        
        print(f"\n✅ Optimización completada en {execution_time/3600:.2f} horas")
        print(f"Mejores soluciones encontradas: {len(results['best_solutions'])}")
        
        # Mostrar mejores resultados
        if results['best_solutions']:
            print("\n🏆 TOP 3 SOLUCIONES ENCONTRADAS:")
            for i, solution in enumerate(results['best_solutions'][:3]):
                archetype = solution.get('archetype', 'unknown')
                print(f"{i+1}. {archetype.upper()} - Fitness: {solution['fitness']:.2f}")
        
        # Analizar y exportar resultados
        print("\nAnalizando y exportando resultados...")
        analyzer = ResultsAnalyzer(scatter_output_dir)
        
        # Guardar resultados completos
        analyzer.save_complete_results(results)
        
        # Generar reporte
        analyzer.generate_summary_report(results)
        
        # Crear archivos de configuración
        if results['best_solutions']:
            analyzer.create_configuration_files(results['best_solutions'])
        
        # Generar visualizaciones
        analyzer.generate_visualizations(results)
        
        # Exportar a Excel
        analyzer.export_to_excel(results)
        
        # Crear paquete de deployment
        analyzer.create_deployment_package(results)
        
        print(f"\n📁 Todos los resultados guardados en: {scatter_output_dir}")
        print("📋 Revisa el archivo 'optimization_report.txt' para un resumen completo")
        print("🎯 Las configuraciones optimizadas están en 'configurations/'")
        print("📊 Las visualizaciones están en 'visualizations/'")
        
    except KeyboardInterrupt:
        print("\n⚠️ Optimización interrumpida por el usuario")
        print("Los resultados parciales pueden estar disponibles en checkpoints")
    except Exception as e:
        print(f"\n❌ Error durante la optimización: {e}")
        import traceback
        traceback.print_exc()

def create_default_scatter_config(config_path):
    """
    Crea un archivo de configuración por defecto para Scatter Search.
    """
    import yaml
    
    default_config = {
        'algorithm': {
            'population_size': 30,
            'ref_set_size': 10,
            'elite_count': 6,
            'diverse_count': 4,
            'max_iterations': 15,
            'max_time_hours': 8.0,
            'combination_probability': 0.7,
            'improvement_probability': 0.3
        },
        'evaluation': {
            'fast': {
                'systems': [1, 2, 3],
                'episodes': 20
            },
            'medium': {
                'systems': [1, 2, 3, 4, 5],
                'episodes': 30
            },
            'full': {
                'systems': [1, 2, 3, 4, 5, 6, 7],
                'episodes': 50
            }
        },
        'output': {
            'save_frequency': 2,
            'checkpoint_dir': './checkpoints/scatter_search'
        }
    }
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    print(f"Configuración por defecto creada en: {config_path}")

def main():
    parser = argparse.ArgumentParser(description="EV Charging Management System")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["train", "train_dqn", "solve", "optimize", "run_milp", "visualize_solution", "scatter_search"],
                        help="Operating mode")
    parser.add_argument("--data_dir", type=str, default="src/configs/system_data",
                        help="Directory containing system configuration files (JSON).")
    parser.add_argument("--output_base_dir", type=str, default="results",
                        help="Base directory for saving all outputs.")
    parser.add_argument("--system_id", type=int, default=1,
                        help="ID of the system configuration to use")
    parser.add_argument("--system", type=str, default=None, 
                        help="Sistema específico a procesar (para modos solve/optimize)")
    parser.add_argument("--all_systems", action="store_true",
                        help="Process all systems")
    parser.add_argument("--all", action="store_true", help="Alias para --all_systems")
    parser.add_argument("--model_path", type=str, default="./model/ev_scheduler_model_pytorch.pt",
                        help="Path to save/load the DQN model.")
    parser.add_argument("--hyperparameters_path", type=str, default="./src/configs/hyperparameters.yaml",
                        help="Path to the YAML file containing DQN hyperparameters.")
    parser.add_argument("--num_episodes", type=int, default=1000,
                        help="Number of training episodes for DQN")
    parser.add_argument("--episodes", type=int, default=30, 
                        help="Episodios por sistema para entrenamiento (modo train)")
    parser.add_argument("--log_frequency", type=int, default=50,
                        help="Frequency of logging episode data to console")
    parser.add_argument("--alpha_cost", type=float, default=0.6, 
                        help="Peso del objetivo de costo (0-1)")
    parser.add_argument("--alpha_satisfaction", type=float, default=0.4, 
                        help="Peso del objetivo de satisfacción (0-1)")
    parser.add_argument("--time_limit", type=int, default=900, 
                        help="Límite de tiempo para MILP (segundos)")
    parser.add_argument("--solution_to_visualize", type=str, default=None,
                        help="Path to a saved solution JSON file to visualize")
    
    # Argumentos específicos para Scatter Search
    parser.add_argument("--scatter_config", type=str, default=None,
                        help="Path to Scatter Search configuration YAML file")

    args = parser.parse_args()

    # Compatibilidad con argumentos
    if args.all:
        args.all_systems = True

    # Crear directorios necesarios
    os.makedirs(args.output_base_dir, exist_ok=True)

    # --- Load Hyperparameters (needed for DQN training) ---
    hyperparameters = {}
    if args.mode in ["train_dqn", "train"]:
        try:
            raw_hyperparameters = load_hyperparameters(args.hyperparameters_path)
            hyperparameters = map_hyperparameters(raw_hyperparameters)
            print(f"Hyperparameters loaded from: {args.hyperparameters_path}")
            print(f"Mapped parameters: {list(hyperparameters.keys())}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading hyperparameters: {e}")
            sys.exit(1)

    # --- Modes of Operation ---
    if args.mode == "scatter_search":
        run_scatter_search_optimization(args)
        
    elif args.mode == "train":
        print("=== MODO DE ENTRENAMIENTO ===")
        
        # Cargar todos los sistemas
        systems = load_all_test_systems(args.data_dir)
        
        if not systems:
            print("No se encontraron sistemas para entrenar.")
            return
        
        print(f"Se cargaron {len(systems)} sistemas de prueba.")
        
        # Configuración de entrenamiento
        train_config = {
            "state_size": 40,
            "action_size": 60,
            "batch_size": 64,                    
            "episodes_per_system": args.episodes,
            "checkpoint_frequency": 5,
            "checkpoint_dir": "./checkpoints",
            "model_path": args.model_path, 
            "patience": 8,
            "resume_from_checkpoint": True,
            "learning_rate": 0.0005,
            "gamma": 0.99,
            "epsilon": 0.9,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.995,
            "memory_size": 12000,
            "target_update_freq": 10,            
            "dueling_network": True
        }
        
        # Entrenar agente
        start_time = time.time()
        agent = train_agent(systems, train_config)
        train_time = time.time() - start_time
        
        print(f"\nEntrenamiento completado en {train_time:.2f} segundos.")
        print(f"Modelo guardado en: {args.model_path}")

    elif args.mode == "train_dqn":
        print("\n--- Starting DQN Training Mode ---")

        systems_to_train = []
        if args.all_systems:
            systems_to_train = get_all_system_ids(args.data_dir)
            if not systems_to_train:
                print(f"No test_system_*.json files found in {args.data_dir}. Exiting.")
                sys.exit(1)
            print(f"Found systems to train: {systems_to_train}")
        else:
            systems_to_train.append(args.system_id)

        for current_system_id in systems_to_train:
            print(f"\n--- Training DQN for System ID: {current_system_id} ---")
            
            # Define specific output subdirectories for the current system
            dqn_output_root = os.path.join(args.output_base_dir, f"dqn_agent_system_{current_system_id}")
            dqn_models_dir = os.path.join(dqn_output_root, "models")
            dqn_logs_dir = os.path.join(dqn_output_root, "logs")
            dqn_plots_dir = os.path.join(dqn_output_root, "plots")

            ensure_directory_exists(dqn_models_dir)
            ensure_directory_exists(dqn_logs_dir)
            ensure_directory_exists(dqn_plots_dir)

            # Load System Configuration
            system_config_filepath = os.path.join(args.data_dir, f"test_system_{current_system_id}.json")
            try:
                system_config = load_system_config(system_config_filepath)
                print(f"System configuration loaded from: {system_config_filepath}")
            except FileNotFoundError as e:
                print(f"Error: {e}. Skipping system {current_system_id}.")
                continue
            except KeyError as e:
                print(f"Error in system config file structure for system {current_system_id}: {e}. Skipping.")
                continue

            # Initialize Environment
            env = EVChargingEnv(system_config)
            
            # Usar valores fijos para compatibilidad
            state_size = 40
            action_size = 60
            
            # Initialize Agent with mapped hyperparameters
            print(f"Creating agent with hyperparameters: {hyperparameters}")
            agent = EnhancedDQNAgent(state_size, action_size, **hyperparameters)

            # Initialize Logger
            logger = SimpleEpisodeLogger(log_frequency=args.log_frequency)

            # Determine model save path for THIS system
            if args.model_path:
                base_model_name = os.path.basename(args.model_path).replace(".pt", "")
                model_save_path = os.path.join(dqn_models_dir, f"{base_model_name}_system_{current_system_id}.pt")
            else:
                model_save_path = get_timestamp_filepath(dqn_models_dir, prefix=f"dqn_model_system_{current_system_id}_", suffix=".pt")
            
            # Train the agent
            print(f"Training DQN agent for {args.num_episodes} episodes for system {current_system_id}...")
            episode_data_log = train_dqn_agent(
                agent, env, args.num_episodes, model_save_path=model_save_path, logger=logger
            )

            # Save episode log
            episode_log_filename = f"dqn_episode_log_system_{current_system_id}.json"
            episode_log_filepath = os.path.join(dqn_logs_dir, episode_log_filename)
            logger.save_episode_data(episode_log_filepath)

            # Plot learning progress
            plot_dqn_learning_progress(episode_data_log, save_dir=dqn_plots_dir, title=f"DQN Learning Progress - System {current_system_id}")
            plot_output_filename = "dqn_learning_progress.png"
            print(f"DQN learning progress plot saved to: {os.path.join(dqn_plots_dir, plot_output_filename)}")

            # Save final schedule
            final_schedule = env.get_schedule()
            if final_schedule:
                schedule_filename = f"dqn_final_schedule_system_{current_system_id}.json"
                schedule_filepath = os.path.join(dqn_logs_dir, schedule_filename)
                with open(schedule_filepath, 'w') as f:
                    json.dump(final_schedule, f, indent=4)
                print(f"Final DQN schedule saved to: {schedule_filepath}")

                # Visualize final DQN schedule
                dqn_schedule_plot_filename = f"dqn_final_schedule_system_{current_system_id}.png"
                visualize_rl_solution(final_schedule, system_config, output_dir=dqn_plots_dir, filename=dqn_schedule_plot_filename)
                print(f"Final DQN schedule visualization saved to: {os.path.join(dqn_plots_dir, dqn_schedule_plot_filename)}")
            else:
                print("No final DQN schedule to save or visualize for this system.")

    elif args.mode == "solve":
        print("=== MODO DE SOLUCIÓN RL ===")
        
        if args.all_systems:
            print("Procesando todos los sistemas con RL...")
            systems = load_all_test_systems(args.data_dir)
            for system_id, config in systems.items():
                print(f"\nSistema {system_id}")
                schedule_rl, rl_metrics = generate_solution(config, model_path=args.model_path)
                rl_metrics["execution_time"] = time.time()
                
                # Guardar solución
                save_solution(schedule_rl, rl_metrics, os.path.join(args.output_base_dir, f"rl_solution_{system_id}.json"))
                
                # Visualizar
                visualize_solution(schedule_rl, config, f"RL Sistema {system_id}: ")
            return
        
        # Cargar sistema específico
        config = None
        if args.system:
            try_paths = [
                args.system,
                os.path.join(args.data_dir, args.system),
                os.path.join(args.data_dir, f"test_system_{args.system}.json") if args.system.isdigit() else None
            ]
        else:
            try_paths = [os.path.join(args.data_dir, f"test_system_{args.system_id}.json")]
        
        for path in try_paths:
            if path is None:
                continue
            try:
                config = load_data(path)
                print(f"Sistema cargado: {path}")
                break
            except Exception as e:
                print(f"Error al cargar {path}: {e}")
        
        if config is None:
            print("No se pudo cargar el sistema.")
            return
        
        # Generar solución RL
        start_time = time.time()
        schedule_rl, metrics = generate_solution(config, model_path=args.model_path)
        solve_time = time.time() - start_time
        metrics["execution_time"] = solve_time
        
        # Guardar solución
        test_number = config.get("test_number", "unknown")
        output_file = os.path.join(args.output_base_dir, f"rl_solution_{test_number}.json")
        save_solution(schedule_rl, metrics, output_file)
        
        # Visualizar
        visualize_solution(schedule_rl, config, "RL: ")

    elif args.mode == "optimize":
        print("=== MODO DE OPTIMIZACIÓN RL+MILP ===")
        
        if args.all_systems:
            print("Procesando todos los sistemas con RL + MILP...")
            systems = load_all_test_systems(args.data_dir)
            for system_id, config in systems.items():
                print(f"\nSistema {system_id}")
                schedule_rl, rl_metrics = generate_solution(config, model_path=args.model_path)
                rl_metrics["execution_time"] = time.time()
                
                schedule_milp, milp_metrics = optimize_solution(
                    config,
                    schedule_rl,
                    alpha_cost=args.alpha_cost,
                    alpha_satisfaction=args.alpha_satisfaction,
                    time_limit=args.time_limit
                )
                
                if schedule_milp is not None:
                    compare_solutions(rl_metrics, milp_metrics)
                    save_solution(schedule_rl, rl_metrics, os.path.join(args.output_base_dir, f"rl_solution_{system_id}.json"))
                    save_solution(schedule_milp, milp_metrics, os.path.join(args.output_base_dir, f"milp_solution_{system_id}.json"))
                    visualize_rl_solution(schedule_rl, config, output_dir=args.output_base_dir, filename=f"rl_solution_{system_id}.png")
                    visualize_milp_solution(schedule_milp, config, output_dir=args.output_base_dir, filename=f"milp_solution_{system_id}.png")
                else:
                    print(f"Fallo en optimización para sistema {system_id}")
            return
        
        # Cargar sistema específico
        config = None
        if args.system:
            try_paths = [
                args.system,
                os.path.join(args.data_dir, args.system),
                os.path.join(args.data_dir, f"test_system_{args.system}.json") if args.system.isdigit() else None
            ]
        else:
            try_paths = [os.path.join(args.data_dir, f"test_system_{args.system_id}.json")]
        
        for path in try_paths:
            if path is None:
                continue
            try:
                config = load_data(path)
                print(f"Sistema cargado: {path}")
                break
            except Exception as e:
                print(f"Error al cargar {path}: {e}")
        
        if config is None:
            print("No se pudo cargar el sistema.")
            return
        
        # Generar solución RL inicial
        print("\n1. GENERANDO SOLUCIÓN RL INICIAL")
        start_time = time.time()
        schedule_rl, rl_metrics = generate_solution(config, model_path=args.model_path)
        rl_time = time.time() - start_time
        rl_metrics["execution_time"] = rl_time
        
        # Optimizar con MILP
        print("\n2. OPTIMIZANDO CON MILP MULTIOBJETIVO")
        schedule_milp, milp_metrics = optimize_solution(
            config, 
            schedule_rl, 
            alpha_cost=args.alpha_cost, 
            alpha_satisfaction=args.alpha_satisfaction,
            time_limit=args.time_limit
        )
        
        # Comparar soluciones
        if schedule_milp is not None:
            compare_solutions(rl_metrics, milp_metrics)
            
            # Guardar soluciones
            test_number = config.get("test_number", "unknown")
            rl_output = os.path.join(args.output_base_dir, f"rl_solution_{test_number}.json")
            milp_output = os.path.join(args.output_base_dir, f"milp_solution_{test_number}.json")
            
            save_solution(schedule_rl, rl_metrics, rl_output)
            save_solution(schedule_milp, milp_metrics, milp_output)
            
            # Visualizar
            visualize_solution(schedule_rl, config, "RL: ")
            visualize_solution(schedule_milp, config, "MILP: ")
        else:
            print("La optimización MILP no produjo una solución válida.")

    elif args.mode == "run_milp":
        print("\n--- Starting MILP Optimization Mode ---")
        
        # Output subdirectories for the current system
        milp_output_root = os.path.join(args.output_base_dir, f"milp_optimizer_system_{args.system_id}")
        milp_solutions_dir = os.path.join(milp_output_root, "solutions")
        milp_plots_dir = os.path.join(milp_output_root, "plots")
        
        ensure_directory_exists(milp_solutions_dir)
        ensure_directory_exists(milp_plots_dir)

        # Load System Configuration
        system_config_filepath = os.path.join(args.data_dir, f"test_system_{args.system_id}.json")
        try:
            system_config = load_system_config(system_config_filepath)
            print(f"System configuration loaded from: {system_config_filepath}")
        except FileNotFoundError as e:
            print(f"Error: {e}. Exiting MILP mode.")
            sys.exit(1)
        except KeyError as e:
            print(f"Error in system config file structure for system {args.system_id}: {e}. Exiting.")
            sys.exit(1)

        # Initialize MILP Optimizer
        milp_optimizer = EVChargingMILP(system_config)

        print("Solving MILP problem...")
        try:
            model, milp_schedule, rejected_details, obj_values = milp_optimizer.solve(
                penalty_unmet=1000.0,
                time_limit=300,
                epsilon_satisfaction=0.8
            )
            
            if milp_schedule:
                print("MILP solution found.")
                print(f"Objective values: {obj_values}")
                
                # Save MILP solution
                milp_solution_filename = f"milp_solution_system_{args.system_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                milp_solution_filepath = os.path.join(milp_solutions_dir, milp_solution_filename)
                solution_data = {
                    "schedule": milp_schedule,
                    "objective_values": obj_values,
                    "rejected_details": rejected_details,
                    "system_id": args.system_id,
                    "timestamp": datetime.now().isoformat()
                }
                with open(milp_solution_filepath, 'w') as f:
                    json.dump(solution_data, f, indent=4)
                print(f"MILP solution saved to: {milp_solution_filepath}")

                # Visualize MILP solution
                milp_plot_filename = f"milp_solution_schedule_system_{args.system_id}.png"
                visualize_milp_solution(milp_schedule, system_config, output_dir=milp_plots_dir, filename=milp_plot_filename)
                print(f"MILP solution visualization saved to: {os.path.join(milp_plots_dir, milp_plot_filename)}")
            else:
                print("No MILP solution found.")
                
        except Exception as e:
            print(f"Error solving MILP: {e}")
            sys.exit(1)

    elif args.mode == "visualize_solution":
        print("\n--- Starting Visualization Mode ---")
        
        # Output root for visualizations
        visualization_output_root = os.path.join(args.output_base_dir, "visualizations")
        visualize_plots_dir = os.path.join(visualization_output_root, "plots")
        ensure_directory_exists(visualize_plots_dir)

        if not args.solution_to_visualize:
            print("Error: Please provide --solution_to_visualize path for visualization mode.")
            sys.exit(1)

        solution_filepath = args.solution_to_visualize
        if not os.path.exists(solution_filepath):
            print(f"Error: Solution file not found at {solution_filepath}")
            sys.exit(1)

        try:
            with open(solution_filepath, 'r') as f:
                solution_data = json.load(f)

            schedule = solution_data.get("schedule")
            # If the loaded JSON is just the schedule list
            if not schedule and isinstance(solution_data, list):
                schedule = solution_data

            if not schedule:
                print("Error: 'schedule' key not found in the provided solution file or file format is incorrect.")
                sys.exit(1)

            # Try to infer system_id from the solution filename or content
            system_id_from_filename = None
            match = re.search(r"system_(\d+)", os.path.basename(solution_filepath))
            if match:
                system_id_from_filename = int(match.group(1))
            
            # Use provided system_id or try to infer from solution
            current_system_id = args.system_id
            if system_id_from_filename:
                current_system_id = system_id_from_filename

            # Load System Configuration using the inferred/provided system_id
            system_config_filepath = os.path.join(args.data_dir, f"test_system_{current_system_id}.json")
            try:
                system_config = load_system_config(system_config_filepath)
                print(f"System configuration loaded from: {system_config_filepath}")
            except FileNotFoundError as e:
                print(f"Error: {e}. Cannot load system config for visualization.")
                sys.exit(1)
            except KeyError as e:
                print(f"Error in system config file structure for system {current_system_id}: {e}. Cannot load config for visualization.")
                sys.exit(1)

            output_filename_base = os.path.basename(solution_filepath).replace(".json", "")
            
            if isinstance(schedule, list) and all(isinstance(item, (list, tuple)) and len(item) >= 4 for item in schedule):
                print("Visualizing DQN-style schedule.")
                output_filename = f"{output_filename_base}_viz_dqn.png"
                visualize_rl_solution(schedule, system_config, output_dir=visualize_plots_dir, filename=output_filename)
            elif isinstance(schedule, dict) and all(isinstance(v, list) for v in schedule.values()):
                print("Visualizing MILP-style schedule.")
                output_filename = f"{output_filename_base}_viz_milp.png"
                visualize_milp_solution(schedule, system_config, output_dir=visualize_plots_dir, filename=output_filename)
            else:
                print("Unknown schedule format. Cannot visualize.")
                sys.exit(1)

            print(f"Solution visualization saved to {os.path.join(visualize_plots_dir, output_filename)}")

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {solution_filepath}: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred during visualization: {e}")
            sys.exit(1)

    else:
        print(f"Modo no reconocido: {args.mode}. Use 'train', 'train_dqn', 'solve', 'optimize', 'run_milp', 'visualize_solution', or 'scatter_search'.")

if __name__ == "__main__":
    main()