import argparse
import json
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import defaultdict

# Importar nuestros módulos
from DQN_agent import EnhancedDQNAgentPyTorch as EnhancedDQNAgent
from MILP_optimizer import EVChargingMILP
from EV_env import EnhancedEVChargingEnv
from train_logger import SystemProgressLogger, create_episode_data

def load_data(json_path):
    """
    Carga el archivo JSON y retorna los datos del sistema.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    energy_prices = sorted(data["energy_prices"], key=lambda x: x["time"])
    times = [ep["time"] for ep in energy_prices]
    prices = [ep["price"] for ep in energy_prices]

    arrivals = sorted(data["arrivals"], key=lambda x: x["id"])
    parking_config = data["parking_config"]
    chargers = parking_config["chargers"]
    station_limit = parking_config["transformer_limit"]
    n_spots = parking_config["n_spots"]

    if len(times) > 1:
        dt = times[1] - times[0]
    else:
        dt = 0.25

    return {
        "times": times,
        "prices": prices,
        "arrivals": arrivals,
        "chargers": chargers,
        "station_limit": station_limit,
        "dt": dt,
        "n_spots": n_spots,
        "test_number": data.get("test_number", 0),
        "parking_config": parking_config,
        "car_brands": data.get("car_brands", []),
        "charger_types": data.get("charger_types", {})
    }

def load_all_test_systems(data_dir="./data"):
    """
    Carga todos los sistemas de prueba disponibles en el directorio de datos.
    Asume que cada archivo tiene un test_number único (ej. test_system_1.json).
    """
    systems = {}
    json_files = []

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.json') and 'test_system' in file:
                json_files.append(os.path.join(root, file))

    json_files.sort()

    for json_file in json_files:
        try:
            config = load_data(json_file)
            test_number = config.get("test_number")
            if test_number is None:
                raise ValueError(f"Archivo {json_file} no contiene 'test_number'")
            if test_number in systems:
                raise ValueError(f"Duplicado de ID detectado: {test_number} en {json_file}")
            systems[test_number] = config
            print(f"Sistema {test_number} cargado: {len(config['arrivals'])} vehículos, {config['n_spots']} plazas, {len(config['chargers'])} cargadores")
        except Exception as e:
            print(f"Error al cargar {json_file}: {e}")

    print(f"\nTotal: {len(systems)} sistemas únicos cargados.")
    return systems


# Solo muestro la función train_agent modificada - el resto del archivo main.py queda igual

def train_agent(systems, config):
    """
    Entrena un agente RL con múltiples sistemas.
    OPTIMIZADO: Configuración balanceada para mejor aprendizaje.
    """
    import time
    from datetime import datetime
    
    # Configuración del agente optimizada
    state_size = config.get("state_size", 32)
    action_size = config.get("action_size", 50)  # REDUCIDO de 150 a 50
    batch_size = config.get("batch_size", 32)    # REDUCIDO de 64 a 32
    episodes_per_system = config.get("episodes_per_system", 15)  # REDUCIDO de 30 a 15
    checkpoint_frequency = config.get("checkpoint_frequency", 5)
    checkpoint_dir = config.get("checkpoint_dir", "./checkpoints")
    model_path = config.get("model_path", "./ev_scheduler_model_pytorch.pt")
    patience = config.get("patience", 3)  # REDUCIDO de 5 a 3
    resume_from_checkpoint = config.get("resume_from_checkpoint", True)
    
    # Crear directorio de checkpoints si no existe
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    progress_logger = SystemProgressLogger()
    
    print(f"\nINICIANDO ENTRENAMIENTO DQN OPTIMIZADO")
    print(f"Configuración:")
    print(f"   - State size: {state_size}")
    print(f"   - Action size: {action_size}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Episodios por sistema: {episodes_per_system}")
    print(f"   - Total de sistemas: {len(systems)}")
    print(f"   - Total episodios estimados: {len(systems) * episodes_per_system}")
    
    # Crear agente avanzado con configuración optimizada
    print(f"\nCreando agente DQN...")
    agent = EnhancedDQNAgent(
    state_size=state_size, 
    action_size=action_size,
    learning_rate=config.get("learning_rate", 0.001),
    gamma=config.get("gamma", 0.95),
    epsilon=config.get("epsilon", 0.9),
    epsilon_min=config.get("epsilon_min", 0.05),
    epsilon_decay=config.get("epsilon_decay", 0.995),
    memory_size=config.get("memory_size", 5000),
    batch_size=batch_size,
    dueling_network=config.get("dueling_network", True),
    target_update_freq=config.get("target_update_freq", 30)
)
    print(f"Agente creado exitosamente")
    
    # Orden de entrenamiento: sistemas más simples primero
    system_order = sorted(systems.keys())
    
    # Resumir entrenamiento desde checkpoint si existe
    current_system_idx = 0
    current_episode = 0
    
    if resume_from_checkpoint:
        print(f"\nBuscando checkpoints...")
        checkpoint_files = []
        for root, _, files in os.walk(checkpoint_dir):
            for file in files:
                if file.endswith('.h5') and 'checkpoint' in file:
                    checkpoint_files.append(os.path.join(root, file))
        
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: os.path.getmtime(x))
            latest_checkpoint = checkpoint_files[-1]
            
            try:
                filename = os.path.basename(latest_checkpoint)
                parts = filename.split('_')
                current_system_idx = int(parts[2])
                current_episode = int(parts[4].split('.')[0])
                
                print(f"Encontrado checkpoint: {latest_checkpoint}")
                print(f"Reanudando desde sistema {current_system_idx}, episodio {current_episode}")
                
                agent.load(latest_checkpoint)
                current_episode += 1
                
                if current_episode >= episodes_per_system:
                    current_system_idx += 1
                    current_episode = 0
            except Exception as e:
                print(f"Error al analizar checkpoint, iniciando desde cero: {e}")
                current_system_idx = 0
                current_episode = 0
        else:
            print(f"No se encontraron checkpoints")
    
    # Si no se encontró checkpoint, intentar cargar modelo pre-entrenado
    if current_system_idx == 0 and current_episode == 0:
        try:
            print(f"Intentando cargar modelo pre-entrenado desde {model_path}...")
            model_loaded = agent.load(model_path)
            if model_loaded:
                print("Modelo pre-entrenado cargado exitosamente.")
                agent.epsilon = 0.5  # Epsilon intermedio para modelo pre-entrenado
                print(f"Epsilon ajustado a {agent.epsilon}")
        except Exception as e:
            print(f"No se pudo cargar modelo pre-entrenado: {e}")
            print("Entrenando nuevo modelo desde cero...")
    
    # Archivo de registro para el progreso
    log_path = os.path.join(checkpoint_dir, "training_log.txt")
    with open(log_path, "a") as log_file:
        log_file.write(f"\n\n--- Nueva sesión de entrenamiento optimizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        
        # Variables para estadísticas globales
        global_start_time = time.time()
        total_episodes_completed = 0
        
        # Entrenar en cada sistema, comenzando desde el índice actual
        for system_idx in range(current_system_idx, len(system_order)):
            system_num = system_order[system_idx]
            system_config = systems[system_num]
            
            log_message = f"\n============================================================\nSISTEMA {system_num} ({system_idx+1}/{len(systems)})\n============================================================"
            print(log_message)
            log_file.write(f"{log_message}\n")
            
            system_info = f"Info del sistema: {len(system_config['arrivals'])} vehículos, {system_config['n_spots']} plazas, {len(system_config['chargers'])} cargadores"
            print(system_info)
            log_file.write(f"{system_info}\n")
            
            # Crear entorno avanzado
            print(f"Creando entorno para sistema {system_num}...")
            env = EnhancedEVChargingEnv(system_config)
            print(f"Entorno creado")
            
            agent.epsilon = config.get("epsilon", 0.9)
            print(f"Epsilon reiniciado a {agent.epsilon}")
            
            # Comenzar desde el episodio actual o desde 0
            start_episode = current_episode if system_idx == current_system_idx else 0
            
            log_message = f"Ejecutando episodios {start_episode+1}-{episodes_per_system}..."
            print(log_message)
            log_file.write(f"{log_message}\n")
            
            # Variables para early stopping
            best_reward = -float('inf')
            no_improvement_count = 0
            best_model_path = os.path.join(checkpoint_dir, f"best_model_sys_{system_idx}.h5")
            
            # Variables para estadísticas del sistema
            system_start_time = time.time()
            episode_times = []
            rewards_history = []
            satisfaction_history = []
            
            # Lista para el logger
            episodes_data = []
            
            for e in range(start_episode, episodes_per_system):
                episode_start_time = time.time()
                
                print(f"\nEpisodio {e+1}/{episodes_per_system} (Sistema {system_num})")
                print(f"   Epsilon actual: {agent.epsilon:.4f}")
                print(f"   Memoria: {len(agent.memory)}/{agent.memory.maxlen}")
                
                state = env.reset()
                total_reward = 0
                done = False
                step_count = 0
                
                # Contadores para diagnóstico
                skip_count = 0
                assign_count = 0
                decisions_made = 0
                total_vehicles = len(env.arrivals)
                
                print(f"   Procesando {total_vehicles} vehículos...", end="", flush=True)
                
                while not done:
                    # Mostrar progreso cada 20 vehículos
                    if decisions_made > 0 and decisions_made % 20 == 0:
                        progress = (decisions_made / total_vehicles) * 100
                        print(f" {progress:.0f}%", end="", flush=True)
                    
                    possible_actions = env._get_possible_actions(state)
                    
                    if len(possible_actions) == 0:
                        print(f"\n   No hay acciones posibles para el vehículo actual")
                        break
                        
                    # Solo mostrar detalles de acción cada 50 vehículos para no saturar
                    verbose_action = (decisions_made % 50 == 0)
                    action = agent.act(state, possible_actions, verbose=verbose_action)
                    
                    if action == -1:
                        print(f"\n   Agente no pudo seleccionar acción válida")
                        break
                        
                    selected_action = possible_actions[action]
                    if selected_action["skip"]:
                        skip_count += 1
                    else:
                        assign_count += 1
                        
                    next_state, reward, done = env.step(action)
                    agent.remember(state, action, reward, next_state, done, system_type=system_num)
                    
                    state = next_state
                    total_reward += reward
                    step_count += 1
                    decisions_made += 1
                    
                    # Entrenar con experiencia con mayor frecuencia
                    if len(agent.memory) > batch_size and decisions_made % 10 == 0:
                        agent.replay()
                
                print(f" Completado")
                
                # Métricas de satisfacción
                energy_metrics = env.get_energy_satisfaction_metrics()
                satisfaction_pct = energy_metrics["total_satisfaction_pct"]
                
                episode_time = time.time() - episode_start_time
                episode_times.append(episode_time)
                rewards_history.append(total_reward)
                satisfaction_history.append(satisfaction_pct)
                
                # CORREGIDO: Guardar datos del episodio para el logger
                episode_data = create_episode_data(
                    e + 1, total_reward, energy_metrics, assign_count, skip_count,
                    total_vehicles, episode_time, agent.epsilon
                )
                episodes_data.append(episode_data)
                
                # Calcular tendencias
                avg_reward_last_3 = np.mean(rewards_history[-3:]) if len(rewards_history) >= 3 else total_reward
                avg_satisfaction_last_3 = np.mean(satisfaction_history[-3:]) if len(satisfaction_history) >= 3 else satisfaction_pct
                
                # Estadísticas del episodio
                log_message = (f"Episodio {e+1} completado:\n"
                             f"   Recompensa: {total_reward:.2f} (promedio últimos 3: {avg_reward_last_3:.2f})\n"
                             f"   Satisfacción energética: {satisfaction_pct:.1f}% (promedio últimos 3: {avg_satisfaction_last_3:.1f}%)\n"
                             f"   Vehículos asignados: {assign_count}/{total_vehicles} ({assign_count/total_vehicles*100:.1f}%)\n"
                             f"   Vehículos saltados: {skip_count}/{total_vehicles} ({skip_count/total_vehicles*100:.1f}%)\n"
                             f"   Tiempo: {episode_time:.1f}s\n"
                             f"   Pasos: {step_count}")
                
                print(log_message)
                log_file.write(f"{log_message}\n")
                
                # Entrenar intensivamente al final del episodio
                if len(agent.memory) > batch_size:
                    print(f"   Entrenando red neuronal...")
                    for _ in range(5):  # Múltiples entrenamientos al final
                        agent.replay()
                
                # Guardar checkpoint periódicamente
                if (e + 1) % checkpoint_frequency == 0 or e == episodes_per_system - 1:
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_sys_{system_idx}_ep_{e+1}.h5")
                    print(f"Guardando checkpoint: {checkpoint_path}")
                    agent.save(checkpoint_path)
                    
                    # También actualizar modelo principal
                    agent.save(model_path)
                    log_file.write(f"Checkpoint guardado: {checkpoint_path}\n")
                    log_file.flush()
                
                # Early stopping basado en recompensa Y satisfacción
                # Calcular métricas móviles
                avg_reward_last_5 = np.mean(rewards_history[-5:]) if len(rewards_history) >= 5 else total_reward
                avg_satisfaction_last_5 = np.mean(satisfaction_history[-5:]) if len(satisfaction_history) >= 5 else satisfaction_pct
                combined_metric = avg_reward_last_5 + avg_satisfaction_last_5

                # Verificamos mejora significativa
                improvement_threshold = 10.0  # Umbral mínimo para considerar mejora útil
                NEGATIVE_REWARD_THRESHOLD = -5000
                
                if combined_metric > best_reward + improvement_threshold:
                    best_reward = combined_metric
                    no_improvement_count = 0
                    agent.save(best_model_path)
                    improvement_msg = f"Nuevo mejor modelo! Métrica combinada: {combined_metric:.2f}"
                    print(improvement_msg)
                    log_file.write(f"{improvement_msg}\n")
                else:
                    no_improvement_count += 1
                    patience_msg = f"Sin mejora. Paciencia: {no_improvement_count}/{patience}"
                    print(patience_msg)
                    log_file.write(f"{patience_msg}\n")

                    # Detención anticipada por bajo rendimiento
                    if len(rewards_history) >= 5 and all(r < NEGATIVE_REWARD_THRESHOLD for r in rewards_history[-5:]):
                        early_stop_msg = "Detención anticipada: todas las recompensas recientes son muy negativas."
                        print(early_stop_msg)
                        log_file.write(f"{early_stop_msg}\n")
                        break

                    if no_improvement_count >= patience:
                        early_stop_msg = f"Early stopping tras {patience} episodios sin mejora"
                        print(early_stop_msg)
                        log_file.write(f"{early_stop_msg}\n")

                        # Cargar el mejor modelo
                        agent.load(best_model_path)
                        print(f"Cargado mejor modelo desde {best_model_path}")
                        break

                
                total_episodes_completed += 1
                
                # Estimación de tiempo restante
                if len(episode_times) > 2:
                    avg_episode_time = np.mean(episode_times[-3:])
                    remaining_episodes = (len(systems) - system_idx - 1) * episodes_per_system + (episodes_per_system - e - 1)
                    estimated_remaining = remaining_episodes * avg_episode_time
                    
                    eta_msg = f"ETA: {estimated_remaining/60:.1f} minutos restantes"
                    print(eta_msg)
            
            # CORREGIDO: Guardar progreso del sistema ANTES del resumen
            system_info_for_logger = {
                "total_vehicles": len(system_config["arrivals"]),
                "n_spots": system_config["n_spots"],
                "n_chargers": len(system_config["parking_config"]["chargers"])
            }
            progress_logger.save_system_progress(system_num, episodes_data, system_info_for_logger)
            
            # Estadísticas del sistema completado
            system_time = time.time() - system_start_time
            avg_reward = np.mean(rewards_history) if rewards_history else 0
            avg_satisfaction = np.mean(satisfaction_history) if satisfaction_history else 0
            best_satisfaction = max(satisfaction_history) if satisfaction_history else 0
            
            system_summary = f"""
RESUMEN SISTEMA {system_num}:
   Tiempo total: {system_time/60:.1f} minutos
   Recompensa promedio: {avg_reward:.2f}
   Mejor métrica combinada: {best_reward:.2f}
   Satisfacción promedio: {avg_satisfaction:.1f}%
   Mejor satisfacción: {best_satisfaction:.1f}%
   Episodios completados: {len(rewards_history)}
   Epsilon final: {agent.epsilon:.4f}
"""
            print(system_summary)
            log_file.write(f"{system_summary}\n")
            
            # Reset current_episode para el siguiente sistema
            current_episode = 0
    
    # Estadísticas finales
    total_time = time.time() - global_start_time
    final_summary = f"""
ENTRENAMIENTO COMPLETADO!
Tiempo total: {total_time/60:.1f} minutos
Episodios totales: {total_episodes_completed}
Memoria final: {len(agent.memory)} experiencias
Epsilon final: {agent.epsilon:.4f}
Modelo guardado en: {model_path}
"""
    print(final_summary)
    
    # Guardar modelo final
    print(f"\nGuardando modelo final en {model_path}...")
    agent.save(model_path)
    print(f"Modelo guardado exitosamente")
    
    return agent

def generate_solution(config, agent=None, model_path="./ev_scheduler_model_pytorch.pt"):
    """
    Genera una solución usando el agente RL.

    Args:
        config: Configuración del sistema
        agent: Agente RL pre-entrenado (opcional)
        model_path: Ruta al modelo pre-entrenado

    Returns:
        schedule_rl: Lista de tuples (ev_id, t_slot, charger_id, slot, power)
        metrics: Métricas de la solución
    """
    print("Generando solución RL...")

    # Cargar agente si no se proporciona
    if agent is None:
        print(f"Intentando cargar modelo desde {model_path}...")
        state_size = 32
        action_size = 150

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
            raise RuntimeError(f"❌ Error crítico al cargar el modelo desde {model_path}: {e}\n\nAsegúrate de que el modelo fue entrenado con state_size={state_size}, action_size={action_size}, y coincide con esta arquitectura.")

    # Crear entorno
    env = EnhancedEVChargingEnv(config)
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
    
    Args:
        config: Configuración del sistema
        rl_schedule: Solución inicial RL
        alpha_cost: Peso del objetivo de costo (0-1)
        alpha_satisfaction: Peso del objetivo de satisfacción (0-1)
        time_limit: Límite de tiempo para la optimización
        
    Returns:
        schedule_milp: Solución optimizada
        metrics: Métricas detalladas
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

def visualize_solution(schedule, config, title_prefix="",show_plot=True):
    """
    Visualiza la solución de carga con gráficos.
    
    Args:
        schedule: Lista de tuplas (ev_id, time_idx, charger_id, slot, power) o
                 dict {ev_id: [(t_start, t_end, charger_id, slot, power), ...]}
        config: Configuración del sistema
        title_prefix: Prefijo para los títulos de los gráficos
    """
    # Convertir a formato de diccionario si es una lista
    schedule_dict = schedule
    if isinstance(schedule, list):
        schedule_dict = defaultdict(list)
        for (ev_id, time_idx, charger_id, slot, power) in schedule:
            t_start = config["times"][time_idx]
            t_end = t_start + config["dt"]
            schedule_dict[ev_id].append((t_start, t_end, charger_id, slot, power))
    
    # --- Gráfico de perfiles de carga ---
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 1, 1)
    
    # Usar una paleta de colores con suficientes colores
    num_evs = len(schedule_dict)
    cmap = plt.cm.get_cmap('tab20', 20)  # 20 colores que se repetirán
    
    # Ordenar EVs por ID para consistencia
    sorted_evs = sorted(schedule_dict.keys())
    
    # Para cada EV, graficar su perfil de carga
    for idx, ev_id in enumerate(sorted_evs):
        intervals = schedule_dict[ev_id]
        
        # Ordenar intervalos por tiempo
        intervals_sorted = sorted(intervals, key=lambda x: x[0])
        
        # Construir perfil escalonado
        times = []
        powers = []
        
        if intervals_sorted:
            # Empezar con potencia 0
            times.append(intervals_sorted[0][0])
            powers.append(0)
            
            # Agregar cada intervalo
            for interval in intervals_sorted:
                t_start, t_end, _, _, power = interval
                
                # Añadir cambio de potencia al inicio
                times.append(t_start)
                powers.append(power)
                
                # Añadir fin del intervalo
                times.append(t_end)
                powers.append(0)
            
            # Graficar perfil escalonado
            plt.step(times, powers, where='post', label=f'EV {ev_id}', 
                    color=cmap(idx % 20))
    
    # Precios de energía en eje secundario
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(config["times"], config["prices"], 'r-', alpha=0.5, label='Precio energía')
    
    # Configurar ejes y leyendas
    ax1.set_xlabel('Tiempo (horas)')
    ax1.set_ylabel('Potencia (kW)')
    ax2.set_ylabel('Precio ($/kWh)', color='r')
    
    plt.title(f'{title_prefix}Perfiles de carga de vehículos eléctricos ({num_evs} EVs)')
    
    # Manejar leyenda según cantidad de EVs
    if num_evs <= 20:
        ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')
    else:
        # Para muchos EVs, omitir leyenda individual
        ax1.text(0.99, 0.99, f'Total: {num_evs} EVs', 
                transform=ax1.transAxes, ha='right', va='top',
                bbox=dict(facecolor='white', alpha=0.8))
    
    # --- Gráfico de asignación de spots ---
    plt.subplot(2, 1, 2)
    
    # Tiempo total
    total_time = config["times"][-1] + config["dt"]
    n_spots = config["n_spots"]
    
    # Crear mapa de asignación por slot
    slot_assignments = {}
    for ev_id, intervals in schedule_dict.items():
        for interval in intervals:
            t_start, t_end, _, slot, _ = interval
            if slot not in slot_assignments:
                slot_assignments[slot] = []
            slot_assignments[slot].append((t_start, t_end, ev_id))
    
    # Graficar cada slot
    for slot in range(n_spots):
        if slot in slot_assignments:
            for t_start, t_end, ev_id in slot_assignments[slot]:
                plt.barh(slot, t_end - t_start, left=t_start, height=0.8, 
                         color=cmap(sorted_evs.index(ev_id) % 20), alpha=0.8)
                
                # Añadir etiqueta si hay espacio
                if t_end - t_start > 0.5:
                    plt.text(t_start + (t_end - t_start) / 2, slot, f'EV {ev_id}', 
                            ha='center', va='center', color='white', fontsize=8)
    
    plt.xlabel('Tiempo (horas)')
    plt.ylabel('Plaza de parqueo')
    plt.title(f'{title_prefix}Asignación de plazas de parqueo')
    plt.yticks(range(n_spots))
    plt.xlim(0, total_time)
    plt.ylim(-0.5, n_spots - 0.5)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Guardar gráficos
    import re

    def sanitize_filename(name):
        """Convierte string en nombre de archivo seguro."""
        return re.sub(r'[^\w\-_.]', '_', name.strip().lower())

    filename = sanitize_filename(f"{title_prefix}_solution_visualization.png")
    save_path = os.path.join(config.get("output_dir", "."), filename)
    plt.savefig(save_path)
    print(f"Gráfica guardada en: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

def save_solution(schedule, metrics, filename):
    """
    Guarda la solución y métricas en un archivo JSON.
    
    Args:
        schedule: Lista de tuplas (ev_id, time_idx, charger_id, slot, power)
        metrics: Diccionario con métricas
        filename: Nombre del archivo
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
    
    Args:
        rl_metrics: Métricas de la solución RL
        milp_metrics: Métricas de la solución MILP
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
    
    # Costo por kWh
    rl_cost_per_kwh = rl_metrics["cost_per_kwh"]
    milp_cost_per_kwh = milp_cost / milp_metrics["obj_values"]["total_energy_delivered"] if milp_metrics["obj_values"]["total_energy_delivered"] > 0 else 0
    
    print(f"Costo por kWh:")
    print(f"  RL:   ${rl_cost_per_kwh:.4f}")
    print(f"  MILP: ${milp_cost_per_kwh:.4f}")
    
    # Vehículos atendidos
    rl_evs = rl_metrics["evs_served"]
    milp_evs = milp_metrics["evs_served"]
    total_evs = rl_metrics["total_evs"]
    
    print(f"Vehículos atendidos:")
    print(f"  RL:   {rl_evs}/{total_evs} ({rl_evs/total_evs*100:.1f}%)")
    print(f"  MILP: {milp_evs}/{total_evs} ({milp_evs/total_evs*100:.1f}%)")
    
    # Tiempo de ejecución
    rl_time = rl_metrics.get("execution_time", "N/A")
    milp_time = milp_metrics.get("solve_time", "N/A")
    
    print(f"Tiempo de ejecución:")
    print(f"  RL:   {rl_time} segundos")
    print(f"  MILP: {milp_time} segundos")

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Sistema avanzado de optimización de carga de vehículos eléctricos")
    
    parser.add_argument("--mode", type=str, choices=["train", "solve", "optimize", "visualize"], 
                      required=True, help="Modo de operación")
    parser.add_argument("--data_dir", type=str, default="./data", 
                      help="Directorio de datos")
    parser.add_argument("--system", type=str, default=None, 
                      help="Sistema específico a procesar (solo para solve/optimize)")
    parser.add_argument("--model_path", type=str, default="./ev_scheduler_model_pytorch.pt", 
                      help="Ruta del modelo RL")
    parser.add_argument("--output_dir", type=str, default="./results", 
                      help="Directorio para resultados")
    parser.add_argument("--alpha_cost", type=float, default=0.6, 
                      help="Peso del objetivo de costo (0-1)")
    parser.add_argument("--alpha_satisfaction", type=float, default=0.4, 
                      help="Peso del objetivo de satisfacción (0-1)")
    parser.add_argument("--time_limit", type=int, default=900, 
                      help="Límite de tiempo para MILP (segundos)")
    parser.add_argument("--episodes", type=int, default=30, 
                      help="Episodios por sistema para entrenamiento")
    parser.add_argument("--all", action="store_true", help="Procesar todos los sistemas en el directorio de datos")

    parser.add_argument("--no_plot", action="store_true", help="No mostrar ni generar gráficas en modo batch (all)")

    parser.add_argument("--no_time_limit", action="store_true", help="Ejecutar MILP sin límite de tiempo")

    args = parser.parse_args()
    
    # Crear directorios necesarios
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Asegurar que los pesos suman 1
    if abs(args.alpha_cost + args.alpha_satisfaction - 1.0) > 1e-6:
        print("Error: Los pesos de los objetivos deben sumar 1.0")
        print(f"Ajustando automáticamente: {args.alpha_cost} + {args.alpha_satisfaction} = {args.alpha_cost + args.alpha_satisfaction}")
        
        total = args.alpha_cost + args.alpha_satisfaction
        args.alpha_cost /= total
        args.alpha_satisfaction /= total
        
        print(f"Nuevos pesos: Costo ({args.alpha_cost:.2f}), Satisfacción ({args.alpha_satisfaction:.2f})")
    
    # Modo de entrenamiento
    if args.mode == "train":
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
            "episodes_per_system": 25,
            "checkpoint_frequency": 5,
            "checkpoint_dir": "./checkpoints",
            "model_path": "ev_scheduler_model_pytorch.pt", 
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
    
    # Modo de solución con RL
    elif args.mode == "solve":
        print("=== MODO DE SOLUCIÓN RL ===")
        
        # Cargar sistema específico
        if args.all:
            print("Procesando todos los sistemas con RL + MILP...")
            systems = load_all_test_systems(args.data_dir)
            for system_id, config in systems.items():
                print(f"\nSistema {system_id}")
                schedule_rl, rl_metrics = generate_solution(config, model_path=args.model_path)
                rl_metrics["execution_time"] = time.time()
                milp_time_limit = None if args.no_time_limit else args.time_limit

                schedule_milp, milp_metrics = optimize_solution(
                    config,
                    schedule_rl,
                    alpha_cost=args.alpha_cost,
                    alpha_satisfaction=args.alpha_satisfaction,
                    time_limit=milp_time_limit
                )
                if schedule_milp is not None:
                    compare_solutions(rl_metrics, milp_metrics)
                    save_solution(schedule_rl, rl_metrics, os.path.join(args.output_dir, f"rl_solution_{system_id}.json"))
                    save_solution(schedule_milp, milp_metrics, os.path.join(args.output_dir, f"milp_solution_{system_id}.json"))
                    visualize_solution(schedule_rl, config, f"RL Sistema {system_id}: ")
                    visualize_solution(schedule_milp, config, f"MILP Sistema {system_id}: ")
                else:
                    print(f" Fallo en optimización para sistema {system_id}")
            return

        
        # Intentar diferentes formas de cargar el sistema
        config = None
        try_paths = [
            args.system,
            os.path.join(args.data_dir, args.system),
            os.path.join(args.data_dir, f"test_system_{args.system}.json") if args.system.isdigit() else None
        ]
        
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
        output_file = os.path.join(args.output_dir, f"rl_solution_{test_number}.json")
        save_solution(schedule_rl, metrics, output_file)
        
        # Visualizar
        visualize_solution(schedule_rl, config, "RL: ")
    
    # Modo de optimización con MILP
    elif args.mode == "optimize":
        print("=== MODO DE OPTIMIZACIÓN RL+MILP ===")
        
        # Cargar sistema específico
        if args.all:
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
                    save_solution(schedule_rl, rl_metrics, os.path.join(args.output_dir, f"rl_solution_{system_id}.json"))
                    save_solution(schedule_milp, milp_metrics, os.path.join(args.output_dir, f"milp_solution_{system_id}.json"))
                    visualize_solution(schedule_rl, config, f"RL Sistema {system_id}: ", show_plot=not args.no_plot)
                    visualize_solution(schedule_milp, config, f"MILP Sistema {system_id}: ", show_plot=not args.no_plot)

                else:
                    print(f" Fallo en optimización para sistema {system_id}")
            return

        
        # Intentar diferentes formas de cargar el sistema
        config = None
        try_paths = [
            args.system,
            os.path.join(args.data_dir, args.system),
            os.path.join(args.data_dir, f"test_system_{args.system}.json") if args.system.isdigit() else None
        ]
        
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
            rl_output = os.path.join(args.output_dir, f"rl_solution_{test_number}.json")
            milp_output = os.path.join(args.output_dir, f"milp_solution_{test_number}.json")
            
            save_solution(schedule_rl, rl_metrics, rl_output)
            save_solution(schedule_milp, milp_metrics, milp_output)
            
            # Visualizar
            visualize_solution(schedule_rl, config, "RL: ")
            visualize_solution(schedule_milp, config, "MILP: ")
        else:
            print("La optimización MILP no produjo una solución válida.")
    
    # Modo de visualización
    elif args.mode == "visualize":
        print("=== MODO DE VISUALIZACIÓN ===")
        
        # Verificar que se especificó un archivo
        if args.system is None:
            print("Error: Se debe especificar un archivo de solución con --system")
            return
        
        # Cargar solución
        try:
            with open(args.system, 'r') as f:
                solution_data = json.load(f)
                
            # Extraer schedule
            schedule = solution_data.get("schedule", [])
            
            # Identificar sistema correspondiente
            system_id = None
            if "metrics" in solution_data and "total_evs" in solution_data["metrics"]:
                total_evs = solution_data["metrics"]["total_evs"]
                
                # Buscar sistema que coincida
                systems = load_all_test_systems(args.data_dir)
                for sys_id, sys_config in systems.items():
                    if len(sys_config["arrivals"]) == total_evs:
                        system_id = sys_id
                        break
            
            if system_id is None:
                print("No se pudo identificar el sistema correspondiente.")
                return
                
            # Cargar configuración del sistema
            config = None
            try_paths = [
                os.path.join(args.data_dir, f"test_system_{system_id}.json")
            ]
            
            for path in try_paths:
                try:
                    config = load_data(path)
                    print(f"Sistema cargado: {path}")
                    break
                except Exception as e:
                    print(f"Error al cargar {path}: {e}")
            
            if config is None:
                print("No se pudo cargar la configuración del sistema.")
                return
            
            # Visualizar
            visualize_solution(schedule, config)
            
        except Exception as e:
            print(f"Error al visualizar solución: {e}")
    
    else:
        print(f"Modo no reconocido: {args.mode}")

if __name__ == "__main__":
    main()