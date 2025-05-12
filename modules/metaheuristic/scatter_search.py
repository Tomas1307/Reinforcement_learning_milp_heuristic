import json
import random
import itertools
import numpy as np
import os
import time
import traceback
from collections import defaultdict
from modules.constructive.heuristic import load_data, HeuristicaConstructivaEVs
from modules.constructive.price_heuristic import HeuristicaPorPrecio
from modules.constructive.time_slot_heuristic import HeuristicaPorVentanaTiempo
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from modules.benchmark_tuning.debug_logger import setup_logger
logger = setup_logger(log_name="scatter_search", log_dir="logs")

# Definir carpeta de resultados
RESULTS_DIR = "../../results/scatter_search_results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def measure_diversity(sol1, sol2):
    """
    Métrica de diversidad basada en 1 - similitud coseno entre dos soluciones.
    Cada solución es una lista de tuplas (ev_id, t_idx, charger_id, slot, power).
    """
    dict1 = {}
    dict2 = {}
    
    # Procesar sol1, convirtiendo None a -1 para charger_id
    for e, t, c, s, p in sol1:
        # Convertir None a -1 para evitar problemas de comparación
        c_value = -1 if c is None else c
        key = (e, t, c_value, s)
        dict1[key] = dict1.get(key, 0) + 1
        
    # Procesar sol2, convirtiendo None a -1 para charger_id
    for e, t, c, s, p in sol2:
        # Convertir None a -1 para mantener consistencia
        c_value = -1 if c is None else c
        key = (e, t, c_value, s)
        dict2[key] = dict2.get(key, 0) + 1

    # Vectorizar los diccionarios para calcular similitud coseno
    vectorizer = DictVectorizer(sparse=False)
    vectors = vectorizer.fit_transform([dict1, dict2])

    similarity = cosine_similarity([vectors[0]], [vectors[1]])[0][0]

    diversity = 1 - similarity
    return diversity

def generar_solucion_aleatoria(config):
    """
    Genera una solución completamente aleatoria para el problema de carga de EVs.
    
    Args:
        config: Diccionario con la configuración del sistema
        
    Returns:
        Lista de tuplas (ev_id, t_idx, charger_id, slot, power) con el schedule aleatorio
    """
    try:
        logger.info("Generando solución completamente aleatoria...")
        
        # Extraer datos de la configuración
        times = config["times"]
        dt = config["dt"]
        arrivals = config["arrivals"]
        chargers = config["chargers"]
        station_limit = config["station_limit"]
        n_spots = config["n_spots"]
        
        # Mapeos para facilitar el acceso
        ev_ids = [arr["id"] for arr in arrivals]
        arrival_time = {arr["id"]: arr["arrival_time"] for arr in arrivals}
        departure_time = {arr["id"]: arr["departure_time"] for arr in arrivals}
        required_energy = {arr["id"]: arr["required_energy"] for arr in arrivals}
        
        charger_ids = [c["charger_id"] for c in chargers]
        max_charger_power = {c["charger_id"]: c["power"] for c in chargers}
        
        # Variables para controlar el estado del sistema
        schedule = []
        occupied_spots = {t: set() for t in range(len(times))}
        occupied_chargers = {t: set() for t in range(len(times))}
        power_used = {t: 0 for t in range(len(times))}
        energy_delivered = {ev_id: 0 for ev_id in ev_ids}
        
        # Aleatoriamente decidir qué EVs serán cargados en cada intervalo
        for t_idx in range(len(times)):
            # Identificar vehículos presentes en este intervalo
            evs_presentes = [
                ev_id for ev_id in ev_ids
                if arrival_time[ev_id] <= times[t_idx] < departure_time[ev_id]
            ]
            
            # Barajar aleatoriamente los EVs presentes y los spots
            random.shuffle(evs_presentes)
            spots_disponibles = list(range(n_spots))
            random.shuffle(spots_disponibles)
            
            # Barajar chargers
            chargers_disponibles = charger_ids.copy()
            random.shuffle(chargers_disponibles)
            
            # Asignar spots y cargadores aleatoriamente
            for ev_id in evs_presentes:
                if not spots_disponibles:
                    break  # No hay más spots disponibles
                    
                slot = spots_disponibles.pop(0)
                occupied_spots[t_idx].add(slot)
                
                # Decidir aleatoriamente si asignar un cargador o no
                if random.random() < 0.7 and chargers_disponibles and energy_delivered[ev_id] < required_energy[ev_id]:
                    # Asignar cargador
                    charger_id = chargers_disponibles.pop(0)
                    occupied_chargers[t_idx].add(charger_id)
                    
                    # Determinar potencia aleatoriamente entre el 30% y 100% del máximo
                    potencia_maxima = min(
                        max_charger_power[charger_id],
                        (required_energy[ev_id] - energy_delivered[ev_id]) / dt,
                        station_limit - power_used[t_idx]
                    )
                    
                    power = random.uniform(0.3 * potencia_maxima, potencia_maxima) if potencia_maxima > 0 else 0
                    
                    if power > 0:
                        schedule.append((ev_id, t_idx, charger_id, slot, power))
                        power_used[t_idx] += power
                        energy_delivered[ev_id] += power * dt
                    else:
                        # Sin potencia, asignar spot sin cargador
                        schedule.append((ev_id, t_idx, None, slot, 0))
                else:
                    # Asignar spot sin cargador
                    schedule.append((ev_id, t_idx, None, slot, 0))
        
        # Comprobar factibilidad y reparar si es necesario
        for t_idx in range(len(times)):
            t_power = sum(power for (_, t, _, _, power) in schedule if t == t_idx)
            if t_power > station_limit:
                # Reducir la potencia proporcionalmente
                factor = station_limit / t_power
                for i, (ev_id, t, charger_id, slot, power) in enumerate(schedule):
                    if t == t_idx and power > 0:
                        schedule[i] = (ev_id, t, charger_id, slot, power * factor)
        
        # Comprobar que no se exceda la energía requerida
        energia_entregada = defaultdict(float)
        for (ev_id, t_idx, charger_id, slot, power) in schedule:
            if power > 0:
                energia_entregada[ev_id] += power * dt
        
        for ev_id, energia in energia_entregada.items():
            if energia > required_energy[ev_id]:
                # Reducir la energía proporcionalmente
                factor = required_energy[ev_id] / energia
                for i, (e, t, c, s, p) in enumerate(schedule):
                    if e == ev_id and p > 0:
                        schedule[i] = (e, t, c, s, p * factor)
        
        logger.info(f"Solución aleatoria generada con {len(schedule)} asignaciones")
        return schedule
    except Exception as e:
        logger.error(f"Error en generar_solucion_aleatoria: {str(e)}")
        logger.debug(traceback.format_exc())
        print("Error en generar_solucion_aleatoria:", e)

def evaluar_costo(schedule, config):
    """
    Evalúa el costo total de un schedule con penalización.
    
    Args:
        schedule: Lista de tuplas (ev_id, t_idx, charger_id, slot, power)
        config: Configuración del sistema
        
    Returns:
        float: Costo total del schedule
    """
    try:
        # Verificar si el schedule está vacío
        if not schedule:
            logger.warning("Evaluando schedule vacío, devolviendo costo máximo")
            return float('inf')
            
        # Extraer datos necesarios
        times = config["times"]
        prices = config["prices"]
        dt = config["dt"]
        arrivals = config["arrivals"]
        
        # Mapeos para facilitar el acceso
        required_energy = {arr["id"]: arr["required_energy"] for arr in arrivals}
        ev_ids = [arr["id"] for arr in arrivals]
        
        # Contador de energía entregada por EV
        energia_entregada = defaultdict(float)
        
        # Calcular costos por intervalo y energía entregada
        costo_operacion = 0.0
        
        for (ev_id, t_idx, charger_id, slot, power) in schedule:
            if charger_id is not None and power > 0:
                energia = power * dt
                costo_operacion += energia * prices[t_idx]
                energia_entregada[ev_id] += energia
        
        # Penalización por energía no entregada
        penalizacion_base = 1000.0
        costo_penalizacion = 0.0
        
        for ev_id in ev_ids:
            energia_requerida = required_energy[ev_id]
            energia_no_entregada = max(0, energia_requerida - energia_entregada[ev_id])
            
            if energia_no_entregada > 0:
                # Penalización progresiva
                porcentaje_no_entregado = energia_no_entregada / energia_requerida
                factor_penalizacion = penalizacion_base * porcentaje_no_entregado
                
                costo_penalizacion += energia_no_entregada * factor_penalizacion
        
        costo_total = costo_operacion + costo_penalizacion
        return costo_total
    except Exception as e:
        logger.error(f"Error en evaluar_costo: {str(e)}")
        logger.debug(traceback.format_exc())
        print("Error en evaluar_costo:", e)


def combine_solutions(u_schedule, v_schedule, time_slots):
    """
    Combina dos soluciones u y v usando time_slots para decidir qué intervalos tomar de u.
    """
    try:
        # Verificar si alguna de las soluciones está vacía
        if not u_schedule:
            return v_schedule.copy() if v_schedule else []
        if not v_schedule:
            return u_schedule.copy() if u_schedule else []
            
        combined = [a for a in u_schedule if a[1] in time_slots]
        combined += [a for a in v_schedule if a[1] not in time_slots]
        return combined
    except Exception as e:
        logger.error(f"Error en combine_solutions: {str(e)}")
        logger.debug(traceback.format_exc())
        print("Error en combine_solutions:", e)


def scatter_search(config_path, N=20, m1=5, m2=5, max_iter=30, save_results=True):
    """
    Ejecuta el algoritmo de Scatter Search y guarda los resultados en formato JSON.

    Args:
        config_path: Ruta al archivo JSON de configuración
        N: Tamaño de la población inicial
        m1: Número de soluciones de calidad en RefSet
        m2: Número de soluciones diversas en RefSet
        max_iter: Número de iteraciones de Scatter Search
        save_results: Si es True, guarda los resultados en un archivo JSON

    Returns:
        dict: Resultado del algoritmo con la mejor solución encontrada
    """
    try:
        logger.info(f"Iniciando Scatter Search con N={N}, m1={m1}, m2={m2}, max_iter={max_iter}")
        start_time = time.time()

        # Cargar datos y configuración
        try:
            data = load_data(config_path)
            logger.info(f"Datos cargados desde {config_path}")
        except Exception as e:
            logger.error(f"Error al cargar datos: {str(e)}")
            logger.debug(traceback.format_exc())
            return {'cost': float('inf'), 'schedule': [], 'error': f"Error al cargar datos: {str(e)}"}

        # Para registrar evolución de costos
        costs_evolution = []

        # 1. Generación de población inicial diversa
        logger.info("Generando población inicial...")
        population = []
        for i in range(N):
            try:
                seed = random.randint(0, 2 ** 32 - 1)
                random.seed(seed)
                np.random.seed(seed)
                
                # Determinar qué método de generación usar
                method = i % 4  # Usar 4 métodos diferentes
                
                if method == 0:
                    # Heurística original (prioridad por urgencia)
                    logger.debug(f"Población {i+1}/{N}: Usando heurística constructiva")
                    heur = HeuristicaConstructivaEVs(data)
                    sched = heur.run(track_progress=False)
                elif method == 1:
                    # Heurística con prioridad por precio
                    logger.debug(f"Población {i+1}/{N}: Usando heurística por precio")
                    heur = HeuristicaPorPrecio(data)  # Nueva heurística 
                    sched = heur.run(track_progress=False)
                elif method == 2:
                    # Heurística con prioridad por ventana de tiempo
                    logger.debug(f"Población {i+1}/{N}: Usando heurística por ventana de tiempo")
                    heur = HeuristicaPorVentanaTiempo(data)  # Nueva heurística
                    sched = heur.run(track_progress=False)
                else:
                    # Solución aleatoria
                    logger.debug(f"Población {i+1}/{N}: Usando solución aleatoria")
                    sched = generar_solucion_aleatoria(data)  # Nueva función
                
                # Verificar que se haya generado una solución válida
                if not sched:
                    logger.warning(f"Solución vacía generada para población {i+1}. Intentando heurística constructiva.")
                    heur = HeuristicaConstructivaEVs(data)
                    sched = heur.run(track_progress=False)
                    if not sched:
                        logger.error(f"No se pudo generar una solución para la población {i+1}")
                        continue
                
                cost = evaluar_costo(sched, data)
                population.append({'schedule': sched, 'cost': cost})

                logger.info(f"Población inicial {i + 1}/{N}: Costo = {cost:.2f}")
            except Exception as e:
                logger.error(f"Error al generar solución {i+1}: {str(e)}")
                logger.debug(traceback.format_exc())
                continue

        # Verificar si se generó alguna solución válida
        if not population:
            logger.error("No se pudo generar ninguna solución válida para la población inicial")
            return {'cost': float('inf'), 'schedule': [], 'error': "Población inicial vacía"}

        # Ajustar m1 y m2 si no hay suficientes soluciones
        m1 = min(m1, len(population))
        m2 = min(m2, len(population) - m1)
        
        if m1 <= 0 or m2 <= 0:
            logger.error(f"No hay suficientes soluciones para formar el RefSet (m1={m1}, m2={m2})")
            # Devolver la mejor solución encontrada
            best_solution = min(population, key=lambda x: x['cost'])
            return best_solution

        # 2. Selección inicial de RefSet
        population.sort(key=lambda x: x['cost'])
        ref1 = population[:m1]
        rest = population[m1:]
        
        # Si no hay suficientes soluciones para la diversidad, usar todas las disponibles
        if not rest:
            logger.warning("No hay suficientes soluciones para la parte diversa del RefSet")
            refset = ref1
        else:
            # Código mejorado
            # En la selección inicial del RefSet
            div_list = []
            for sol in rest:
                try:
                    diversities = []
                    for r in ref1:
                        try:
                            div = measure_diversity(sol['schedule'], r['schedule'])
                            diversities.append(div)
                        except Exception as e:
                            logger.error(f"Error calculando diversidad individual: {str(e)}")
                            # Continuar con la siguiente comparación
                    
                    if diversities:  # Si hay al menos una medida de diversidad válida
                        total_diversity = sum(diversities)
                        div_list.append((sol, total_diversity))
                    else:
                        logger.warning(f"No se pudo calcular ninguna diversidad válida para la solución")
                except Exception as e:
                    logger.error(f"Error procesando solución para diversidad: {str(e)}")

            # Si div_list está vacío, necesitamos una estrategia alternativa
            if not div_list:
                logger.warning("No se pudo calcular diversidad para ninguna solución")
                
                # Alternativa: usar las mejores soluciones restantes por costo
                m2_actual = min(m2, len(rest))
                ref2 = rest[:m2_actual]
            else:
                # Ordenar por diversidad y seleccionar
                div_list.sort(key=lambda x: x[1], reverse=True)
                m2_actual = min(m2, len(div_list))
                ref2 = [sol for sol, _ in div_list[:m2_actual]]

            # Formar el RefSet
            refset = ref1 + ref2

            # Verificar que el RefSet no esté vacío
            if not refset:
                raise ValueError("Error: RefSet vacío. No se puede continuar con el algoritmo.")

        # Encontrar mejor solución inicial
        best_solution = min(refset, key=lambda x: x['cost'])
        costs_evolution.append(best_solution['cost'])
        logger.info(f"Mejor solución inicial: Costo = {best_solution['cost']:.2f}")

        # 3. Bucle principal de Scatter Search
        for iteration in range(max_iter):
            logger.info(f"Iniciando iteración {iteration + 1}/{max_iter}")
            candidates = []

            # Generar todas las combinaciones
            for u, v in itertools.combinations(refset, 2):
                try:
                    all_times = {a[1] for a in u['schedule']}
                    # Verificar que hay tiempos disponibles
                    if not all_times:
                        logger.warning("No hay intervalos de tiempo en la solución u, saltando combinación")
                        continue
                        
                    k = max(1, int(0.2 * len(all_times)))
                    time_slots = set(random.sample(list(all_times), k))

                    comb = combine_solutions(u['schedule'], v['schedule'], time_slots)
                    
                    # Verificar que la combinación es válida
                    if not comb:
                        logger.warning("Combinación vacía, saltando")
                        continue
                        
                    repair_heur = HeuristicaConstructivaEVs(data)
                    repaired = repair_heur._reconstruye_solucion(comb)

                    repair_heur.schedule = repaired
                    improved = repair_heur.run(track_progress=False)
                    
                    # Verificar si la solución es válida
                    if not improved:
                        logger.warning("Solución mejorada vacía, saltando")
                        continue
                        
                    cost_imp = evaluar_costo(improved, data)
                    candidates.append({'schedule': improved, 'cost': cost_imp})
                    logger.debug(f"Nueva solución candidata generada: Costo = {cost_imp:.2f}")
                except Exception as e:
                    logger.error(f"Error al generar combinación: {str(e)}")
                    logger.debug(traceback.format_exc())
                    continue

            # Verificar si se generaron candidatos
            if not candidates:
                logger.warning("No se generaron candidatos en esta iteración")
                # Continuar con la próxima iteración
                continue

            # 4. Actualizar RefSet
            # 4. Actualizar RefSet
            all_sols = refset + candidates
            all_sols.sort(key=lambda x: x['cost'])

            # Ajustar m1 y m2 si no hay suficientes soluciones
            m1_actual = min(m1, len(all_sols))

            ref1 = all_sols[:m1_actual]
            rest = all_sols[m1_actual:]

            if not rest:
                logger.warning("No hay suficientes soluciones para la parte diversa del RefSet actualizado")
                refset = ref1
            else:
                div_list = []
                for sol in rest:
                    # Verificar si la solución actual es válida
                    if 'schedule' not in sol or sol['schedule'] is None or len(sol['schedule']) == 0:
                        logger.warning("Encontrada solución con schedule inválido, ignorando")
                        continue
                    
                    # Calcular diversidad para cada solución en ref1
                    valid_diversity_scores = []
                    for r in ref1:
                        # Verificar si la solución de referencia es válida
                        if 'schedule' not in r or r['schedule'] is None or len(r['schedule']) == 0:
                            logger.warning("Solución de referencia inválida en ref1, ignorando")
                            continue
                            
                        try:
                            div_score = measure_diversity(sol['schedule'], r['schedule'])
                            valid_diversity_scores.append(div_score)
                        except Exception as e:
                            logger.error(f"Error calculando diversidad: {str(e)}")
                            # Continuar con la siguiente comparación
                    
                    # Solo añadir la solución si se pudo calcular al menos una diversidad válida
                    if valid_diversity_scores:
                        total_score = sum(valid_diversity_scores)
                        div_list.append((sol, total_score))
                
                # Solo ordenar y seleccionar si hay soluciones válidas
                if div_list:
                    div_list.sort(key=lambda x: x[1], reverse=True)
                    m2_actual = min(m2, len(div_list))
                    ref2 = [sol for sol, _ in div_list[:m2_actual]]
                    refset = ref1 + ref2
                else:
                    logger.warning("No se pudieron calcular diversidades válidas, usando solo soluciones de calidad")
                    refset = ref1

            # Actualizar mejor solución
            if refset:  # Verificar que el refset no esté vacío
                current_best = min(refset, key=lambda x: x['cost'])
                costs_evolution.append(current_best['cost'])

                if current_best['cost'] < best_solution['cost']:
                    best_solution = current_best
                    logger.info(f"Nueva mejor solución encontrada: Costo = {current_best['cost']:.2f}")

                logger.info(f"Iteración {iteration + 1}/{max_iter}: Mejor costo = {current_best['cost']:.2f}")
            else:
                logger.error("RefSet vacío después de actualización")
                break

        # 5. Calcular estadísticas finales
        end_time = time.time()
        total_time = end_time - start_time

        # Construir la heurística con la mejor solución para obtener estadísticas
        best_heur = HeuristicaConstructivaEVs(data)
        best_heur.schedule = best_solution['schedule']

        # Obtener resultados en el formato de la heurística
        try:
            heur_result = best_heur.get_resultados()
        except Exception as e:
            logger.error(f"Error al obtener resultados de la heurística: {str(e)}")
            logger.debug(traceback.format_exc())
            
            # Devolver al menos la solución encontrada
            return {
                'schedule': best_solution['schedule'],
                'cost': best_solution['cost'],
                'extra_info': {
                    'total_time': total_time,
                    'error': f"Error al formatear resultados: {str(e)}"
                }
            }

        # Construir el formato de resultado final
        resultado = {}

        # Convertir el schedule al formato requerido
        schedule_formateado = {}
        for ev_id, intervals in heur_result["schedule"].items():
            schedule_formateado[str(ev_id)] = intervals

        resultado["schedule"] = schedule_formateado

        # Calcular rejected_details para cada EV no satisfecho
        rejected_details = {}
        for ev_id in best_heur.ev_ids:
            required = best_heur.required_energy[ev_id]
            delivered = best_heur.energy_delivered[ev_id]
            if delivered < required:
                unmet = required - delivered
                rejected_details[str(ev_id)] = {
                    "required_energy": required,
                    "delivered_energy": delivered,
                    "unmet_energy": unmet,
                    "penalty_cost": unmet * 1000
                }

        # Añadir la información extra (similar al formato MILP pero sin milp_time)
        resultado["extra_info"] = {
            "rejected_details": rejected_details,
            "total_time": total_time,
            "cost_evolution": costs_evolution
        }

        resultado["cost"] = best_solution['cost']

        # Guardar el resultado en el formato especificado
        if save_results:
            try:
                logger.info(f"Guardando resultado con costo {resultado['cost']}")
                instance_name = os.path.splitext(os.path.basename(config_path))[0]
                output_path = os.path.join(RESULTS_DIR, f"resultados_scatter_instancia_{instance_name}.json")

                with open(output_path, 'w') as f:
                    json.dump(resultado, f, indent=4)

                logger.info(f"Resultados guardados en: {output_path}")
                logger.info(f"Tiempo total de ejecución: {total_time:.2f} segundos")
                logger.info(f"Costo final: {best_solution['cost']:.2f}")
            except Exception as e:
                logger.error(f"Error al guardar el resultado: {str(e)}")
                logger.debug(traceback.format_exc())

        return resultado
        
    except Exception as e:
        logger.critical(f"Error general en scatter_search: {str(e)}")
        logger.debug(traceback.format_exc())
        # Devolver un diccionario con información sobre el error
        return {
            'cost': float('inf'),
            'schedule': [],
            'error': f"Error crítico: {str(e)}"
        }

"""
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python scatter_search_modified.py <archivo_json> [N] [m1] [m2] [max_iter]")
        sys.exit(1)

    config_path = sys.argv[1]

    # Parámetros opcionales
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    m1 = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    m2 = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    max_iter = int(sys.argv[5]) if len(sys.argv) > 5 else 30

    # Ejecutar Scatter Search
    print(f"Ejecutando Scatter Search para {config_path}")
    print(f"Parámetros: N={N}, m1={m1}, m2={m2}, max_iter={max_iter}")

    result = scatter_search(
        config_path=config_path,
        N=N,
        m1=m1,
        m2=m2,
        max_iter=max_iter,
        save_results=True
    )

"""