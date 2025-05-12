import numpy as np
import random
import json
import os
import time
import itertools
from tqdm import tqdm
import argparse
from modules.metaheuristic.scatter_search import scatter_search
from .debug_logger import setup_logger  

OUTPUT_DIR = "hyperparameter/hyperparameter_results_scatter"
PARTIAL_RESULTS_DIR = os.path.join(OUTPUT_DIR, "partial_results")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

# Crear logger global
logger = setup_logger(log_name="scatter_search_tuning", log_dir=LOGS_DIR)

def run_grid_search(config_path, param_grid, n_trials=3, resume=True):
    """
    Realiza una búsqueda en cuadrícula de hiperparámetros para Scatter Search
    para un archivo de configuración específico y guarda los resultados en formato TXT.
    Permite continuar desde la última combinación probada.

    Args:
        config_path: Ruta al archivo JSON con la configuración
        param_grid: Diccionario con los parámetros a explorar
        n_trials: Número de pruebas para cada combinación
        resume: Si es True, intentará continuar desde la última combinación probada
    """
    try:
        logger.info(f"Iniciando búsqueda en cuadrícula para {config_path}")
        logger.info(f"Parámetros a explorar: {param_grid}")
        
        # Verificar que el archivo existe
        if not os.path.exists(config_path):
            logger.error(f"Error: No se encontró el archivo {config_path}")
            return

        # Crear directorios si no existen
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(PARTIAL_RESULTS_DIR, exist_ok=True)

        # Nombre de la configuración
        config_name = os.path.splitext(os.path.basename(config_path))[0]
        logger.info(f"Procesando configuración: {config_name}")

        # Generar todas las combinaciones de parámetros
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))

        logger.info(f"Búsqueda en cuadrícula con {len(param_combinations)} combinaciones de parámetros")

        # Preparar archivo de resultados
        results_file = os.path.join(OUTPUT_DIR, f"instancia_{config_name}_scatter_search_grid_complete.txt")
        partial_results_file = os.path.join(PARTIAL_RESULTS_DIR, f"instancia_{config_name}_scatter_search_partial.txt")

        # Crear el archivo con encabezado si no existe
        if not os.path.exists(results_file):
            with open(results_file, 'w') as f:
                # Crear encabezado con todos los parámetros y métricas
                header = "config_name," + ",".join(param_names) + ",final_cost"
                f.write(header + "\n")
        
        # Crear archivo de resultados parciales si no existe
        if not os.path.exists(partial_results_file):
            with open(partial_results_file, 'w') as f:
                # Encabezado para resultados parciales
                header = "config_name," + ",".join(param_names) + ",trial,cost"
                f.write(header + "\n")

        # Determinar combinaciones ya probadas si se quiere continuar
        skip_combinations = 0
        skip_trials = 0
        current_combination = None
        
        if resume:
            try:
                logger.info("Determinando punto de continuación...")
                # Intentar determinar el punto de continuación desde el archivo principal
                processed_combinations = set()
                with open(results_file, 'r') as f:
                    # Saltar la línea de encabezado
                    next(f)
                    for line in f:
                        if line.strip():
                            parts = line.strip().split(',')
                            # Los primeros len(param_names) + 1 elementos son config_name y parámetros
                            param_values_str = parts[1:len(param_names) + 1]
                            # Convertir strings a tipos apropiados
                            processed_param_values = []
                            for i, value in enumerate(param_values_str):
                                # Intentar convertir a entero o flotante si es posible
                                try:
                                    if isinstance(param_values[i][0], int):
                                        processed_param_values.append(int(value))
                                    elif isinstance(param_values[i][0], float):
                                        processed_param_values.append(float(value))
                                    else:
                                        processed_param_values.append(value)
                                except (ValueError, IndexError):
                                    processed_param_values.append(value)

                            processed_combinations.add(tuple(processed_param_values))

                # Contar cuántas combinaciones ya se han procesado
                for i, combo in enumerate(param_combinations):
                    if combo in processed_combinations:
                        skip_combinations += 1
                    else:
                        # Esta es la primera combinación no procesada
                        break
                        
                # Si no hemos completado todas las combinaciones, verificar resultados parciales
                if skip_combinations < len(param_combinations):
                    current_combination = param_combinations[skip_combinations]
                    
                    # Verificar si hay pruebas parciales para la combinación actual
                    if os.path.exists(partial_results_file):
                        with open(partial_results_file, 'r') as f:
                            next(f)  # Saltar encabezado
                            for line in f:
                                if line.strip():
                                    parts = line.strip().split(',')
                                    # Extraer parámetros de la línea
                                    line_params = []
                                    for i, value in enumerate(parts[1:len(param_names) + 1]):
                                        try:
                                            if isinstance(param_values[i][0], int):
                                                line_params.append(int(value))
                                            elif isinstance(param_values[i][0], float):
                                                line_params.append(float(value))
                                            else:
                                                line_params.append(value)
                                        except (ValueError, IndexError):
                                            line_params.append(value)
                                    
                                    # Si coincide con la combinación actual, incrementar el contador de pruebas
                                    if tuple(line_params) == current_combination:
                                        trial_num = int(parts[len(param_names) + 1])
                                        skip_trials = max(skip_trials, trial_num)

                if skip_combinations == len(param_combinations):
                    logger.info("¡Todas las combinaciones ya han sido procesadas!")
                    return
                elif skip_combinations > 0:
                    logger.info(f"Continuando desde la combinación {skip_combinations + 1}/{len(param_combinations)}")
                    if skip_trials > 0:
                        logger.info(f"Ya se han completado {skip_trials} pruebas para esta combinación")

            except Exception as e:
                logger.error(f"Error al determinar punto de continuación: {str(e)}")
                logger.info("Iniciando desde el principio...")
                skip_combinations = 0
                skip_trials = 0

        # Procesar cada combinación de parámetros, saltando las ya probadas
        for i, params in enumerate(tqdm(param_combinations[skip_combinations:],
                                        initial=skip_combinations,
                                        total=len(param_combinations),
                                        desc=f"Evaluando {config_name}")):
            # Crear diccionario de parámetros
            param_dict = dict(zip(param_names, params))

            # Extraer parámetros para scatter_search
            N = param_dict.get('population_size', 20)
            m1 = param_dict.get('quality_size', 5)
            m2 = param_dict.get('diversity_size', 5)
            max_iter = param_dict.get('max_iterations', 30)

            # Resultados de las pruebas
            costs = []
            
            # Identificador de la combinación actual para archivos parciales
            param_id = "_".join([f"{name}_{param_dict[name]}" for name in param_names])
            logger.info(f"Evaluando combinación: {param_id}")

            # Ejecutar múltiples pruebas
            start_trial = skip_trials if i == 0 else 0
            for trial in range(start_trial, n_trials):
                try:
                    logger.info(f"Ejecutando prueba {trial + 1}/{n_trials} para combinación {i + skip_combinations + 1}/{len(param_combinations)}")
                    logger.info(f"Parámetros: N={N}, m1={m1}, m2={m2}, max_iter={max_iter}")
                    
                    # Tiempo inicial
                    trial_start_time = time.time()
                    
                    # Ejecutar scatter_search con los parámetros actuales
                    try:
                        result = scatter_search(
                            config_path,
                            N=N,
                            m1=m1,
                            m2=m2,
                            max_iter=max_iter,
                            save_results=False  # No guardar resultados individuales durante el tuning
                        )
                        
                        # Registrar el resultado completo para debugging
                        logger.debug(f"Resultado completo: {result}")
                        
                        # CORRECCIÓN: Verificar si result es None antes de acceder a 'cost'
                        if result is None:
                            logger.error("La función scatter_search devolvió None. Saltando esta prueba.")
                            continue
                    except Exception as e:
                        logger.error(f"Error al ejecutar scatter_search: {str(e)}")
                        logger.exception("Excepción completa:")
                        continue

                    if 'cost' in result:
                        trial_cost = result['cost']
                        costs.append(trial_cost)
                                
                        # Tiempo final
                        trial_time = time.time() - trial_start_time
                        
                        # Guardar resultado parcial
                        param_values_str = [str(param_dict[name]) for name in param_names]
                        partial_line = config_name + "," + ",".join(param_values_str) + f",{trial+1},{trial_cost}"
                        
                        with open(partial_results_file, 'a') as f:
                            f.write(partial_line + "\n")
                            f.flush()
                            os.fsync(f.fileno())
                        
                        logger.info(f"Prueba {trial + 1} completada. Costo: {trial_cost:.2f}. Tiempo: {trial_time:.2f} segundos")
                        logger.info(f"Resultado parcial guardado en: {partial_results_file}")
                    else:
                        logger.warning(f"La función scatter_search devolvió un resultado sin 'cost': {result}")
                except Exception as e:
                    logger.error(f"Error en prueba {trial + 1}: {str(e)}")
                    logger.exception("Excepción completa:")
                    continue

            # Calcular costo promedio si hay resultados válidos
            if costs:
                avg_cost = np.mean(costs)

                # Preparar línea de resultados
                param_values_str = [str(param_dict[name]) for name in param_names]
                result_line = config_name + "," + ",".join(param_values_str) + f",{avg_cost}"

                # Guardar resultado en el archivo
                try:
                    with open(results_file, 'a') as f:
                        f.write(result_line + "\n")
                        # Forzar escritura a disco para evitar pérdida de datos
                        f.flush()
                        os.fsync(f.fileno())
                    logger.info(f"Resultados para combinación {i + skip_combinations + 1} guardados en: {results_file}")
                    logger.info(f"Parámetros: {param_dict}, Costo promedio: {avg_cost:.2f}")
                except Exception as e:
                    logger.error(f"Error al guardar resultados finales: {str(e)}")
                    # Guardar en un archivo de respaldo como plan B
                    backup_file = os.path.join(OUTPUT_DIR, "backup_results.txt")
                    try:
                        with open(backup_file, 'a') as bf:
                            bf.write(result_line + "\n")
                            bf.flush()
                        logger.info(f"Backup guardado en: {backup_file}")
                    except Exception as be:
                        logger.critical(f"ERROR AL GUARDAR BACKUP: {str(be)}")
            else:
                logger.warning(f"No se obtuvieron resultados válidos para: {param_dict}")
    except Exception as e:
        logger.critical(f"Error en la búsqueda en cuadrícula: {str(e)}")
        logger.exception("Excepción completa:")

def main():
    parser = argparse.ArgumentParser(description="Tuning de hiperparámetros para Scatter Search")
    parser.add_argument('config_path', type=str,
                        help='Ruta al archivo JSON específico para hacer el tuning')
    parser.add_argument('--n_trials', type=int, default=3,
                        help='Número de pruebas por configuración')
    parser.add_argument('--no-resume', action='store_true',
                        help='No continuar desde la última combinación (comenzar desde cero)')

    args = parser.parse_args()
    
    logger.info(f"Iniciando tuning con args: {args}")

    # Definir parámetros para la búsqueda en cuadrícula
    param_grid = {
        "population_size": [10],  # Desde pequeña hasta muy grande
        "quality_size": [5],  # Diferentes tamaños para subpoblación de calidad
        "diversity_size": [5],  # Diferentes tamaños para subpoblación de diversidad
        "max_iterations": [30]  # Desde pocas hasta muchas iteraciones
    }

    # Ejecutar búsqueda en cuadrícula para el archivo específico
    run_grid_search(args.config_path, param_grid, args.n_trials, resume=not args.no_resume)

    logger.info(f"Tuning completado. Resultados guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()