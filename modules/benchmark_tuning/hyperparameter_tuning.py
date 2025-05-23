import numpy as np
import random
import matplotlib.pyplot as plt
import os
import glob
import time
import pandas as pd
from itertools import product
from functools import partial
import copy
import math
from tqdm import tqdm

# Importar las funciones necesarias del archivo principal
# Asumiendo que la clase se llama "heuristic.py"
from modules.constructive.heuristic import load_data, HeuristicaConstructivaEVs, save_schedule_to_json

class HyperparameterTuner:
    """
    Clase para realizar búsqueda de hiperparámetros para la heurística constructiva.
    Permite explorar diferentes configuraciones y encontrar los parámetros óptimos.
    """
    
    def __init__(self, config_paths, output_dir="hyperparameter/hyperparameter_results_constructive", n_trials=10):
        """
        Inicializa el tuner con los archivos de configuración y parámetros básicos.
        
        Args:
            config_paths: Lista de rutas a archivos JSON con las configuraciones de prueba
            output_dir: Directorio donde guardar los resultados
            n_trials: Número de pruebas para cada configuración de hiperparámetros
        """
        self.config_paths = config_paths
        self.output_dir = output_dir
        self.n_trials = n_trials
        
        # Crear el directorio de salida si no existe
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Cargar las configuraciones
        self.configs = {}
        for path in config_paths:
            config_name = os.path.splitext(os.path.basename(path))[0]
            self.configs[config_name] = load_data(path)
        
        # Inicializar resultados
        self.results = []
        
    def grid_search(self, param_grid, n_jobs=1, actual_job=None):
        """
        Realiza una búsqueda en cuadrícula sobre los hiperparámetros especificados.
        Soporta reanudación desde el último punto guardado.

        Args:
            param_grid: Diccionario con los parámetros a explorar y sus posibles valores
            n_jobs: Número de particiones totales a crear
            actual_job: Índice de la partición actual a procesar (1 a n_jobs)

        Returns:
            DataFrame con los resultados de la búsqueda
        """
        print(f"Iniciando búsqueda en cuadrícula")
        if actual_job is not None:
            print(f"Procesando partición {actual_job} de {n_jobs}")
        print(f"Parámetros a explorar: {param_grid}")

        # Generar todas las combinaciones de parámetros
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        total_combinations = len(param_combinations)
        print(f"Total de combinaciones: {total_combinations}")
        
        # Si se está utilizando partición, seleccionar solo la parte correspondiente
        if actual_job is not None:
            # Validar que actual_job está en el rango correcto
            if actual_job < 1 or actual_job > n_jobs:
                raise ValueError(f"actual_job debe estar entre 1 y {n_jobs}")
            
            # Calcular el tamaño de cada partición
            partition_size = math.ceil(total_combinations / n_jobs)
            
            # Calcular índices de inicio y fin para esta partición
            start_idx = (actual_job - 1) * partition_size
            end_idx = min(actual_job * partition_size, total_combinations)
            
            # Seleccionar solo las combinaciones de esta partición
            param_combinations = param_combinations[start_idx:end_idx]
            
            print(f"Partición {actual_job}: procesando {len(param_combinations)} combinaciones de parámetros")
            print(f"Rango de índices: {start_idx} a {end_idx-1}")

        # Para cada archivo de configuración
        for config_name, config in self.configs.items():
            print(f"\nProcesando configuración: {config_name}")

            # Construir el nombre del archivo incluyendo el método y partición
            file_suffix = f"grid_part{actual_job}of{n_jobs}" if actual_job is not None else "grid"
            instance_file = os.path.join(self.output_dir, f"instancia_{config_name}_{file_suffix}.txt")
            
            # Verificar si hay un punto de reanudación
            completed_params = set()
            if os.path.exists(instance_file):
                try:
                    # Cargar resultados existentes
                    existing_df = pd.read_csv(instance_file)
                    print(f"Encontrado archivo de resultados previo con {len(existing_df)} registros")
                    
                    # Identificar combinaciones ya procesadas
                    for _, row in existing_df.iterrows():
                        param_tuple = tuple(row[param] for param in param_names if param in row)
                        if len(param_tuple) == len(param_names):  # Verificar que estén todos los parámetros
                            completed_params.add(param_tuple)
                    
                    print(f"Se reanudarán {len(param_combinations) - len(completed_params)} combinaciones pendientes")
                    print(f"Se omitirán {len(completed_params)} combinaciones ya procesadas")
                    
                    # Cargar resultados previos
                    self.results.extend(existing_df.to_dict('records'))
                except Exception as e:
                    print(f"Error al leer el archivo existente: {e}")
                    print("Se iniciarán los resultados desde cero")
                    if os.path.exists(instance_file):
                        # Hacer backup del archivo con problemas
                        backup_file = f"{instance_file}.bak.{int(time.time())}"
                        os.rename(instance_file, backup_file)
                        print(f"Archivo con problemas respaldado como {backup_file}")
            
            # Preparar la función para ejecución
            run_trial_partial = partial(
                self._run_trial,
                config_name=config_name,
                config=config,
                param_names=param_names
            )

            overall_results = []  # Acumula resultados nuevos para esta instancia
            intermediate_results = []  # Buffer para cada N iteraciones
            
            # Procesar solo las combinaciones pendientes
            for i, params in enumerate(tqdm(param_combinations, desc=f"Procesando {config_name}")):
                # Omitir si ya se procesó esta combinación
                if params in completed_params:
                    continue
                
                result = run_trial_partial(params)
                overall_results.append(result)
                intermediate_results.append(result)
                
                # Cada 2 iteraciones se escribe el bloque en el archivo para guardar progreso
                if (i + 1) % 2 == 0 or len(intermediate_results) >= 2:
                    if intermediate_results:  # Solo guardar si hay resultados nuevos
                        df_intermediate = pd.DataFrame(intermediate_results)
                        # Si el archivo ya existe, se añade sin cabecera
                        if os.path.exists(instance_file):
                            df_intermediate.to_csv(instance_file, index=False, sep=",", mode='a', header=False)
                        else:
                            df_intermediate.to_csv(instance_file, index=False, sep=",")
                        intermediate_results = []  # Reiniciar buffer

            # Guardar los resultados restantes si quedaron algunos
            if intermediate_results:
                df_intermediate = pd.DataFrame(intermediate_results)
                if os.path.exists(instance_file):
                    df_intermediate.to_csv(instance_file, index=False, sep=",", mode='a', header=False)
                else:
                    df_intermediate.to_csv(instance_file, index=False, sep=",")

            print(f"\nResultados guardados para {config_name} en {instance_file}")
            print(f"Se procesaron {len(overall_results)} nuevas combinaciones")
            
            # No extender resultados globales si no hay nuevos
            if overall_results:
                self.results.extend(overall_results)
                
                # Guardar resultados de esta partición
                results_df = pd.DataFrame(self.results)
                results_suffix = f"grid_part{actual_job}of{n_jobs}" if actual_job is not None else "grid"
                results_path = os.path.join(self.output_dir, f"search_results_{results_suffix}.csv")
                results_df.to_csv(results_path, index=False)
                print(f"\nResultados de esta partición guardados en {results_path}")

        return pd.DataFrame(self.results)


    
    def random_search(self, param_distributions, n_samples=20, n_jobs=1, actual_job=None):
        """
        Realiza una búsqueda aleatoria sobre el espacio de hiperparámetros.
        Soporta reanudación desde el último punto guardado.
        
        Args:
            param_distributions: Diccionario con parámetros y sus distribuciones/rangos
            n_samples: Número de muestras aleatorias a evaluar
            n_jobs: Número de particiones totales a crear
            actual_job: Índice de la partición actual a procesar (1 a n_jobs)
        
        Returns:
            DataFrame con los resultados de la búsqueda
        """
        print(f"Iniciando búsqueda aleatoria")
        if actual_job is not None:
            print(f"Procesando partición {actual_job} de {n_jobs}")
        print(f"Parámetros a explorar: {param_distributions}")
        
        # Generar muestras aleatorias
        param_names = list(param_distributions.keys())
        
        # Configurar semilla aleatoria para reproducibilidad
        # Usamos una semilla diferente para cada job para asegurar distribuciones distintas
        if actual_job is not None:
            random.seed(42 + actual_job)  # Semilla base + actual_job
        else:
            random.seed(42)
        
        # Calcular cuántas muestras debe generar esta partición
        samples_per_job = n_samples
        if actual_job is not None:
            samples_per_job = math.ceil(n_samples / n_jobs)
            if actual_job == n_jobs:  # El último job puede tener menos muestras
                samples_per_job = n_samples - (n_jobs - 1) * math.ceil(n_samples / n_jobs)
                samples_per_job = max(1, samples_per_job)  # Al menos una muestra
        
        print(f"Generando {samples_per_job} muestras aleatorias")
        
        # Generar muestras para esta partición
        param_samples = []
        for _ in range(samples_per_job):
            sample = []
            for dist in param_distributions.values():
                if isinstance(dist, list):
                    # Para opciones discretas
                    sample.append(random.choice(dist))
                elif isinstance(dist, tuple) and len(dist) == 2:
                    # Para rangos continuos (min, max)
                    if isinstance(dist[0], int) and isinstance(dist[1], int):
                        # Rango de enteros
                        sample.append(random.randint(dist[0], dist[1]))
                    else:
                        # Rango de flotantes
                        sample.append(random.uniform(dist[0], dist[1]))
                else:
                    raise ValueError(f"Distribución no válida: {dist}")
            param_samples.append(tuple(sample))
        
        # Para cada archivo de configuración
        for config_name, config in self.configs.items():
            print(f"\nProcesando configuración: {config_name}")
            
            # Construir el nombre del archivo incluyendo el método y partición
            file_suffix = f"random_part{actual_job}of{n_jobs}" if actual_job is not None else "random"
            instance_file = os.path.join(self.output_dir, f"instancia_{config_name}_{file_suffix}.txt")
            
            # Verificar si hay un punto de reanudación para búsqueda aleatoria
            completed_params_count = 0
            if os.path.exists(instance_file):
                try:
                    # Cargar resultados existentes
                    existing_df = pd.read_csv(instance_file)
                    completed_params_count = len(existing_df)
                    print(f"Encontrado archivo de resultados previo con {completed_params_count} registros")
                    
                    # Cargar resultados previos
                    self.results.extend(existing_df.to_dict('records'))
                    
                    # Para búsqueda aleatoria no filtramos por parámetros (son aleatorios),
                    # sino que verificamos si ya tenemos suficientes muestras
                    if completed_params_count >= samples_per_job:
                        print(f"Ya se procesaron todas las muestras requeridas ({completed_params_count})")
                        print(f"Omitiendo esta configuración")
                        continue
                    
                    # Ajustar el número de muestras a generar
                    remaining_samples = samples_per_job - completed_params_count
                    print(f"Generando {remaining_samples} muestras adicionales")
                    param_samples = param_samples[:remaining_samples]
                    
                except Exception as e:
                    print(f"Error al leer el archivo existente: {e}")
                    print("Se iniciarán los resultados desde cero")
                    if os.path.exists(instance_file):
                        # Hacer backup del archivo con problemas
                        backup_file = f"{instance_file}.bak.{int(time.time())}"
                        os.rename(instance_file, backup_file)
                        print(f"Archivo con problemas respaldado como {backup_file}")
            
            # Preparar la función para ejecución
            run_trial_partial = partial(
                self._run_trial, 
                config_name=config_name,
                config=config,
                param_names=param_names
            )
            
            overall_results = []  # Acumula resultados nuevos para esta instancia
            intermediate_results = []  # Buffer para cada N iteraciones
            
            # Procesar las muestras de parámetros
            for i, params in enumerate(tqdm(param_samples, desc=f"Procesando {config_name}")):
                result = run_trial_partial(params)
                overall_results.append(result)
                intermediate_results.append(result)
                
                # Cada 2 iteraciones se escribe el bloque en el archivo para guardar progreso
                if (i + 1) % 2 == 0 or len(intermediate_results) >= 2:
                    if intermediate_results:  # Solo guardar si hay resultados nuevos
                        df_intermediate = pd.DataFrame(intermediate_results)
                        # Si el archivo ya existe, se añade sin cabecera
                        if os.path.exists(instance_file):
                            df_intermediate.to_csv(instance_file, index=False, sep=",", mode='a', header=False)
                        else:
                            df_intermediate.to_csv(instance_file, index=False, sep=",")
                        intermediate_results = []  # Reiniciar buffer

            # Guardar los resultados restantes si quedaron algunos
            if intermediate_results:
                df_intermediate = pd.DataFrame(intermediate_results)
                if os.path.exists(instance_file):
                    df_intermediate.to_csv(instance_file, index=False, sep=",", mode='a', header=False)
                else:
                    df_intermediate.to_csv(instance_file, index=False, sep=",")

            print(f"\nResultados guardados para {config_name} en {instance_file}")
            print(f"Se procesaron {len(overall_results)} nuevas muestras")
            
            # No extender resultados globales si no hay nuevos
            if overall_results:
                self.results.extend(overall_results)

        # Guardar resultados de esta partición solo si hay resultados
        if self.results:
            results_df = pd.DataFrame(self.results)
            results_suffix = f"random_part{actual_job}of{n_jobs}" if actual_job is not None else "random"
            results_path = os.path.join(self.output_dir, f"search_results_{results_suffix}.csv")
            results_df.to_csv(results_path, index=False)
            print(f"\nResultados de esta partición guardados en {results_path}")
            return results_df
        else:
            print("\nNo se generaron nuevos resultados")
            return pd.DataFrame()
    
    def _run_trial(self, params, config_name, config, param_names):
        """
        Ejecuta una prueba con una configuración específica de hiperparámetros.
        
        Args:
            params: Tupla con los valores de los parámetros
            config_name: Nombre de la configuración
            config: Datos de configuración
            param_names: Nombres de los parámetros
        
        Returns:
            dict: Resultados de la prueba
        """
        # Crear un diccionario con los parámetros
        param_dict = dict(zip(param_names, params))
        
        # Copiar configuración para no modificar la original
        config_copy = copy.deepcopy(config)
        
        # Crear la instancia de la heurística con la configuración actual
        heuristica = HeuristicaConstructivaEVs(config_copy)
        
        # Aplicar los hiperparámetros a la instancia
        # Estos son los posibles parámetros que podemos ajustar:
        # (basados en el código de la heurística original)
        
        # Parámetros para la fase de mejora
        if 'max_iteraciones' in param_dict:
            max_iteraciones = param_dict['max_iteraciones']
        else:
            max_iteraciones = 1000  # Valor por defecto
            
        if 'temperatura_inicial' in param_dict:
            temperatura_inicial = param_dict['temperatura_inicial']
        else:
            temperatura_inicial = 1.0  # Valor por defecto
            
        if 'factor_enfriamiento' in param_dict:
            factor_enfriamiento = param_dict['factor_enfriamiento']
        else:
            factor_enfriamiento = 0.95  # Valor por defecto
            
        if 'umbral_reinicio' in param_dict:
            umbral_reinicio = param_dict['umbral_reinicio']
        else:
            umbral_reinicio = 20  # Valor por defecto
        
        # Parámetros para las estrategias de perturbación
        if 'prob_perturbacion_intervalos' in param_dict:
            prob_perturbacion_intervalos = param_dict['prob_perturbacion_intervalos']
        else:
            prob_perturbacion_intervalos = 0.4  # Valor por defecto
            
        if 'prob_perturbacion_vehiculos' in param_dict:
            prob_perturbacion_vehiculos = param_dict['prob_perturbacion_vehiculos']
        else:
            prob_perturbacion_vehiculos = 0.3  # Valor por defecto
            
        if 'prob_perturbacion_hibrida' in param_dict:
            prob_perturbacion_hibrida = param_dict['prob_perturbacion_hibrida']
        else:
            prob_perturbacion_hibrida = 0.2  # Valor por defecto
            
        # Parámetros para la función de urgencia
        if 'factor_completitud' in param_dict:
            factor_completitud = param_dict['factor_completitud']
        else:
            factor_completitud = 1.0  # Valor por defecto
            
        if 'factor_ventana' in param_dict:
            factor_ventana = param_dict['factor_ventana']
        else:
            factor_ventana = 1.5  # Valor por defecto
        
        # Guardar la configuración original
        run_original = heuristica.run
        calcular_urgencia_original = heuristica._calcular_urgencia
        
        # Sobreescribir métodos con los nuevos hiperparámetros
        def run_method_modified(track_progress=False):
            """Versión modificada del método run con hiperparámetros personalizados"""
            print("Ejecutando HeurísticaConstructivaEVs mejorada...")
            
            # Fase 1: Construcción inicial de la solución con continuidad
            heuristica._fase_construccion_inicial()
            
            # Fase 2: Mejora mediante exploración del espacio de soluciones
            schedule_actual = heuristica.schedule.copy()
            mejor_schedule = schedule_actual.copy()
            mejor_costo = heuristica._evaluar_costo(mejor_schedule)
            intentos_sin_mejora = 0
            temperatura = temperatura_inicial
            
            # Para seguimiento de progreso
            costos_por_iteracion = []
            if track_progress:
                costos_por_iteracion.append(mejor_costo)
            
            print(f"Iniciando fase de mejora con {max_iteraciones} iteraciones...")
            for i in range(max_iteraciones):
                # Seleccionar estrategia de perturbación con probabilidades ajustadas
                p = random.random()
                prob_acumulada = 0
                
                prob_acumulada += prob_perturbacion_intervalos
                if p <= prob_acumulada:
                    estrategia = "Perturbacion_Intervalos"
                else:
                    prob_acumulada += prob_perturbacion_vehiculos
                    if p <= prob_acumulada:
                        estrategia = "Perturbacion_Vehiculos"
                    else:
                        prob_acumulada += prob_perturbacion_hibrida
                        if p <= prob_acumulada:
                            estrategia = "Perturbacion_Hibrida"
                        else:
                            estrategia = "Perturbacion_Sesiones"
                
                # Aplicar perturbación
                nuevo_schedule = heuristica._perturba_solucion_mejorado(schedule_actual, estrategia)
                
                # Reconstruir solución
                nuevo_schedule = heuristica._reconstruye_solucion(nuevo_schedule)
                
                # Evaluar nueva solución
                nuevo_costo = heuristica._evaluar_costo(nuevo_schedule)
                
                # Criterio de aceptación
                if nuevo_costo < mejor_costo:
                    mejor_schedule = nuevo_schedule.copy()
                    mejor_costo = nuevo_costo
                    schedule_actual = nuevo_schedule.copy()
                    intentos_sin_mejora = 0
                    if i % 10 == 0:
                        print(f"Iteración {i}: Nuevo mejor costo = {mejor_costo:.2f}")
                else:
                    # Aceptación probabilística (Simulated Annealing)
                    delta = nuevo_costo - mejor_costo
                    prob_aceptacion = np.exp(-delta / (temperatura * mejor_costo / 1000))
                    if random.random() < prob_aceptacion:
                        schedule_actual = nuevo_schedule.copy()
                    intentos_sin_mejora += 1
                
                # Registrar costo para seguimiento
                if track_progress:
                    costos_por_iteracion.append(mejor_costo)
                
                # Ajustar temperatura
                temperatura *= factor_enfriamiento
                
                # Reinicio si estancado
                if intentos_sin_mejora >= umbral_reinicio:
                    schedule_actual = mejor_schedule.copy()
                    intentos_sin_mejora = 0
                    temperatura = temperatura_inicial / 2  # Reiniciar con temperatura moderada
                    print(f"Iteración {i}: Reinicio tras {umbral_reinicio} intentos sin mejora. Temperatura = {temperatura:.4f}")
            
            print(f"Mejora completada. Costo final: {mejor_costo:.2f}")
            heuristica.schedule = mejor_schedule
            
            if track_progress:
                return heuristica.schedule, costos_por_iteracion
            else:
                return heuristica.schedule
        
        def calcular_urgencia_modified(ev_id, t_idx):
            """Versión modificada del método _calcular_urgencia con hiperparámetros personalizados"""
            # Tiempo restante hasta la salida (en horas)
            tiempo_hasta_salida = max(0.25, (heuristica.departure_time[ev_id] - heuristica.times[t_idx]))
            
            # Energía restante por entregar
            energia_restante = heuristica.energy_remaining[ev_id]
            
            # Factor de completitud - ajustado según hiperparámetros
            nivel_completitud = heuristica.energy_delivered[ev_id] / (heuristica.required_energy[ev_id] + 0.001)
            factor_completitud_ajustado = 1 + nivel_completitud * factor_completitud
            
            # Factor de ventana de tiempo - ajustado según hiperparámetros
            tiempo_total = max(0.25, (heuristica.departure_time[ev_id] - heuristica.arrival_time[ev_id]))
            factor_ventana_ajustado = min(factor_ventana, 1 + 2 * (tiempo_hasta_salida / tiempo_total))
            
            # Urgencia base: energía restante / tiempo restante
            urgencia_base = energia_restante / tiempo_hasta_salida if tiempo_hasta_salida > 0 else float('inf')
            
            # Urgencia ajustada con los factores adicionales
            return urgencia_base * factor_completitud_ajustado * factor_ventana_ajustado
        
        # Reemplazar métodos con versiones modificadas
        self.run_method_original = heuristica.run
        heuristica.run = run_method_modified
        heuristica._calcular_urgencia = calcular_urgencia_modified
        
        try:
            # Medir tiempo de ejecución
            start_time = time.time()
            schedule, costos_iteracion = heuristica.run(track_progress=True)
            execution_time = time.time() - start_time
            
            # Obtener estadísticas de la solución
            resultado = heuristica.get_resultados()
            stats = resultado["estadisticas"]
            
            # Crear registro con resultados
            result = {
                "config_name": config_name,
                "execution_time": execution_time,
                "final_cost": stats["costo_total"],
                "energy_satisfaction": stats["porcentaje_satisfaccion"],
                "evs_satisfied": stats["evs_satisfechos_completamente"],
                "evs_total": stats["evs_totales"],
                "evs_satisfaction_percentage": stats["porcentaje_evs_satisfechos"],
                "improvement_iterations": len(costos_iteracion),
                "cost_improvement": (costos_iteracion[0] - costos_iteracion[-1]) / costos_iteracion[0] * 100
            }
            
            # Añadir parámetros utilizados
            for name, value in param_dict.items():
                result[name] = value
                
            return result
            
        finally:
            # Restaurar métodos originales
            heuristica.run = run_original
            heuristica._calcular_urgencia = calcular_urgencia_original
    
    
    def analyze_results(self, output_dir=None, method="grid"):
        """
        Analiza los resultados combinados de todas las particiones.
        
        Args:
            output_dir: Directorio con los resultados (si es None, se usa self.output_dir)
            method: Método de búsqueda ("grid" o "random")
        
        Returns:
            DataFrame con los mejores hiperparámetros
        """
        if output_dir is None:
            output_dir = self.output_dir
            
        print(f"Analizando resultados combinados del método {method}...")
        
        # Buscar todos los archivos de resultados parciales
        results_files = glob.glob(os.path.join(output_dir, f"search_results_{method}_part*.csv"))
        instance_files = glob.glob(os.path.join(output_dir, f"instancia_*_{method}_part*.txt"))
        
        # Si no hay archivos particionados, buscar los archivos globales
        if not results_files:
            results_files = glob.glob(os.path.join(output_dir, f"search_results_{method}.csv"))
        
        if not instance_files:
            instance_files = glob.glob(os.path.join(output_dir, f"instancia_*_{method}.txt"))
        
        # Combinar todos los resultados
        all_results = []
        
        # Primero intentar con los archivos de resultados completos
        for file in results_files:
            try:
                df = pd.read_csv(file)
                all_results.append(df)
                print(f"Cargados {len(df)} resultados de {file}")
            except Exception as e:
                print(f"Error al cargar {file}: {e}")
        
        # Si no hay resultados, intentar con los archivos de instancia
        if not all_results:
            for file in instance_files:
                try:
                    df = pd.read_csv(file)
                    all_results.append(df)
                    print(f"Cargados {len(df)} resultados de {file}")
                except Exception as e:
                    print(f"Error al cargar {file}: {e}")
        
        if not all_results:
            raise FileNotFoundError(f"No se encontraron archivos de resultados para el método {method}")
        
        # Combinar todos los DataFrames
        results_df = pd.concat(all_results, ignore_index=True)
        print(f"Total de resultados combinados: {len(results_df)}")
        
        # Guardar resultados combinados
        combined_path = os.path.join(output_dir, f"{method}_search_results_combined.csv")
        results_df.to_csv(combined_path, index=False)
        print(f"Resultados combinados guardados en {combined_path}")
        
        # Para cada configuración, encontrar la mejor combinación de hiperparámetros
        config_names = results_df["config_name"].unique()
        best_params = []
        
        for config in config_names:
            config_results = results_df[results_df["config_name"] == config]
            
            # Ordenar por costo final (menor es mejor)
            best_result = config_results.sort_values("final_cost").iloc[0]
            
            # Mostrar mejores parámetros
            print(f"\nMejores parámetros para {config}:")
            for col in best_result.index:
                if col not in ["config_name", "execution_time", "final_cost", 
                               "energy_satisfaction", "evs_satisfied", "evs_total",
                               "evs_satisfaction_percentage", "improvement_iterations",
                               "cost_improvement"]:
                    print(f"  {col}: {best_result[col]}")
            
            print(f"Costo final: {best_result['final_cost']:.2f}")
            print(f"Satisfacción de energía: {best_result['energy_satisfaction']:.2f}%")
            print(f"EVs satisfechos: {best_result['evs_satisfied']}/{best_result['evs_total']} ({best_result['evs_satisfaction_percentage']:.2f}%)")
            print(f"Tiempo de ejecución: {best_result['execution_time']:.2f} segundos")
            print(f"Mejora de costo: {best_result['cost_improvement']:.2f}%")
            
            best_params.append(best_result.to_dict())
        
        # Crear DataFrame con los mejores parámetros
        best_params_df = pd.DataFrame(best_params)
        best_params_path = os.path.join(output_dir, f"{method}_best_parameters.csv")
        best_params_df.to_csv(best_params_path, index=False)
        
        print(f"\nMejores parámetros guardados en {best_params_path}")
        
        # Visualizar la importancia de los parámetros
        self._plot_parameter_importance(results_df, method=method)
        
        return best_params_df
    
    def _plot_parameter_importance(self, results_df):
        """
        Genera gráficos para visualizar la importancia de cada parámetro.
        
        Args:
            results_df: DataFrame con los resultados
        """
        # Identificar columnas de parámetros (excluyendo métricas y metadatos)
        metric_cols = ["config_name", "execution_time", "final_cost", 
                       "energy_satisfaction", "evs_satisfied", "evs_total",
                       "evs_satisfaction_percentage", "improvement_iterations",
                       "cost_improvement"]
        
        param_cols = [col for col in results_df.columns if col not in metric_cols]
        
        if not param_cols:
            print("No se encontraron columnas de parámetros para analizar")
            return
        
        # Para cada configuración, crear gráficos
        config_names = results_df["config_name"].unique()
        
        for config in config_names:
            config_results = results_df[results_df["config_name"] == config]
            
            # Crear directorio para gráficos si no existe
            plots_dir = os.path.join(self.output_dir, "plots", config)
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)
            
            # Para cada parámetro, crear un gráfico que muestre su efecto en el costo final
            for param in param_cols:
                if len(config_results[param].unique()) > 1:  # Solo parámetros que varían
                    plt.figure(figsize=(10, 6))
                    
                    # Si el parámetro tiene pocos valores únicos, usar boxplot
                    if len(config_results[param].unique()) <= 10:
                        plt.boxplot([config_results[config_results[param] == val]["final_cost"] 
                                    for val in sorted(config_results[param].unique())],
                                    labels=[str(val) for val in sorted(config_results[param].unique())])
                        plt.xlabel(param)
                        plt.ylabel("Costo Final")
                        plt.title(f"Efecto de {param} en el Costo Final - {config}")
                        plt.grid(True)
                    else:
                        # Si tiene muchos valores, usar scatter plot
                        plt.scatter(config_results[param], config_results["final_cost"], alpha=0.6)
                        plt.xlabel(param)
                        plt.ylabel("Costo Final")
                        plt.title(f"Efecto de {param} en el Costo Final - {config}")
                        plt.grid(True)
                    
                    # Guardar gráfico
                    plot_path = os.path.join(plots_dir, f"param_effect_{param}.png")
                    plt.savefig(plot_path)
                    plt.close()
            
            # Crear gráfico de dispersión para tiempo de ejecución vs. costo final
            plt.figure(figsize=(10, 6))
            plt.scatter(config_results["execution_time"], config_results["final_cost"], alpha=0.6)
            plt.xlabel("Tiempo de Ejecución (s)")
            plt.ylabel("Costo Final")
            plt.title(f"Tiempo de Ejecución vs. Costo Final - {config}")
            plt.grid(True)
            plot_path = os.path.join(plots_dir, "execution_time_vs_cost.png")
            plt.savefig(plot_path)
            plt.close()
            
            # Crear gráfico de mejora de costo vs. número de iteraciones
            plt.figure(figsize=(10, 6))
            plt.scatter(config_results["improvement_iterations"], config_results["cost_improvement"], alpha=0.6)
            plt.xlabel("Número de Iteraciones")
            plt.ylabel("Mejora de Costo (%)")
            plt.title(f"Iteraciones vs. Mejora de Costo - {config}")
            plt.grid(True)
            plot_path = os.path.join(plots_dir, "iterations_vs_improvement.png")
            plt.savefig(plot_path)
            plt.close()
        
        print(f"Gráficos generados en {os.path.join(self.output_dir, 'plots')}")


def main():
    """
    Función principal que ejecuta el benchmark de hiperparámetros.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark de hiperparámetros para la heurística constructiva")
    
    # Definir subcomandos
    subparsers = parser.add_subparsers(dest="command", help="Comando a ejecutar")
    
    # Subcomando para búsqueda de hiperparámetros estilo CLI (no interactivo)
    search_parser = subparsers.add_parser("search", help="Ejecutar búsqueda de hiperparámetros (modo CLI)")
    search_parser.add_argument("--config", type=str, help="Ruta a un archivo JSON específico")
    search_parser.add_argument("--data_dir", type=str, default="./results/cluster_results/representative_files", 
                        help="Directorio con archivos JSON")
    search_parser.add_argument("--output_dir", type=str, default="hyperparameter/hyperparameter_results_constructive",
                        help="Directorio para resultados")
    search_parser.add_argument("--method", type=str, choices=["grid", "random"], default="grid", 
                        help="Método de búsqueda")
    search_parser.add_argument("--n_jobs", type=int, default=1, 
                        help="Número total de particiones para dividir el trabajo")
    search_parser.add_argument("--actual_job", type=int, 
                        help="Índice de la partición actual (1 a n_jobs)")
    search_parser.add_argument("--n_trials", type=int, default=3, 
                        help="Número de pruebas por configuración")
    search_parser.add_argument("--n_samples", type=int, default=20, 
                        help="Número de muestras para búsqueda aleatoria")
    search_parser.add_argument("--analyze", action="store_true", 
                        help="Solo analizar resultados existentes sin ejecutar búsqueda")
    
    # Subcomando para ejecutar con los mejores parámetros ya encontrados
    run_parser = subparsers.add_parser("run", help="Ejecutar heurística con los mejores parámetros")
    run_parser.add_argument("json_path", type=str, help="Archivo JSON con la configuración")
    run_parser.add_argument("params_path", type=str, help="Archivo CSV con los mejores parámetros")
    run_parser.add_argument("--output_path", type=str, help="Ruta para guardar resultados")
    
    # Subcomando para ejecutar en modo interactivo (como el original)
    interactive_parser = subparsers.add_parser("interactive", help="Ejecutar en modo interactivo (como el original)")
    
    args = parser.parse_args()
    
    # Modo interactivo (como el main original)
    if args.command == "interactive" or args.command is None:
        # Buscar archivos de configuración
        data_dir = "../../results/cluster_results/representative_files"  # Directorio con los archivos JSON
        json_files = glob.glob(os.path.join(data_dir, "*.json"))
        
        if not json_files:
            print(f"No se encontraron archivos JSON en {data_dir}")
            print("Por favor, especifique la ubicación de los archivos de configuración.")
            data_dir = input("Directorio de datos: ")
            json_files = glob.glob(os.path.join(data_dir, "*.json"))
            
            if not json_files:
                print(f"No se encontraron archivos JSON en {data_dir}")
                return
        
        # Mostrar archivos disponibles y seleccionar un subconjunto
        print("Archivos de configuración encontrados:")
        for i, file in enumerate(json_files):
            print(f"{i+1}. {os.path.basename(file)}")
        
        selected_indices = input("Seleccione archivos a usar (números separados por comas, o 'all' para todos): ")
        
        if selected_indices.lower() == "all":
            selected_files = json_files
        else:
            try:
                indices = [int(idx.strip()) - 1 for idx in selected_indices.split(",")]
                selected_files = [json_files[idx] for idx in indices if 0 <= idx < len(json_files)]
            except:
                print("Selección inválida. Usando todos los archivos.")
                selected_files = json_files
        
        print(f"Se usarán {len(selected_files)} archivos para el benchmark")
        
        # Configurar el tuner
        output_dir = input("Directorio de salida (default: hyperparameter/hyperparameter_results_constructive): ").strip()
        if not output_dir:
            output_dir = "../../hyperparameter/hyperparameter_results_constructive"
        
        n_trials = input("Número de pruebas por configuración (default: 3): ").strip()
        if not n_trials:
            n_trials = 3
        else:
            n_trials = int(n_trials)
        
        tuner = HyperparameterTuner(selected_files, output_dir, n_trials)
        
        # Seleccionar método de búsqueda
        search_method = input("Método de búsqueda (grid/random): ").lower().strip()
        
        if search_method == "grid":
            # Definir parámetros para búsqueda en cuadrícula
            param_grid = {
                "max_iteraciones": [200, 500, 1000],
                "temperatura_inicial": [0.5, 1.0, 2.0],
                "factor_enfriamiento": [0.9, 0.95, 0.98],
                "umbral_reinicio": [10, 20, 30],
                "prob_perturbacion_intervalos": [0.3, 0.4, 0.5],
                "prob_perturbacion_vehiculos": [0.2, 0.3, 0.4],
                "factor_completitud": [0.5, 1.0, 1.5],
                "factor_ventana": [1.0, 1.5, 2.0]
            }
            
            # Ejecutar búsqueda en cuadrícula
            n_jobs = input("Número de procesos paralelos (default: 1): ").strip()
            if not n_jobs:
                n_jobs = 1
            else:
                n_jobs = int(n_jobs)
            
            results_df = tuner.grid_search(param_grid, n_jobs)
            
        elif search_method == "random":
            # Definir parámetros para búsqueda aleatoria
            param_distributions = {
                "max_iteraciones": (100, 2000),
                "temperatura_inicial": (0.1, 5.0),
                "factor_enfriamiento": (0.8, 0.99),
                "umbral_reinicio": (5, 50),
                "prob_perturbacion_intervalos": (0.1, 0.6),
                "prob_perturbacion_vehiculos": (0.1, 0.5),
                "prob_perturbacion_hibrida": (0.1, 0.4),
                "factor_completitud": (0.2, 2.0),
                "factor_ventana": (0.5, 3.0)
            }
            
            # Número de muestras aleatorias
            n_samples = input("Número de muestras aleatorias (default: 20): ").strip()
            if not n_samples:
                n_samples = 20
            else:
                n_samples = int(n_samples)
            
            # Ejecutar búsqueda aleatoria
            n_jobs = input("Número de procesos paralelos (default: 1): ").strip()
            if not n_jobs:
                n_jobs = 1
            else:
                n_jobs = int(n_jobs)
            
            results_df = tuner.random_search(param_distributions, n_samples, n_jobs)
            
        else:
            print("Método de búsqueda no reconocido. Usando búsqueda aleatoria con configuración por defecto.")
            param_distributions = {
                "max_iteraciones": (100, 1000),
                "temperatura_inicial": (0.5, 2.0),
                "factor_enfriamiento": (0.9, 0.99),
                "umbral_reinicio": (10, 30),
                "prob_perturbacion_intervalos": (0.3, 0.5),
                "prob_perturbacion_vehiculos": (0.2, 0.4),
                "factor_completitud": (0.5, 1.5),
                "factor_ventana": (1.0, 2.0)
            }
            results_df = tuner.random_search(param_distributions, 10, 1)
        
        # Analizar resultados
        best_params = tuner.analyze_results(results_df)
        
        # Guardar resultados en formato más legible
        with open(os.path.join(output_dir, "best_parameters_summary.txt"), "w") as f:
            f.write("RESUMEN DE MEJORES HIPERPARÁMETROS\n")
            f.write("=================================\n\n")
            
            for config in best_params["config_name"].unique():
                f.write(f"Configuración: {config}\n")
                f.write("-" * 50 + "\n")
                
                row = best_params[best_params["config_name"] == config].iloc[0]
                
                # Métricas
                f.write(f"Costo final: {row['final_cost']:.2f}\n")
                f.write(f"Satisfacción de energía: {row['energy_satisfaction']:.2f}%\n")
                f.write(f"EVs satisfechos: {row['evs_satisfied']}/{row['evs_total']} ({row['evs_satisfaction_percentage']:.2f}%)\n")
                f.write(f"Tiempo de ejecución: {row['execution_time']:.2f} segundos\n")
                f.write(f"Mejora de costo: {row['cost_improvement']:.2f}%\n\n")
                
                # Parámetros
                f.write("Mejores hiperparámetros:\n")
                for col in row.index:
                    if col not in ["config_name", "execution_time", "final_cost", 
                                "energy_satisfaction", "evs_satisfied", "evs_total",
                                "evs_satisfaction_percentage", "improvement_iterations",
                                "cost_improvement"]:
                        f.write(f"  {col}: {row[col]}\n")
                
                f.write("\n\n")
        
        print(f"Resumen guardado en {os.path.join(output_dir, 'best_parameters_summary.txt')}")
        
    # Modo búsqueda no interactivo (CLI)    
    elif args.command == "search":
        # Determinar qué archivos usar
        if args.config:
            json_files = [args.config]
        else:
            json_files = glob.glob(os.path.join(args.data_dir, "*.json"))
            if not json_files:
                print(f"No se encontraron archivos JSON en {args.data_dir}")
                exit(1)
        
        # Si solo se quiere analizar resultados
        if args.analyze:
            tuner = HyperparameterTuner(json_files, args.output_dir, args.n_trials)
            tuner.analyze_results(method=args.method)
            exit(0)
        
        # Crear directorio de salida si no existe
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        # Inicializar el tuner
        tuner = HyperparameterTuner(json_files, args.output_dir, args.n_trials)
        
        # Ejecutar la búsqueda según el método elegido
        if args.method == "grid":
            # Definir parámetros para búsqueda en cuadrícula
            param_grid = {
                "max_iteraciones": [200, 500, 1000],
                "temperatura_inicial": [0.5, 1.0, 2.0],
                "factor_enfriamiento": [0.8, 0.9, 0.98],
                "umbral_reinicio": [30],
                "prob_perturbacion_intervalos": [0.3, 0.5],
                "prob_perturbacion_vehiculos": [0.2, 0.4],
                "factor_completitud": [0.5, 1.0, 1.5],
                "factor_ventana": [1.0, 1.5, 2.0]
            }
            
            # Ejecutar búsqueda en cuadrícula con el trabajo específico
            results_df = tuner.grid_search(param_grid, n_jobs=args.n_jobs, actual_job=args.actual_job)
            
        else:  # random search
            # Definir parámetros para búsqueda aleatoria
            param_distributions = {
                "max_iteraciones": (100, 2000),
                "temperatura_inicial": (0.1, 5.0),
                "factor_enfriamiento": (0.8, 0.99),
                "umbral_reinicio": (5, 50),
                "prob_perturbacion_intervalos": (0.1, 0.6),
                "prob_perturbacion_vehiculos": (0.1, 0.5),
                "prob_perturbacion_hibrida": (0.1, 0.4),
                "factor_completitud": (0.2, 2.0),
                "factor_ventana": (0.5, 3.0)
            }
            
            # Ejecutar búsqueda aleatoria con el trabajo específico
            results_df = tuner.random_search(
                param_distributions, 
                n_samples=args.n_samples, 
                n_jobs=args.n_jobs, 
                actual_job=args.actual_job
            )
    
    # Modo run con los mejores parámetros
    elif args.command == "run":
        # Ejecutar la heurística con los mejores parámetros
        run_with_best_parameters(args.json_path, args.params_path, args.output_path)
    
    else:
        parser.print_help()



def run_with_best_parameters(json_path, params_path, output_path=None):
    """
    Ejecuta la heurística con los mejores parámetros encontrados.
    
    Args:
        json_path: Ruta al archivo JSON de configuración
        params_path: Ruta al archivo CSV con los mejores parámetros
        output_path: Ruta donde guardar los resultados (opcional)
    """
    # Cargar configuración
    config = load_data(json_path)
    
    # Cargar mejores parámetros
    best_params_df = pd.read_csv(params_path)
    config_name = os.path.splitext(os.path.basename(json_path))[0]
    
    # Encontrar parámetros para esta configuración
    if config_name in best_params_df["config_name"].values:
        params = best_params_df[best_params_df["config_name"] == config_name].iloc[0]
    else:
        print(f"No se encontraron parámetros para {config_name}, usando el primer conjunto de parámetros")
        params = best_params_df.iloc[0]
    
    # Extraer parámetros
    max_iteraciones = int(params.get("max_iteraciones", 1000))
    temperatura_inicial = float(params.get("temperatura_inicial", 1.0))
    factor_enfriamiento = float(params.get("factor_enfriamiento", 0.95))
    umbral_reinicio = int(params.get("umbral_reinicio", 20))
    prob_perturbacion_intervalos = float(params.get("prob_perturbacion_intervalos", 0.4))
    prob_perturbacion_vehiculos = float(params.get("prob_perturbacion_vehiculos", 0.3))
    prob_perturbacion_hibrida = float(params.get("prob_perturbacion_hibrida", 0.2))
    factor_completitud = float(params.get("factor_completitud", 1.0))
    factor_ventana = float(params.get("factor_ventana", 1.5))
    
    print(f"Ejecutando heurística para {config_name} con los mejores parámetros:")
    print(f"  max_iteraciones: {max_iteraciones}")
    print(f"  temperatura_inicial: {temperatura_inicial}")
    print(f"  factor_enfriamiento: {factor_enfriamiento}")
    print(f"  umbral_reinicio: {umbral_reinicio}")
    print(f"  prob_perturbacion_intervalos: {prob_perturbacion_intervalos}")
    print(f"  prob_perturbacion_vehiculos: {prob_perturbacion_vehiculos}")
    print(f"  prob_perturbacion_hibrida: {prob_perturbacion_hibrida}")
    print(f"  factor_completitud: {factor_completitud}")
    print(f"  factor_ventana: {factor_ventana}")
    
    # Crear instancia de heurística
    heuristica = HeuristicaConstructivaEVs(config)
    
    # Modificar método run y calcular_urgencia para usar los mejores parámetros
    run_original = heuristica.run
    calcular_urgencia_original = heuristica._calcular_urgencia
    
    def run_with_best_params(track_progress=False):
        """Versión modificada del método run con los mejores hiperparámetros"""
        print("Ejecutando HeurísticaConstructivaEVs con parámetros optimizados...")
        
        # Fase 1: Construcción inicial de la solución con continuidad
        heuristica._fase_construccion_inicial()
        
        # Fase 2: Mejora mediante exploración del espacio de soluciones
        schedule_actual = heuristica.schedule.copy()
        mejor_schedule = schedule_actual.copy()
        mejor_costo = heuristica._evaluar_costo(mejor_schedule)
        intentos_sin_mejora = 0
        temperatura = temperatura_inicial
        
        # Para seguimiento de progreso
        costos_por_iteracion = []
        if track_progress:
            costos_por_iteracion.append(mejor_costo)
        
        print(f"Iniciando fase de mejora con {max_iteraciones} iteraciones...")
        for i in range(max_iteraciones):
            # Seleccionar estrategia de perturbación con probabilidades ajustadas
            p = random.random()
            prob_acumulada = 0
            
            prob_acumulada += prob_perturbacion_intervalos
            if p <= prob_acumulada:
                estrategia = "Perturbacion_Intervalos"
            else:
                prob_acumulada += prob_perturbacion_vehiculos
                if p <= prob_acumulada:
                    estrategia = "Perturbacion_Vehiculos"
                else:
                    prob_acumulada += prob_perturbacion_hibrida
                    if p <= prob_acumulada:
                        estrategia = "Perturbacion_Hibrida"
                    else:
                        estrategia = "Perturbacion_Sesiones"
            
            # Aplicar perturbación
            nuevo_schedule = heuristica._perturba_solucion_mejorado(schedule_actual, estrategia)
            
            # Reconstruir solución
            nuevo_schedule = heuristica._reconstruye_solucion(nuevo_schedule)
            
            # Evaluar nueva solución
            nuevo_costo = heuristica._evaluar_costo(nuevo_schedule)
            
            # Criterio de aceptación
            if nuevo_costo < mejor_costo:
                mejor_schedule = nuevo_schedule.copy()
                mejor_costo = nuevo_costo
                schedule_actual = nuevo_schedule.copy()
                intentos_sin_mejora = 0
                if i % 10 == 0:
                    print(f"Iteración {i}: Nuevo mejor costo = {mejor_costo:.2f}")
            else:
                # Aceptación probabilística (Simulated Annealing)
                delta = nuevo_costo - mejor_costo
                prob_aceptacion = np.exp(-delta / (temperatura * mejor_costo / 1000))
                if random.random() < prob_aceptacion:
                    schedule_actual = nuevo_schedule.copy()
                intentos_sin_mejora += 1
            
            # Registrar costo para seguimiento
            if track_progress:
                costos_por_iteracion.append(mejor_costo)
            
            # Ajustar temperatura
            temperatura *= factor_enfriamiento
            
            # Reinicio si estancado
            if intentos_sin_mejora >= umbral_reinicio:
                schedule_actual = mejor_schedule.copy()
                intentos_sin_mejora = 0
                temperatura = temperatura_inicial / 2  # Reiniciar con temperatura moderada
                print(f"Iteración {i}: Reinicio tras {umbral_reinicio} intentos sin mejora. Temperatura = {temperatura:.4f}")
        
        print(f"Mejora completada. Costo final: {mejor_costo:.2f}")
        heuristica.schedule = mejor_schedule
        
        if track_progress:
            return heuristica.schedule, costos_por_iteracion
        else:
            return heuristica.schedule
    
    def calcular_urgencia_with_best_params(ev_id, t_idx):
        """Versión modificada del método _calcular_urgencia con los mejores hiperparámetros"""
        # Tiempo restante hasta la salida (en horas)
        tiempo_hasta_salida = max(0.25, (heuristica.departure_time[ev_id] - heuristica.times[t_idx]))
        
        # Energía restante por entregar
        energia_restante = heuristica.energy_remaining[ev_id]
        
        # Factor de completitud - ajustado según hiperparámetros
        nivel_completitud = heuristica.energy_delivered[ev_id] / (heuristica.required_energy[ev_id] + 0.001)
        factor_completitud_ajustado = 1 + nivel_completitud * factor_completitud
        
        # Factor de ventana de tiempo - ajustado según hiperparámetros
        tiempo_total = max(0.25, (heuristica.departure_time[ev_id] - heuristica.arrival_time[ev_id]))
        factor_ventana_ajustado = min(factor_ventana, 1 + 2 * (tiempo_hasta_salida / tiempo_total))
        
        # Urgencia base: energía restante / tiempo restante
        urgencia_base = energia_restante / tiempo_hasta_salida if tiempo_hasta_salida > 0 else float('inf')
        
        # Urgencia ajustada con los factores adicionales
        return urgencia_base * factor_completitud_ajustado * factor_ventana_ajustado
    
    # Reemplazar métodos con versiones optimizadas
    heuristica.run = run_with_best_params
    heuristica._calcular_urgencia = calcular_urgencia_with_best_params
    
    try:
        # Ejecutar heurística y medir tiempo
        start_time = time.time()
        schedule, costos_iteracion = heuristica.run(track_progress=True)
        execution_time = time.time() - start_time
        
        # Obtener resultados
        resultado = heuristica.get_resultados()
        stats = resultado["estadisticas"]
        
        print("\nResultados con parámetros optimizados:")
        print(f"- Costo total: ${stats['costo_total']:.2f}")
        print(f"- Energía requerida total: {stats['energia_requerida_total']:.2f} kWh")
        print(f"- Energía entregada total: {stats['energia_entregada_total']:.2f} kWh")
        print(f"- Porcentaje de satisfacción: {stats['porcentaje_satisfaccion']:.2f}%")
        print(f"- EVs satisfechos completamente: {stats['evs_satisfechos_completamente']}/{stats['evs_totales']} ({stats['porcentaje_evs_satisfechos']:.2f}%)")
        print(f"- Porcentaje de EVs con alguna carga: {stats['porcentaje_evs_con_alguna_carga']:.2f}%")
        print(f"- Tiempo de ejecución: {execution_time:.2f} segundos")
        
        # Guardar resultados
        if output_path is None:
            output_dir = "optimized_results"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, f"{config_name}_optimized.json")
        
        # Añadir extra_info a los resultados
        resultado["extra_info"] = {
            "optimized_parameters": {
                "max_iteraciones": max_iteraciones,
                "temperatura_inicial": temperatura_inicial,
                "factor_enfriamiento": factor_enfriamiento,
                "umbral_reinicio": umbral_reinicio,
                "prob_perturbacion_intervalos": prob_perturbacion_intervalos,
                "prob_perturbacion_vehiculos": prob_perturbacion_vehiculos,
                "prob_perturbacion_hibrida": prob_perturbacion_hibrida,
                "factor_completitud": factor_completitud,
                "factor_ventana": factor_ventana
            },
            "execution_time": execution_time
        }
        
        save_schedule_to_json(resultado, output_path)
        print(f"Resultados guardados en {output_path}")
        
        # Generar gráficos
        config_data = config
        
        # Graficar perfiles de carga
        from modules.constructive.heuristic import plot_charging_schedule
        fig_charge = plot_charging_schedule(config_data, resultado["schedule"])
        fig_charge.savefig(os.path.join(os.path.dirname(output_path), f"{config_name}_optimized_carga.png"))
        plt.close(fig_charge)
        
        # Graficar asignación de parqueo
        from modules.constructive.heuristic import plot_parking_schedule
        fig_parking = plot_parking_schedule(config_data, resultado["schedule"])
        fig_parking.savefig(os.path.join(os.path.dirname(output_path), f"{config_name}_optimized_parqueo.png"))
        plt.close(fig_parking)
        
        # Graficar evolución del costo
        from modules.constructive.heuristic import plot_cost_evolution
        fig_cost = plot_cost_evolution(costos_iteracion, f"Evolución del costo (parámetros optimizados) - {config_name}")
        fig_cost.savefig(os.path.join(os.path.dirname(output_path), f"{config_name}_optimized_evolucion_costo.png"))
        plt.close(fig_cost)
        
        return resultado
        
    finally:
        # Restaurar métodos originales 
        heuristica.run = run_original
        heuristica._calcular_urgencia = calcular_urgencia_original

if __name__ == "__main__":
    main()