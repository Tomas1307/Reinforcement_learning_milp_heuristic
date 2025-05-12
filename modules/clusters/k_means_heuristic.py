import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import random

# Configuración de rutas predeterminadas
REPRESENTATIVE_FILES_DIR = "../../results/cluster_results/representative_files"
HYPERPARAMETER_RESULTS_DIR = "../../hyperparameter/hyperparameter_results_constructive"
BEST_PARAMS_FILE = os.path.join(HYPERPARAMETER_RESULTS_DIR, "best_parameters_summary.txt")

# Cargar hiperparámetros óptimos desde los resultados del análisis
def load_best_hyperparameters():
    """
    Carga los hiperparámetros óptimos para cada sistema desde los archivos csv.
    
    Returns:
        dict: Diccionario con los mejores hiperparámetros para cada sistema
    """
    # Valores predeterminados basados en el análisis previo
    best_params = {
        "test_system_2": {
            "max_iteraciones": 500,
            "temperatura_inicial": 1.0,
            "factor_enfriamiento": 0.98,
            "umbral_reinicio": 30,
            "prob_perturbacion_intervalos": 0.3,
            "prob_perturbacion_vehiculos": 0.2,
            "factor_completitud": 1.5,
            "factor_ventana": 2.0
        },
        "test_system_9": {
            "max_iteraciones": 500,
            "temperatura_inicial": 0.5,
            "factor_enfriamiento": 0.9,
            "umbral_reinicio": 30,
            "prob_perturbacion_intervalos": 0.5,
            "prob_perturbacion_vehiculos": 0.2,
            "factor_completitud": 1.0,
            "factor_ventana": 1.5
        },
        "test_system_10": {
            "max_iteraciones": 200,
            "temperatura_inicial": 0.5,
            "factor_enfriamiento": 0.8,
            "umbral_reinicio": 30,
            "prob_perturbacion_intervalos": 0.5,
            "prob_perturbacion_vehiculos": 0.4,
            "factor_completitud": 0.5,
            "factor_ventana": 2.0
        }
    }
    
    # Intentar cargar desde archivos CSV
    try:
        for system in ["2", "9", "10"]:
            file_path = os.path.join(HYPERPARAMETER_RESULTS_DIR, f"instancia_test_system_{system}_grid_complete.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                if not df.empty:
                    # Encontrar la fila con el menor costo
                    best_row = df.loc[df['final_cost'].idxmin()]
                    
                    # Extraer parámetros
                    params = {}
                    for param in ["max_iteraciones", "temperatura_inicial", "factor_enfriamiento", 
                                 "umbral_reinicio", "prob_perturbacion_intervalos", 
                                 "prob_perturbacion_vehiculos", "factor_completitud", "factor_ventana"]:
                        if param in best_row:
                            params[param] = best_row[param]
                    
                    if params:
                        best_params[f"test_system_{system}"] = params
                        print(f"Cargados hiperparámetros óptimos para test_system_{system} desde CSV")
    except Exception as e:
        print(f"Error al cargar desde CSV: {e}")
        print("Usando hiperparámetros predeterminados")
    
    return best_params

# Cargar y extraer características de las instancias de referencia desde los archivos JSON
def load_reference_instances():
    """
    Carga y extrae características de las instancias de referencia.
    
    Returns:
        tuple: (reference_features, reference_names, scaler)
    """
    reference_features = []
    reference_names = []
    
    # Buscar instancias de referencia
    ref_files = []
    for system in ["2", "9", "10"]:
        ref_path = os.path.join(REPRESENTATIVE_FILES_DIR, f"test_system_{system}.json")
        if os.path.exists(ref_path):
            ref_files.append((f"test_system_{system}", ref_path))
    
    if not ref_files:
        print("No se encontraron instancias de referencia. Usando valores predeterminados.")
        # Usar características predeterminadas si no hay archivos
        reference_data = {
            "test_system_2": {
                'num_evs': 23,
                'num_spots': 5,
                'num_chargers': 3,
                'station_limit': 30.0,
                'time_window': 24.0,
                'total_required_energy': 250.75,
                'avg_required_energy': 10.9,
                'avg_stay_time': 3.5,
                'price_std': 0.08,
                'price_mean': 0.15,
                'price_range': 0.25,
                'demand_supply_ratio': 0.12,
                'avg_vehicles_present': 3.2,
                'max_vehicles_present': 5,
                'vehicle_congestion': 1.0
            },
            "test_system_9": {
                'num_evs': 278,
                'num_spots': 15,
                'num_chargers': 8,
                'station_limit': 150.0,
                'time_window': 24.0,
                'total_required_energy': 3245.65,
                'avg_required_energy': 11.7,
                'avg_stay_time': 4.2,
                'price_std': 0.09,
                'price_mean': 0.18,
                'price_range': 0.3,
                'demand_supply_ratio': 0.35,
                'avg_vehicles_present': 10.5,
                'max_vehicles_present': 15,
                'vehicle_congestion': 1.0
            },
            "test_system_10": {
                'num_evs': 554,
                'num_spots': 20,
                'num_chargers': 12,
                'station_limit': 200.0,
                'time_window': 24.0,
                'total_required_energy': 6785.25,
                'avg_required_energy': 12.2,
                'avg_stay_time': 3.8,
                'price_std': 0.07,
                'price_mean': 0.16,
                'price_range': 0.28,
                'demand_supply_ratio': 0.45,
                'avg_vehicles_present': 16.3,
                'max_vehicles_present': 20,
                'vehicle_congestion': 1.0
            }
        }
        
        for name, features in reference_data.items():
            reference_names.append(name)
            reference_features.append(list(features.values()))
    else:
        # Cargar desde archivos JSON
        from modules.constructive.heuristic import load_data
        
        for name, path in ref_files:
            try:
                config = load_data(path)
                features, _ = extract_instance_features(config)
                reference_names.append(name)
                reference_features.append(features)
                print(f"Cargadas características para {name} desde {path}")
            except Exception as e:
                print(f"Error al cargar {name}: {e}")
    
    # Escalar las características
    scaler = StandardScaler()
    reference_features_scaled = scaler.fit_transform(reference_features)
    
    return reference_features_scaled, reference_names, scaler

# Extrae características de una instancia para el clustering
def extract_instance_features(config):
    """
    Extrae características relevantes de una instancia para el clustering.
    
    Args:
        config: Diccionario con la configuración del sistema
        
    Returns:
        list, dict: Vector de características y diccionario de características
    """
    features = {}
    
    # Número de vehículos
    features['num_evs'] = len(config['arrivals'])
    
    # Número de spots y cargadores
    features['num_spots'] = config['n_spots']
    features['num_chargers'] = len(config['chargers'])
    
    # Capacidad total de la estación
    features['station_limit'] = config['station_limit']
    
    # Ventana temporal total
    if config['times']:
        features['time_window'] = config['times'][-1] - config['times'][0]
    else:
        features['time_window'] = 0
    
    # Demanda de energía total
    total_required_energy = sum(arrival['required_energy'] for arrival in config['arrivals'])
    features['total_required_energy'] = total_required_energy
    
    # Promedio de energía requerida por vehículo
    features['avg_required_energy'] = total_required_energy / features['num_evs'] if features['num_evs'] > 0 else 0
    
    # Tiempo promedio de estancia de los vehículos
    total_stay_time = sum(arrival['departure_time'] - arrival['arrival_time'] for arrival in config['arrivals'])
    features['avg_stay_time'] = total_stay_time / features['num_evs'] if features['num_evs'] > 0 else 0
    
    # Variación de precios
    if config['prices']:
        features['price_std'] = np.std(config['prices'])
        features['price_mean'] = np.mean(config['prices'])
        features['price_range'] = max(config['prices']) - min(config['prices'])
    else:
        features['price_std'] = 0
        features['price_mean'] = 0
        features['price_range'] = 0
    
    # Ratio demanda/oferta
    power_capacity = sum(c['power'] for c in config['chargers'])
    if features['time_window'] > 0 and power_capacity > 0:
        features['demand_supply_ratio'] = total_required_energy / (power_capacity * features['time_window'])
    else:
        features['demand_supply_ratio'] = 0
    
    # Número promedio de vehículos presentes simultáneamente
    if config['times']:
        vehicles_present = []
        for t in config['times']:
            count = sum(1 for arr in config['arrivals'] if arr['arrival_time'] <= t < arr['departure_time'])
            vehicles_present.append(count)
        features['avg_vehicles_present'] = np.mean(vehicles_present)
        features['max_vehicles_present'] = max(vehicles_present)
    else:
        features['avg_vehicles_present'] = 0
        features['max_vehicles_present'] = 0
    
    # Congestión de vehículos
    features['vehicle_congestion'] = features['max_vehicles_present'] / features['num_spots'] if features['num_spots'] > 0 else 0
    
    # Convertir a vector
    feature_vector = list(features.values())
    
    return feature_vector, features

# Asignar instancia a uno de los clusters de referencia
def assign_to_cluster(instance_features, reference_features, reference_names, scaler):
    """
    Asigna una instancia al clusters más cercano.
    
    Args:
        instance_features: Vector de características de la instancia
        reference_features: Matriz de características de instancias de referencia
        reference_names: Nombres de las instancias de referencia
        scaler: Objeto StandardScaler para normalizar las características
        
    Returns:
        str: Nombre del clusters asignado
    """
    # Normalizar características de la instancia
    instance_features_scaled = scaler.transform([instance_features])
    
    # Calcular distancias a todas las instancias de referencia
    distances = cdist(instance_features_scaled, reference_features, 'euclidean')[0]
    
    # Encontrar la instancia de referencia más cercana
    closest_idx = np.argmin(distances)
    closest_system = reference_names[closest_idx]
    
    # Guardar las distancias para mostrar
    distances_dict = {name: dist for name, dist in zip(reference_names, distances)}
    
    return closest_system, distances_dict

# Visualización de la asignación
def visualize_clustering(instance_features, instance_name, reference_features, reference_names, scaler, assigned_cluster):
    """
    Visualiza la asignación de la instancia a un clusters usando PCA para reducción de dimensionalidad.
    
    Args:
        instance_features: Vector de características de la instancia
        instance_name: Nombre de la instancia
        reference_features: Matriz de características de instancias de referencia
        reference_names: Nombres de las instancias de referencia
        scaler: Objeto StandardScaler para normalizar las características
        assigned_cluster: Nombre del clusters asignado
    """
    from sklearn.decomposition import PCA
    
    # Normalizar características de la instancia
    instance_features_scaled = scaler.transform([instance_features])
    
    # Combinar instancia actual con las de referencia
    all_features = np.vstack([reference_features, instance_features_scaled])
    all_names = reference_names + [instance_name]
    
    # Aplicar PCA para visualización 2D
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(all_features)
    
    # Separar puntos de referencia y la instancia actual
    ref_points = features_2d[:len(reference_features)]
    instance_point = features_2d[len(reference_features):]
    
    # Crear gráfico
    plt.figure(figsize=(10, 8))
    
    # Graficar puntos de referencia
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    for i, (name, point) in enumerate(zip(reference_names, ref_points)):
        marker_color = colors[i % len(colors)]
        highlight = 'red' if name == assigned_cluster else marker_color
        plt.scatter(point[0], point[1], s=300, alpha=0.6, color=highlight, label=name, 
                   edgecolors='black', linewidth=2 if name == assigned_cluster else 1)
        plt.annotate(name, (point[0], point[1]), fontsize=12)
    
    # Graficar instancia actual
    plt.scatter(instance_point[0][0], instance_point[0][1], s=500, alpha=0.6, color='cyan', 
               label=instance_name, edgecolors='black', linewidth=2)
    plt.annotate(instance_name, (instance_point[0][0], instance_point[0][1]), fontsize=12)
    
    # Línea que conecta la instancia con el clusters asignado
    for i, name in enumerate(reference_names):
        if name == assigned_cluster:
            plt.plot([instance_point[0][0], ref_points[i][0]], 
                     [instance_point[0][1], ref_points[i][1]], 
                     'k--', alpha=0.5, linewidth=2)
    
    plt.title(f'Asignación de {instance_name} al clusters {assigned_cluster}')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Guardar gráfico
    plot_dir = "../../hyperparameter/hyperparameter_analysis_constructive/cluster_plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    plt.savefig(os.path.join(plot_dir, f"cluster_assignment_{instance_name}.png"))
    plt.close()

# Función principal para obtener los mejores hiperparámetros para una instancia
def get_best_hyperparameters_for_instance(config, instance_name="new_instance", visualize=True):
    """
    Obtiene los mejores hiperparámetros para una instancia basado en clustering.
    
    Args:
        config: Diccionario con la configuración del sistema
        instance_name: Nombre de la instancia
        visualize: Si es True, genera visualización del clustering
        
    Returns:
        dict: Mejores hiperparámetros para la instancia
    """
    # Cargar hiperparámetros óptimos
    best_params = load_best_hyperparameters()
    
    # Extraer características de la instancia
    instance_features, features_dict = extract_instance_features(config)
    
    # Cargar instancias de referencia
    reference_features, reference_names, scaler = load_reference_instances()
    
    # Asignar instancia al clusters más cercano
    assigned_cluster, distances = assign_to_cluster(
        instance_features, reference_features, reference_names, scaler
    )
    
    print(f"\nAsignación de clusters para {instance_name}:")
    for system, distance in distances.items():
        print(f"  Distancia a {system}: {distance:.4f}")
    print(f"Instancia asignada al clusters: {assigned_cluster}")
    
    # Visualizar asignación
    if visualize:
        visualize_clustering(
            instance_features, instance_name, reference_features, 
            reference_names, scaler, assigned_cluster
        )
    
    # Devolver los hiperparámetros del clusters asignado
    selected_params = best_params.get(assigned_cluster)
    
    # Mostrar hiperparámetros seleccionados
    print(f"\nHiperparámetros seleccionados para {instance_name} (basados en {assigned_cluster}):")
    for param, value in selected_params.items():
        print(f"  {param}: {value}")
    
    return selected_params, assigned_cluster, features_dict

# Función para integrar en heuristic.py
def modify_heuristic_with_best_params(heuristica, config, instance_name="new_instance"):
    """
    Modifica la heurística para utilizar los mejores hiperparámetros basados en clustering.
    
    Args:
        heuristica: Instancia de HeuristicaConstructivaEVs
        config: Configuración del sistema
        instance_name: Nombre de la instancia
        
    Returns:
        tuple: (heuristica modificada, hiperparámetros seleccionados, clusters asignado, features)
    """
    # Obtener los mejores hiperparámetros para esta instancia
    selected_params, assigned_cluster, features = get_best_hyperparameters_for_instance(
        config, instance_name
    )
    
    # Imprimir a qué clusters se asignó la instancia
    print(f"La instancia '{instance_name}' ha sido asignada al clusters: {assigned_cluster}")
    
    # Agregar un print para mostrar los hiperparámetros seleccionados
    print(f"[modify_heuristic_with_best_params] Hiperparámetros seleccionados para {instance_name} (clusters {assigned_cluster}):")
    for param, value in selected_params.items():
        print(f"  {param}: {value}")
    
    # Guardar los métodos originales
    run_original = heuristica.run
    calcular_urgencia_original = heuristica._calcular_urgencia
    
    # Extraer parámetros
    max_iteraciones = selected_params.get('max_iteraciones', 1000)
    temperatura_inicial = selected_params.get('temperatura_inicial', 1.0)
    factor_enfriamiento = selected_params.get('factor_enfriamiento', 0.95)
    umbral_reinicio = selected_params.get('umbral_reinicio', 20)
    prob_perturbacion_intervalos = selected_params.get('prob_perturbacion_intervalos', 0.4)
    prob_perturbacion_vehiculos = selected_params.get('prob_perturbacion_vehiculos', 0.3)
    prob_perturbacion_hibrida = selected_params.get('prob_perturbacion_hibrida', 0.2)
    factor_completitud = selected_params.get('factor_completitud', 1.0)
    factor_ventana = selected_params.get('factor_ventana', 1.5)
    
    # Definir método run modificado
    def run_with_best_params(track_progress=False):
        """Versión modificada del método run con los mejores hiperparámetros"""
        print(f"Ejecutando HeurísticaConstructivaEVs con parámetros óptimos para clusters {assigned_cluster}...")
        print(f"max_iteraciones: {max_iteraciones}, temp_inicial: {temperatura_inicial}, factor_enfriamiento: {factor_enfriamiento}")
        
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
    
    # Definir método calcular_urgencia modificado
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
    
    # Almacenar los métodos originales para restaurarlos si es necesario
    heuristica._run_original = run_original
    heuristica._calcular_urgencia_original = calcular_urgencia_original
    
    return heuristica, selected_params, assigned_cluster, features


# Este script puede usarse como módulo independiente o ejecutarse directamente
if __name__ == "__main__":
    import sys
    import importlib.util
    
    # Intentar importar heuristic.py
    heuristic_path = "../constructive/heuristic.py"
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("Uso: python hyperparameter_selector.py [ruta_json] [output_dir]")
            print("Si no se especifica ruta_json, se analizan los hiperparámetros sin aplicarlos")
            sys.exit(0)
        json_path = sys.argv[1]
        if os.path.isfile(json_path) and json_path.endswith(".py"):
            heuristic_path = json_path
            json_path = sys.argv[2] if len(sys.argv) > 2 else None
        else:
            json_path = sys.argv[1]
    else:
        json_path = None
    
    # Cargar el módulo heuristic
    try:
        spec = importlib.util.spec_from_file_location("heuristic", heuristic_path)
        heuristic = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(heuristic)
        print(f"Módulo heuristic.py cargado correctamente desde {heuristic_path}")
    except Exception as e:
        print(f"Error al cargar heuristic.py: {e}")
        print("Análisis de hiperparámetros no funcionará completamente sin heuristic.py")
        heuristic = None
    
    # Si se proporciona un archivo JSON, analizar y aplicar hiperparámetros
    if json_path and heuristic is not None:
        print(f"Analizando instancia: {json_path}")
        
        config = heuristic.load_data(json_path)
        instance_name = os.path.splitext(os.path.basename(json_path))[0]
        
        # Crear la heurística
        heuristica = heuristic.HeuristicaConstructivaEVs(config)
        
        # Aplicar hiperparámetros óptimos
        heuristica, selected_params, assigned_cluster, features = modify_heuristic_with_best_params(
            heuristica, config, instance_name
        )
        
        # Ejecutar la heurística y medir tiempo
        import time
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
        if len(sys.argv) > 3:
            output_dir = sys.argv[3]
        else:
            output_dir = "optimized_results"
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
                
        output_path = os.path.join(output_dir, f"{instance_name}_optimized.json")
        
        # Añadir extra_info a los resultados
        resultado["extra_info"] = {
            "optimized_parameters": selected_params,
            "assigned_cluster": assigned_cluster,
            "instance_features": features,
            "execution_time": execution_time
        }
        
        heuristic.save_schedule_to_json(resultado, output_path)
        print(f"Resultados guardados en {output_path}")
        
        # Generar gráficos
        # Graficar perfiles de carga
        fig_charge = heuristic.plot_charging_schedule(config, resultado["schedule"])
        fig_charge.savefig(os.path.join(output_dir, f"{instance_name}_optimized_carga.png"))
        plt.close(fig_charge)
        
        # Graficar asignación de parqueo
        fig_parking = heuristic.plot_parking_schedule(config, resultado["schedule"])
        fig_parking.savefig(os.path.join(output_dir, f"{instance_name}_optimized_parqueo.png"))
        plt.close(fig_parking)
        
        # Graficar evolución del costo
        fig_cost = heuristic.plot_cost_evolution(costos_iteracion, f"Evolución del costo - {instance_name}")
        fig_cost.savefig(os.path.join(output_dir, f"{instance_name}_optimized_evolucion_costo.png"))
        plt.close(fig_cost)