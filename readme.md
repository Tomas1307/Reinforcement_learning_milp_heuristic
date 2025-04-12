# Sistema de Optimización de Carga para Vehículos Eléctricos

Este proyecto implementa un sistema completo para la gestión y optimización de la carga de vehículos eléctricos en estaciones de carga, utilizando diferentes enfoques algorítmicos:

1. **Heurística constructiva con mejoras adaptativas** - Algoritmo personalizado con búsqueda local
2. **Aprendizaje por refuerzo (RL)** - Utilizando Deep Q-Networks (DQN)
3. **Programación lineal entera mixta (MILP)** - Para refinamiento de soluciones

## Estructura del Proyecto

```
.
├── checkpoints/           # Puntos de control para modelos de RL
├── cluster_results/       # Resultados del análisis de clustering
│   └── representative_files/  # Instancias representativas
├── data/                  # Instancias de prueba
├── heuristic_results/     # Resultados generados por el algoritmo heurístico
├── hyperparameter_analysis/ # Análisis de hiperparámetros
├── hyperparameter_results/ # Resultados de la búsqueda de hiperparámetros 
├── plots/                 # Gráficos generados por el sistema
├── results/               # Resultados generados por RL y MILP
├── clusters.py            # Análisis de clustering de instancias
├── ev_scheduler_generalized_model.h5 # Modelo DQN pre-entrenado
├── heuristic.py           # Implementación de la heurística constructiva
├── hyperparameter_tuning.py # Búsqueda y optimización de hiperparámetros
├── k_means_heuristic.py   # Selección de parámetros basada en clustering
├── model131.py            # Implementación del modelo RL y MILP
├── plot_tables.py         # Generación de tablas para reportes
├── plots_heuristics.py    # Visualización de resultados de la heurística
├── plots_milp.py          # Visualización de resultados MILP/RL
├── pseudocodigo_heuristica.md # Documentación del algoritmo heurístico
├── results.py             # Análisis comparativo de resultados
```

## Características del Sistema

- **Generalización**: El sistema puede adaptarse a diferentes tipos de instancias con variabilidad en número de EVs, plazas, cargadores y patrones de precios.
- **Optimización de múltiples objetivos**: Minimiza el costo de carga mientras maximiza la satisfacción de la demanda energética.
- **Selección adaptativa de parámetros**: Utiliza clustering para seleccionar automáticamente los mejores hiperparámetros según las características de la instancia.
- **Visualización avanzada**: Múltiples herramientas de visualización para evaluar la calidad de las soluciones.
- **Análisis de capacidad**: Herramientas para simular y analizar diferentes escenarios de capacidad de estacionamiento.

## Componentes Principales

### 1. Algoritmo Heurístico (`heuristic.py`)

La heurística constructiva mejorada implementa una solución inicial seguida de una fase de mejora mediante búsqueda local. Incluye:

- Construcción inicial con priorización inteligente
- Uso de simulated annealing para exploración del espacio de soluciones
- Estrategias adaptativas de perturbación
- Consolidación de sesiones de carga

### 2. Aprendizaje por Refuerzo (`model131.py`)

El componente de RL implementa un agente DQN con las siguientes características:

- Estados generalizados que capturan las características relevantes del problema
- Acciones discretizadas para manejar espacios de acción complejos
- Memoria de experiencia por sistema para transferencia de conocimiento
- Protecciones contra errores para robustez en producción

### 3. Refinamiento con MILP (`model131.py`)

El componente MILP implementa un modelo exacto para refinamiento de soluciones:

- Variables de decisión para asignación de cargadores y potencia
- Restricciones de capacidad para slots, cargadores y transformador
- Manejo de violaciones mediante variables de holgura
- Función objetivo que minimiza costo y penaliza energía no satisfecha

### 4. Análisis de Clustering (`clusters.py`)

El sistema utiliza clustering para analizar y clasificar las instancias:

- Extracción automática de características relevantes
- K-means para agrupar instancias similares
- Selección de instancias representativas
- Análisis de importancia de características

### 5. Optimización de Hiperparámetros (`hyperparameter_tuning.py`)

El sistema incluye un módulo completo para optimización de hiperparámetros:

- Búsqueda en cuadrícula y búsqueda aleatoria
- Análisis de sensibilidad por parámetro
- Visualización 3D de espacios de hiperparámetros
- Paralelización para aceleración del proceso

## Requisitos del Sistema

- Python 3.8+
- TensorFlow 2.5+
- PuLP para MILP
- Scikit-learn para análisis de clustering
- Pandas y NumPy para manipulación de datos
- Matplotlib y Seaborn para visualizaciones

## Uso del Sistema

### 1. Preparación de Datos

Coloque sus instancias en formato JSON en el directorio `data/` con la siguiente estructura:

```json
{
  "energy_prices": [{"time": float, "price": float}, ...],
  "arrivals": [
    {
      "id": int,
      "arrival_time": float,
      "departure_time": float,
      "required_energy": float
    },
    ...
  ],
  "parking_config": {
    "n_spots": int,
    "chargers": [{"charger_id": int, "power": float}, ...],
    "transformer_limit": float
  }
}
```

## Ejecución del Algoritmo Heurístico (heuristic.py)

El algoritmo heurístico (`heuristic.py`) puede ejecutarse de forma independiente para probar su rendimiento en instancias específicas o para procesar múltiples instancias en lote.

### Opciones de Ejecución

```bash
# Para obtener ayuda sobre los comandos disponibles
python heuristic.py

# Para una sola instancia
python heuristic.py data/test_system_1.json [archivo_salida]

# Para procesar todos los archivos en un directorio
python heuristic.py --all [directorio_datos] [directorio_salida]
```

#### Procesamiento de Una Instancia

Cuando se ejecuta para una sola instancia, el script admite dos parámetros:

1. **archivo_json** (obligatorio): Ruta al archivo JSON con los datos del sistema
   - Ejemplo: `python heuristic.py data/test_system_1.json`

2. **archivo_salida** (opcional): Ruta donde guardar los resultados
   - Si no se especifica, se guarda en `heuristic_results/[nombre_instancia]_resultado.json`
   - Ejemplo: `python heuristic.py data/test_system_1.json mi_solucion.json`

**Proceso y salidas:**
- Carga automáticamente la configuración de hiperparámetros óptima basada en clustering
- Ejecuta la heurística constructiva con mejoras adaptativas
- Muestra estadísticas de la solución (costo, energía, satisfacción)
- Genera tres gráficos:
  - `[nombre]_carga.png`: Perfil de carga de los vehículos
  - `[nombre]_parqueo.png`: Asignación de plazas de estacionamiento
  - `[nombre]_evolucion_costo.png`: Evolución del costo durante la optimización
- Guarda el resultado en formato JSON con:
  - Schedule detallado por vehículo
  - Estadísticas de la solución
  - Detalles de vehículos no satisfechos
  - Tiempo de ejecución

#### Procesamiento por Lotes

Con la opción `--all`, el script admite dos parámetros adicionales:

1. **directorio_datos** (opcional): Directorio que contiene los archivos JSON
   - Valor predeterminado: `./data`
   - Ejemplo: `python heuristic.py --all mis_instancias/`

2. **directorio_salida** (opcional): Directorio donde guardar los resultados
   - Valor predeterminado: `./resultados`
   - Ejemplo: `python heuristic.py --all data/ mis_resultados/`

**Proceso y salidas:**
- Procesa todos los archivos JSON encontrados en el directorio
- Genera los mismos resultados individuales que el modo de instancia única
- Crea un archivo CSV con resultados comparativos (`resultados_comparativos.csv`)
- Muestra una tabla resumen en consola con métricas clave

### Características del algoritmo heurístico

El algoritmo implementa:
- Construcción inicial con priorización basada en urgencia de carga
- Continuidad prioritaria para mantener sesiones de carga ininterrumpidas
- Fase de mejora con múltiples estrategias de perturbación:
  - Perturbación por intervalos: modifica asignaciones en períodos específicos
  - Perturbación por vehículos: remueve y reasigna vehículos seleccionados
  - Perturbación híbrida: combina ambos enfoques
  - Perturbación de sesiones: enfocada en sesiones de carga continuas
- Consolidación que rellena "huecos" en las sesiones de carga
- Selección automática de hiperparámetros basada en clustering

## Ejecución del Programa Principal (model131.py)

El archivo principal `model131.py` integra todos los componentes del sistema y ofrece diferentes modos de operación. Para ejecutarlo:

```bash
python model131.py
```

### Modos de Operación y Opciones

Al ejecutar el programa, se presentará un menú con los siguientes modos:

```
Sistema de optimización de carga de vehículos eléctricos
--------------------------------------------------------

Modos disponibles:
1. Entrenamiento generalizado (crear modelo para todos los sistemas)
2. Solución para una instancia específica (usando modelo generalizado)
3. Verificar progreso de entrenamiento y checkpoints
4. Procesar TODAS las instancias automáticamente (RL y MILP)
5. Visualizar solución MILP desde archivo
6. Generar tabla de resultados comparativa (solo JSON)

Seleccione modo (1-6): 
```

#### Modo 1: Entrenamiento Generalizado

Este modo entrena el modelo de aprendizaje por refuerzo utilizando todas las instancias disponibles.

**Opciones solicitadas:**
- **Número de episodios por sistema**: Recomendado entre 30-50. Mayor número = mejor calidad de soluciones pero más tiempo de entrenamiento.
- **Frecuencia de guardado de checkpoints**: Típicamente 5-10. Determina cada cuántos episodios se guarda el progreso.
- **Reanudar desde último checkpoint**: [s/n] Permite continuar un entrenamiento previo desde el punto donde se interrumpió.

**Ejemplo:**
```
Número de episodios por sistema (recomendado: 30-50): 30
Guardar checkpoint cada cuántos episodios (recomendado: 5): 5
¿Desea reanudar desde el último checkpoint? (s/n): n
```

#### Modo 2: Solución para una Instancia Específica

Resuelve una instancia específica utilizando el modelo generalizado y MILP.

**Opciones solicitadas:**
- **Nombre del archivo JSON**: Admite tres formatos:
  - Ruta completa: `./data/test_system_1.json`
  - Solo nombre: `test_system_1.json`
  - Solo número: `1` (buscará `test_system_1.json`)

**Ejemplo:**
```
Introduzca el nombre del archivo JSON (ej: test_system_1.json): 5
```

El sistema generará:
1. Solución RL (resultados_rl_instancia_X.json)
2. Solución MILP refinada (resultados_milp_instancia_X.json)

#### Modo 3: Verificar Progreso de Entrenamiento

Muestra información sobre los checkpoints disponibles y el progreso del entrenamiento.

**Opciones adicionales:**
- **Visualizar progreso gráficamente**: [s/n] Genera gráficas de progreso de entrenamiento.
- **Eliminar checkpoints antiguos**: [s/n] Opción para liberar espacio manteniendo solo los checkpoints más recientes.

**Ejemplo:**
```
¿Desea visualizar el progreso gráficamente? (s/n): s
¿Desea eliminar checkpoints antiguos para liberar espacio? (s/n): n
```

#### Modo 4: Procesar TODAS las Instancias Automáticamente

Procesa en lote todas las instancias disponibles en el directorio de datos.

**Sin opciones adicionales** - El sistema:
- Verifica cuáles instancias ya están procesadas
- Aplica límites de tiempo adaptativos según el tamaño de cada instancia
- Guarda resultados en el directorio `results/`

#### Modo 5: Visualizar Solución MILP desde Archivo

Genera visualizaciones a partir de una solución previamente calculada.

**Opciones solicitadas:**
- **Número de instancia**: Identificador numérico de la instancia a visualizar.

**Ejemplo:**
```
Ingrese el número de instancia (ej: 10): 3
```

Genera dos gráficos:
- Perfiles de carga de vehículos
- Asignación de plazas de parqueo a lo largo del tiempo

#### Modo 6: Análisis de Infactibilidad

Genera una tabla comparativa enfocada en detectar instancias infactibles.

**Sin opciones adicionales** - El sistema analiza todos los resultados disponibles y genera:
- Tabla CSV/Excel con análisis de infactibilidad
- Estadísticas sobre vehículos con energía cero
- Métricas de satisfacción de demanda


### Ejemplo de Flujo de Trabajo Típico

1. **Entrenamiento inicial** (Modo 1):
   ```bash
   python model131.py
   # Seleccionar modo 1
   # Especificar 30 episodios por sistema
   # Guardar checkpoint cada 5 episodios
   ```

2. **Procesar todas las instancias** (Modo 4):
   ```bash
   python model131.py
   # Seleccionar modo 4
   ```

3. **Analizar resultados** (Modo 6):
   ```bash
   python model131.py
   # Seleccionar modo 6
   ```

4. **Visualizar solución específica** (Modo 5):
   ```bash
   python model131.py
   # Seleccionar modo 5
   # Ingresar número de instancia (ej: 10)
   ```

### Requerimientos de Hardware

El sistema está optimizado para ejecutarse en equipos con las siguientes características:

- CPU multi-núcleo (4+ núcleos recomendados)
- 8GB+ RAM (16GB+ recomendado para instancias grandes)
- GPU opcional pero recomendada para entrenamiento acelerado

Para instancias muy grandes (500+ vehículos), se recomienda utilizar el procesamiento por lotes (implementado automáticamente).

### 2. Usando el Algoritmo Heurístico Directamente

Para comparar los resultados entre diferentes enfoques, puede ejecutar primero la heurística:

```bash
# Procesar una instancia específica
python heuristic.py data/test_system_1.json

# Procesar todas las instancias para comparación
python heuristic.py --all data/ heuristic_results/
```

Para problemas complejos con muchos vehículos (>200), la heurística puede ofrecer soluciones rápidas y de buena calidad.

### 3. Entrenamiento del Modelo RL

```bash
python model131.py
# Seleccione modo 1 para entrenamiento generalizado
```

### 4. Optimización de Hiperparámetros

```bash
python hyperparameter_tuning.py
```

### 5. Solución para una Instancia Específica

```bash
python model131.py
# Seleccione modo 2 para resolver una instancia específica
```

### 6. Procesamiento por Lotes

```bash
python model131.py
# Seleccione modo 4 para procesar todas las instancias
```

### 7. Generación de Tablas Comparativas

```bash
python results.py
```

### 8. Comparación entre Heurística y RL+MILP

Para comparar soluciones entre la heurística y el enfoque combinado de RL+MILP:

```bash
# Generar visualización comparativa
python plot_comparativo.py heuristic_results/test_system_1_resultado.json results/resultados_milp_instancia_1.json
```

Esto generará un gráfico de comparación mostrando los perfiles de carga y diferencias en la utilización de slots.

## Ejecución de Herramientas de Análisis y Visualización

### Ejecutar plots_milp.py

El archivo `plots_milp.py` implementa un completo conjunto de herramientas para analizar los resultados y evaluar el rendimiento de la infraestructura de carga. Para ejecutarlo:

```bash
python plots_milp.py
```

Al iniciar el programa, se presentará un menú interactivo con las siguientes opciones:

```
=== MENÚ DE ANÁLISIS ===
1. Comparación de costos (RL vs MILP)
2. Análisis de congestión del parqueadero
3. Análisis de vehículos rechazados
4. Análisis de uso del transformador
5. Simulación de capacidad óptima
6. Salir
```

Para cada opción, se puede elegir analizar:
- Todas las instancias
- Una instancia específica
- Las 3 instancias más congestionadas

#### 1. Comparación de costos (RL vs MILP)

Genera gráficas comparativas entre las soluciones de RL y MILP, mostrando:
- Costos totales
- Porcentaje de mejora
- Energía entregada
- Porcentaje de EVs atendidos
- Tiempos de ejecución

Los resultados se guardan como `plots/cost_comparison.png` y `plots/time_comparison.png`.

#### 2. Análisis de congestión del parqueadero

Visualiza cómo varía la demanda de plazas a lo largo del tiempo:
- Vehículos presentes vs. capacidad disponible
- Porcentaje de utilización
- Vehículos sin espacio en cada intervalo
- Identificación de períodos con exceso de demanda

Los resultados se guardan como `plots/parking_congestion.png`.

#### 3. Análisis de vehículos rechazados

Analiza los vehículos que no pueden ser atendidos por falta de espacio:
- Distribución de rechazos por hora del día
- Características de los vehículos rechazados
- Congestión y rechazos a lo largo del tiempo
- Cálculo del costo de oportunidad

Los resultados se guardan como `plots/rejected_vehicles_analysis.png`.

#### 4. Análisis de uso del transformador

Examina la utilización del transformador a lo largo del tiempo:
- Comparación entre soluciones RL y MILP
- Momentos de mayor demanda energética
- Porcentaje de tiempo cerca del límite
- Eficiencia en el uso de la potencia disponible

Los resultados se guardan como `plots/transformer_usage.png`.

#### 5. Simulación de capacidad óptima

Simula diferentes configuraciones de capacidad para encontrar el punto óptimo:
- Tasa de rechazo vs. capacidad
- ROI y revenue por capacidad
- Identificación de la capacidad óptima económica
- Recomendaciones específicas de dimensionamiento

Los resultados se guardan como `plots/capacity_simulation.png`.

### Ejecutar plots_heuristics.py

El archivo `plots_heuristics.py` proporciona herramientas para analizar el impacto de los hiperparámetros en el rendimiento del algoritmo heurístico. Para ejecutarlo:

```bash
python plots_heuristics.py
```

Este programa analiza automáticamente los resultados de optimización de hiperparámetros guardados en el directorio `hyperparameter_results/` y genera las siguientes visualizaciones:

#### Análisis de parámetros individuales
Para cada parámetro, genera gráficos mostrando su efecto en el costo final:
- Gráficos de dispersión o boxplots según el número de valores únicos
- Líneas de tendencia para identificar patrones
- Guardados como `[param]_vs_cost.png`

#### Análisis de interacciones entre parámetros
Genera visualizaciones 3D y mapas de calor para pares de parámetros:
- Superficies 3D mostrando la interacción
- Mapas de calor con valores de costo
- Gráficos de contorno para identificar regiones óptimas
- Guardados como `[param1]_[param2]_3d.png`

#### Análisis de correlación
Genera matrices de correlación entre parámetros y métricas:
- Identificación de los parámetros más influyentes
- Correlaciones entre parámetros y resultados
- Guardado como `correlation_matrix.png`

#### Dashboards de resultados
Genera un dashboard completo con los hallazgos más importantes:
- Top 5 parámetros más influyentes
- Matriz de correlación reducida
- Información de la mejor configuración
- Guardado como `dashboard.png`

#### Combinaciones óptimas de parámetros
Visualiza las 4 mejores combinaciones de parámetros:
- Mapas de calor con marcadores para valores óptimos
- Guardado como `top_combinations_heatmap.png`

Los resultados se organizan en directorios por sistema:
```
hyperparameter_analysis/
├── best_parameters_summary.txt
├── test_system_2/
│   ├── dashboard.png
│   ├── correlation_matrix.png
│   ├── [param]_vs_cost.png
│   ├── [param1]_[param2]_3d.png
│   └── top_combinations_heatmap.png
├── test_system_9/
│   └── ...
└── test_system_10/
    └── ...
```

El archivo `best_parameters_summary.txt` contiene un resumen detallado de los mejores hiperparámetros encontrados para cada sistema.

## Optimización de Hiperparámetros (hyperparameter_tuning.py)

El sistema incluye un completo módulo de optimización de hiperparámetros que permite encontrar la configuración óptima para el algoritmo heurístico, adaptándose a diferentes tipos de instancias de forma automática.

### Opciones de Ejecución

```bash
# Para ver las opciones disponibles
python hyperparameter_tuning.py --help

# Modo interactivo
python hyperparameter_tuning.py interactive

# Búsqueda en cuadrícula con parámetros específicos
python hyperparameter_tuning.py search --method grid --data_dir ./data --output_dir hyperparameter_results

# Búsqueda aleatoria con particionamiento para ejecución paralela
python hyperparameter_tuning.py search --method random --n_samples 30 --n_jobs 4 --actual_job 1

# Análisis de resultados existentes sin ejecutar búsqueda
python hyperparameter_tuning.py search --analyze --method grid

# Ejecutar heurística con los mejores parámetros encontrados
python hyperparameter_tuning.py run data/test_system_1.json hyperparameter_results/grid_best_parameters.csv
```

### Modos de Operación

El script soporta tres modos principales:

#### 1. Modo Interactivo (`interactive`)

Guía al usuario a través del proceso de selección de:
- Archivos de configuración a utilizar
- Método de búsqueda (cuadrícula o aleatorio)
- Parámetros específicos a explorar
- Número de pruebas por configuración
- Directorio de salida

Este modo es ideal para usuarios que están comenzando a explorar el espacio de hiperparámetros.

#### 2. Modo de Búsqueda (`search`)

Permite ejecutar procesos de búsqueda de hiperparámetros mediante línea de comandos con opciones detalladas:

**Opciones disponibles:**
- `--config`: Ruta a un archivo JSON específico
- `--data_dir`: Directorio con archivos JSON (default: ./cluster_results/representative_files)
- `--output_dir`: Directorio para resultados (default: hyperparameter_results)
- `--method`: Método de búsqueda (grid/random)
- `--n_jobs`: Número total de particiones para procesar en paralelo
- `--actual_job`: Índice de la partición actual (1 a n_jobs)
- `--n_trials`: Número de pruebas por configuración
- `--n_samples`: Número de muestras para búsqueda aleatoria
- `--analyze`: Solo analizar resultados existentes sin ejecutar búsqueda

La opción de particionamiento permite ejecutar búsquedas en múltiples máquinas o núcleos para acelerar el proceso.

#### 3. Modo de Ejecución con Parámetros Óptimos (`run`)

Ejecuta la heurística con los mejores parámetros previamente encontrados:

**Parámetros requeridos:**
- `json_path`: Archivo JSON con la configuración del sistema
- `params_path`: Archivo CSV con los mejores parámetros
- `--output_path`: Ruta para guardar resultados (opcional)

Este modo es útil para aplicar los mejores parámetros encontrados a nuevas instancias.

### Métodos de Búsqueda

#### Búsqueda en Cuadrícula (Grid Search)

Explora sistemáticamente todas las combinaciones de hiperparámetros especificadas:

```python
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
```

#### Búsqueda Aleatoria (Random Search)

Explora muestras aleatorias dentro de rangos especificados:

```python
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
```

### Análisis de Resultados

El sistema genera automáticamente análisis detallados de los resultados:

- **Resumen de mejores parámetros**: Archivo de texto con los mejores parámetros encontrados para cada configuración
- **Gráficos de efecto de parámetros**: Visualizaciones que muestran el impacto de cada parámetro en el costo final
- **Análisis de correlación**: Muestra qué parámetros tienen mayor impacto en las métricas

Los resultados se organizan en el directorio especificado:
```
hyperparameter_results/
├── grid_search_results_combined.csv     # Resultados combinados de todas las particiones
├── grid_best_parameters.csv             # Mejores parámetros en formato CSV
├── best_parameters_summary.txt          # Resumen de mejores parámetros en formato texto
├── instancia_test_system_X_grid.txt     # Resultados detallados por instancia
└── plots/                               # Directorio con gráficos de análisis
```

### Características de Resiliencia y Paralelización

El sistema incluye características avanzadas para manejar búsquedas a gran escala:

- **Reanudación automática**: Detecta y reanuda búsquedas interrumpidas
- **Particionamiento de trabajo**: Divide el espacio de búsqueda para ejecución en múltiples máquinas
- **Guardado incremental**: Guarda resultados periódicamente para evitar pérdidas por interrupciones
- **Respaldo automático**: Crea copias de seguridad de archivos corruptos o dañados

### Ejemplo de Uso Completo

```bash
# 1. Ejecutar búsqueda en cuadrícula en 4 particiones
python hyperparameter_tuning.py search --method grid --n_jobs 4 --actual_job 1 --data_dir ./data
python hyperparameter_tuning.py search --method grid --n_jobs 4 --actual_job 2 --data_dir ./data
python hyperparameter_tuning.py search --method grid --n_jobs 4 --actual_job 3 --data_dir ./data
python hyperparameter_tuning.py search --method grid --n_jobs 4 --actual_job 4 --data_dir ./data

# 2. Analizar resultados combinados
python hyperparameter_tuning.py search --analyze --method grid

# 3. Ejecutar con los mejores parámetros
python hyperparameter_tuning.py run data/test_system_5.json hyperparameter_results/grid_best_parameters.csv
```

Este flujo permite explorar eficientemente el espacio de hiperparámetros y aplicar los mejores resultados a nuevas instancias.

## Contribuciones

El proyecto integra varios enfoques complementarios para obtener soluciones optimizadas:

1. La heurística proporciona soluciones rápidas de buena calidad
2. El RL genera políticas que se adaptan a diferentes tipos de instancias
3. El MILP refina las soluciones para obtener mejoras adicionales
4. El análisis de clustering permite seleccionar parámetros óptimos automáticamente

## Resultados Clave

- **Reducción de costos**: Hasta un 15-20% comparado con políticas simples
- **Satisfacción de demanda**: >95% de la energía requerida en la mayoría de instancias
- **Tiempo de solución**: Segundos para la heurística, minutos para RL+MILP
- **Escalabilidad**: Probado con instancias de hasta 500+ vehículos

## Autor

Tomás Acosta Bernal

## Referencias

Li, H., Han, B., Li, G., Wang, K., Xu, J., & Khan, M. W. (2024). Decentralized collaborative optimal scheduling for EV charging stations based on multi‐agent reinforcement learning. *IET Generation, Transmission & Distribution*, *18*(6), 1172–1183. https://doi.org/10.1049/gtd2.13047

Rizopoulos, D., & Esztergár-Kiss, D. (2023). Heuristic time-dependent personal scheduling problem with electric vehicles. *Transportation (Dordrecht)*, *50*(5), 2009–2048. https://doi.org/10.1007/s11116-022-10300-0

Li, J., Xu, Y., Zhang, J., Gao, C., & Sun, H. (2025). Distributed EV scheduling in distribution networks with reserve market participation under ambiguous probability distribution. *Applied Energy*, *383*, 125269-. https://doi.org/10.1016/j.apenergy.2024.125269

Sone, S. P., Lehtomaki, J. J., Khan, Z., Umebayashi, K., & Kim, K. S. (2024). Robust EV Scheduling in Charging Stations Under Uncertain Demands and Deadlines. *IEEE Transactions on Intelligent Transportation Systems*, *25*(12), 21484–21499. https://doi.org/10.1109/TITS.2024.3466514

Ran, Y., Liao, H., Liang, H., Lu, L., & Zhong, J. (2024). Optimal Scheduling Strategies for EV Charging and Discharging in a Coupled Power–Transportation Network with V2G Scheduling and Dynamic Pricing. Energies (Basel), 17(23), 6167-. https://doi.org/10.3390/en17236167

Mao, T., Zhang, X., & Zhou, B. (2019). Intelligent Energy Management Algorithms for EV-charging Scheduling with Consideration of Multiple EV Charging Modes. Energies (Basel), 12(2), 265-. https://doi.org/10.3390/en12020265

Shohan, M. J. A., Islam, M. M., Owais, S., & Faruque, M. O. (2024). Optimal Energy Management of EVs at Workplaces and Residential Buildings Using Heuristic Graph-Search Algorithm. Energies (Basel), 17(21), 5278-. https://doi.org/10.3390/en17215278

---

Este proyecto fue desarrollado como parte de una investigación sobre optimización de infraestructura de carga para vehículos eléctricos.
