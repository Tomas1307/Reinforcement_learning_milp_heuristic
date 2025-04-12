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

### 2. Entrenamiento del Modelo RL

```bash
python model131.py
# Seleccione modo 1 para entrenamiento generalizado
```

### 3. Optimización de Hiperparámetros

```bash
python hyperparameter_tuning.py
```

### 4. Solución para una Instancia Específica

```bash
python model131.py
# Seleccione modo 2 para resolver una instancia específica
```

### 5. Procesamiento por Lotes

```bash
python model131.py
# Seleccione modo 4 para procesar todas las instancias
```

### 6. Generación de Tablas Comparativas

```bash
python results.py
```

## Análisis de Rendimiento

Para analizar el rendimiento del sistema, se incluyen varias herramientas:

- **Análisis de clustering**: Permite entender las similitudes entre instancias
- **Visualización de soluciones**: Muestra perfiles de carga y asignación de slots
- **Análisis de congestión**: Identifica períodos de demanda excesiva
- **Simulación de capacidad óptima**: Ayuda a dimensionar instalaciones

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

---

Este proyecto fue desarrollado como parte de una investigación sobre optimización de infraestructura de carga para vehículos eléctricos.
