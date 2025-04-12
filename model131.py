import json
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import pulp
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus
import os
import glob
import sys
import time

#########################################
# Funciones de lectura y datos
#########################################
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
        "test_number": data.get("test_number", 0)
    }

def load_all_test_systems(data_dir="./data"):
    """
    Carga todos los sistemas de prueba disponibles en el directorio de datos.
    
    Returns:
        dict: Mapeo de número de sistema a configuración
    """
    systems = {}
    json_files = glob.glob(os.path.join(data_dir, "test_system_*.json"))
    
    for json_file in json_files:
        try:
            config = load_data(json_file)
            test_number = config["test_number"]
            systems[test_number] = config
            print(f"Sistema {test_number} cargado: {len(config['arrivals'])} vehículos, {config['n_spots']} plazas, {len(config['chargers'])} cargadores")
        except Exception as e:
            print(f"Error al cargar {json_file}: {e}")
        
    return systems

#########################################
# Clase del entorno para RL con protecciones
#########################################
class EVChargingEnv:
    """
    Entorno de simulación para la carga de vehículos eléctricos con RL.
    Diseñado para ser escalable a cualquier tamaño de instancia.
    """
    def __init__(self, config):
        """Inicializa el entorno con la configuración del sistema."""
        self.times = config["times"]
        self.prices = config["prices"]
        self.arrivals = config["arrivals"]
        self.chargers = config["chargers"]
        self.station_limit = config["station_limit"]
        self.dt = config["dt"]
        self.n_spots = config["n_spots"]
        self.test_number = config.get("test_number", 0)
        
        # Detectar características adicionales disponibles
        self.has_brand_info = any('brand' in ev for ev in self.arrivals)
        self.has_battery_info = any('battery_capacity' in ev for ev in self.arrivals)
        
        # Normalización de precios (importante para generalización)
        self.min_price = min(self.prices)
        self.max_price = max(self.prices)
        self.normalized_prices = [(p - self.min_price) / (self.max_price - self.min_price + 1e-6) 
                                 for p in self.prices]
        
        # Cálculo de estadísticas del sistema (para generalización)
        self.max_charger_power = max(c["power"] for c in self.chargers)
        self.total_charging_capacity = sum(c["power"] for c in self.chargers)
        self.avg_required_energy = np.mean([arr["required_energy"] for arr in self.arrivals])
        self.max_required_energy = max(arr["required_energy"] for arr in self.arrivals)
        self.avg_stay_duration = np.mean([(arr["departure_time"] - arr["arrival_time"]) 
                                         for arr in self.arrivals])
        
        # Mapeos para facilitar el acceso
        self.ev_ids = [arr["id"] for arr in self.arrivals]
        self.arrival_time = {arr["id"]: arr["arrival_time"] for arr in self.arrivals}
        self.departure_time = {arr["id"]: arr["departure_time"] for arr in self.arrivals}
        self.required_energy = {arr["id"]: arr["required_energy"] for arr in self.arrivals}
        
        # Información adicional si está disponible
        if self.has_battery_info:
            self.battery_capacity = {arr["id"]: arr.get("battery_capacity", 50) for arr in self.arrivals}
        
        if self.has_brand_info:
            self.brands = {arr["id"]: arr.get("brand", "Unknown") for arr in self.arrivals}
            # Crear un mapeo numérico para las marcas
            unique_brands = list(set(self.brands.values()))
            self.brand_to_id = {brand: i/max(1, len(unique_brands)) for i, brand in enumerate(unique_brands)}
        
        self.charger_ids = [c["charger_id"] for c in self.chargers]
        self.max_charger_power_dict = {c["charger_id"]: c["power"] for c in self.chargers}
        
        # Estado del entorno
        self.current_ev_idx = 0
        self.current_time_idx = 0
        self.spot_assignments = {}  # Asignación de spot por EV
        self.charging_schedule = []  # [(ev_id, time_idx, charger_id, spot, power)]
        self.occupied_spots = {t: set() for t in range(len(self.times))}
        self.occupied_chargers = {t: set() for t in range(len(self.times))}
        self.power_used = {t: 0 for t in range(len(self.times))}
        self.energy_delivered = {ev_id: 0 for ev_id in self.ev_ids}
        
    def reset(self):
        """Reinicia el entorno al estado inicial."""
        self.current_ev_idx = 0
        self.current_time_idx = 0
        self.spot_assignments = {}
        self.charging_schedule = []
        self.occupied_spots = {t: set() for t in range(len(self.times))}
        self.occupied_chargers = {t: set() for t in range(len(self.times))}
        self.power_used = {t: 0 for t in range(len(self.times))}
        self.energy_delivered = {ev_id: 0 for ev_id in self.ev_ids}
        
        return self._get_state()
    
    def _get_state(self):
        """
        Obtiene el estado actual del entorno para el RL.
        """
        if self.current_ev_idx >= len(self.arrivals):
            # Si ya procesamos todos los EVs, retornamos un estado terminal
            return None
        
        ev = self.arrivals[self.current_ev_idx]
        ev_id = ev["id"]
        
        # Calcular los intervalos en los que el EV está presente
        ev_time_indices = [i for i, t in enumerate(self.times) 
                          if self.arrival_time[ev_id] <= t < self.departure_time[ev_id]]
        
        if not ev_time_indices:
            # Si el EV no coincide con ningún intervalo, pasamos al siguiente
            self.current_ev_idx += 1
            return self._get_state()
        
        # Cálculo de características normalizadas para mejor generalización
        stay_duration = self.departure_time[ev_id] - self.arrival_time[ev_id]
        energy_requirement_normalized = self.required_energy[ev_id] / self.max_required_energy
        stay_duration_normalized = stay_duration / max(self.times)  # Normalizado al tiempo máximo
        energy_delivered_ratio = self.energy_delivered[ev_id] / self.required_energy[ev_id] if self.required_energy[ev_id] > 0 else 0
        
        # Características del EV actual (normalizadas)
        ev_features = [
            self.arrival_time[ev_id] / max(self.times),  # Tiempo de llegada normalizado
            self.departure_time[ev_id] / max(self.times),  # Tiempo de salida normalizado
            energy_requirement_normalized,  # Energía requerida normalizada
            len(ev_time_indices) / len(self.times),  # Proporción de tiempo disponible
            energy_delivered_ratio,  # Proporción de energía ya entregada
            min(ev_time_indices) / len(self.times),  # Primer intervalo normalizado
            max(ev_time_indices) / len(self.times)   # Último intervalo normalizado
        ]
        
        # Añadir información de batería si está disponible
        if self.has_battery_info:
            battery_capacity_normalized = self.battery_capacity[ev_id] / 100.0  # Normalizado a 100kWh
            ev_features.append(battery_capacity_normalized)
        
        # Añadir información de marca si está disponible
        if self.has_brand_info:
            brand_id = self.brand_to_id[self.brands[ev_id]]
            ev_features.append(brand_id)
        
        # Disponibilidad de spots para cada intervalo de tiempo
        available_spots_ratio = []
        for t in ev_time_indices:
            ratio = (self.n_spots - len(self.occupied_spots[t])) / self.n_spots
            available_spots_ratio.append(ratio)
        
        # Disponibilidad de cargadores para cada intervalo de tiempo
        available_chargers_ratio = []
        for t in ev_time_indices:
            ratio = (len(self.charger_ids) - len(self.occupied_chargers[t])) / len(self.charger_ids)
            available_chargers_ratio.append(ratio)
        
        # Precios de energía normalizados en los intervalos relevantes
        relevant_prices = [self.normalized_prices[t] for t in ev_time_indices]
        
        # Capacidad restante del transformador en cada intervalo (normalizada)
        transformer_capacity_ratio = [(self.station_limit - self.power_used[t]) / self.station_limit 
                                     for t in ev_time_indices]
        
        # Valores agregados para simplificar el estado
        avg_available_spots = np.mean(available_spots_ratio)
        avg_available_chargers = np.mean(available_chargers_ratio)
        min_price = np.min(relevant_prices)
        avg_price = np.mean(relevant_prices)
        min_transformer_capacity = np.min(transformer_capacity_ratio)
        avg_transformer_capacity = np.mean(transformer_capacity_ratio)
        
        # Calcular métricas adicionales útiles
        total_energy_needed = sum(self.required_energy[id] for id in self.ev_ids) - sum(self.energy_delivered.values())
        total_energy_capacity = self.total_charging_capacity * len(self.times) * self.dt
        system_demand_ratio = min(1.0, total_energy_needed / (total_energy_capacity + 1e-6))
        
        # Urgencia de carga
        time_remaining = (self.departure_time[ev_id] - self.times[min(ev_time_indices)]) / max(self.times)
        energy_needed = (self.required_energy[ev_id] - self.energy_delivered[ev_id]) / self.max_required_energy
        charging_urgency = energy_needed / (time_remaining + 1e-6)
        charging_urgency = min(1.0, charging_urgency)  # Normalizado entre 0 y 1
        
        # Estado completo más detallado para referencia
        full_state = {
            "ev_features": ev_features,
            "spot_availability": available_spots_ratio,
            "charger_availability": available_chargers_ratio,
            "relevant_prices": relevant_prices,
            "transformer_capacity": transformer_capacity_ratio,
            "time_indices": ev_time_indices,
            
            # Características agregadas para facilitar la generalización
            "avg_available_spots": avg_available_spots,
            "avg_available_chargers": avg_available_chargers,
            "min_price": min_price,
            "avg_price": avg_price,
            "min_transformer_capacity": min_transformer_capacity,
            "avg_transformer_capacity": avg_transformer_capacity,
            
            # Características del sistema para generalización
            "system_type": self.test_number,
            "n_spots_total": self.n_spots,
            "n_chargers_total": len(self.charger_ids),
            "transformer_limit": self.station_limit,
            "max_charger_power": self.max_charger_power,
            "total_charging_capacity": self.total_charging_capacity,
            
            # Características adicionales si están disponibles
            "system_demand_ratio": system_demand_ratio,
            "charging_urgency": charging_urgency,
            "time_remaining": time_remaining
        }
        
        # Añadir propiedades de batería si están disponibles
        if self.has_battery_info:
            full_state["battery_capacity"] = self.battery_capacity[ev_id] / 100.0
        
        return full_state
    
    def _get_possible_actions(self, state):
        """
        Determina las acciones posibles dado el estado actual.
        """
        if state is None:
            return []
        
        ev_id = self.arrivals[self.current_ev_idx]["id"]
        time_indices = state["time_indices"]
        
        # Determinar qué slots están disponibles durante todo el período
        consistent_spots = list(range(self.n_spots))
        for t in time_indices:
            consistent_spots = [s for s in consistent_spots if s not in self.occupied_spots[t]]
        
        actions = []
        
        # Siempre incluir la opción de saltar como última opción
        skip_action = {"skip": True}
        
        # Calcular la potencia requerida ideal (energía/tiempo disponible)
        energy_needed = self.required_energy[ev_id] - self.energy_delivered[ev_id]
        available_time = len(time_indices) * self.dt
        
        if available_time > 0:
            ideal_power = energy_needed / available_time
        else:
            return [skip_action]  # No hay tiempo disponible, solo podemos saltar
        
        # Para cada spot disponible, generamos acciones posibles
        for spot in consistent_spots:
            # Determinar niveles de potencia según las capacidades del sistema
            power_levels = []
            
            # Potencia ideal
            power_levels.append(ideal_power)
            
            # Niveles relativos a la ideal
            power_levels.append(ideal_power * 0.75)
            power_levels.append(ideal_power * 0.5)
            
            # Niveles según cargadores disponibles
            available_powers = sorted(set(self.max_charger_power_dict.values()))
            for power in available_powers:
                if power >= ideal_power * 1.5 and power <= self.max_charger_power:
                    power_levels.append(power)
            
            # Eliminar duplicados y ordenar
            power_levels = sorted(set(power_levels))
            
            for required_power in power_levels:
                # Eliminamos niveles de potencia que exceden la capacidad máxima
                if required_power > self.max_charger_power:
                    continue
                    
                # Verificamos si hay suficiente capacidad de cargador y transformador
                feasible_charging = True
                charger_assignments = {}
                
                for t in time_indices:
                    # Buscamos un cargador disponible con suficiente potencia
                    available_chargers = [c for c in self.charger_ids 
                                        if c not in self.occupied_chargers[t] and 
                                        self.max_charger_power_dict[c] >= required_power and
                                        self.power_used[t] + required_power <= self.station_limit]
                    
                    if not available_chargers:
                        feasible_charging = False
                        break
                    
                    # Asignamos el cargador con la potencia más adecuada
                    best_charger = min(available_chargers, 
                                    key=lambda c: self.max_charger_power_dict[c])
                    charger_assignments[t] = (best_charger, required_power)
                
                if feasible_charging:
                    actions.append({
                        "skip": False,
                        "spot": spot,
                        "charging_profile": charger_assignments,
                        "power_level": required_power
                    })
        
        # Si no hay acciones factibles de carga, solo agregamos la opción de saltar
        if not actions:
            actions.append(skip_action)
        else:
            # Añadimos la opción de saltar al final
            actions.append(skip_action)
        
        return actions
    
    def step(self, action_idx):
        """
        Ejecuta una acción y avanza el entorno al siguiente estado.
        Retorna el nuevo estado, la recompensa, y si es un estado terminal.
        """
        state = self._get_state()
        if state is None:
            return None, 0, True
        
        actions = self._get_possible_actions(state)
        
        # Protección contra índices inválidos
        if action_idx < 0 or action_idx >= len(actions):
            # Acción inválida, usamos la última (skip)
            action_idx = len(actions) - 1
        
        action = actions[action_idx]
        ev_id = self.arrivals[self.current_ev_idx]["id"]
        
        if action["skip"]:
            # El EV no recibe carga, penalizamos en proporción a su demanda no satisfecha
            energy_deficit = self.required_energy[ev_id] - self.energy_delivered[ev_id]
            # Penalización proporcional a la energía no entregada y normalizada por la energía máxima
            normalized_deficit = energy_deficit / self.max_required_energy
            reward = -20 * normalized_deficit
            self.current_ev_idx += 1
            return self._get_state(), reward, self.current_ev_idx >= len(self.arrivals)
        
        # Ejecutamos la acción: asignamos spot y cargadores
        spot = action["spot"]
        charging_profile = action["charging_profile"]
        
        total_cost = 0
        total_energy = 0
        
        for t, (charger, power) in charging_profile.items():
            # Registrar la asignación
            self.charging_schedule.append((ev_id, t, charger, spot, power))
            
            # Actualizar el estado del entorno
            self.occupied_spots[t].add(spot)
            self.occupied_chargers[t].add(charger)
            self.power_used[t] += power
            
            # Calcular la energía entregada y el costo
            energy = power * self.dt
            self.energy_delivered[ev_id] += energy
            total_energy += energy
            total_cost += energy * self.prices[t]
        
        # Calculamos la satisfacción de energía normalizada
        energy_satisfaction_ratio = min(1.0, self.energy_delivered[ev_id] / self.required_energy[ev_id])
        
        # Recompensa basada en la energía entregada menos el costo normalizado
        normalized_cost = total_cost / (self.max_price * total_energy) if total_energy > 0 else 0
        reward = 50 * energy_satisfaction_ratio - 30 * normalized_cost
        
        # Bono por eficiencia energética y uso de precios bajos
        if total_energy > 0:
            avg_price_paid = total_cost / total_energy
            price_efficiency = 1 - (avg_price_paid - self.min_price) / (self.max_price - self.min_price + 1e-6)
            reward += 10 * price_efficiency
        
        # Pasamos al siguiente EV
        self.current_ev_idx += 1
        
        return self._get_state(), reward, self.current_ev_idx >= len(self.arrivals)
    
    def get_schedule(self):
        """Retorna el schedule de carga completo generado por el agente RL."""
        return self.charging_schedule

#########################################
# Agente DQN generalizado con protecciones
#########################################
class GeneralizedDQNAgent:
    """
    Agente de Deep Q-Learning para el problema de scheduling de carga de EVs.
    Con protecciones para evitar errores de índice y memoria.
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        
        # Memoria por sistema para transferencia
        self.system_memories = defaultdict(lambda: deque(maxlen=2000))
        
    def _build_model(self):
        """
        Construye la red neuronal para aproximar la función Q
        """
        model = keras.Sequential([
            keras.layers.Dense(64, input_dim=self.state_size, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done, system_type=0):
        """
        Almacena una experiencia en memoria.
        Con verificación de índice válido.
        """
        if action >= 0 and action < self.action_size:
            experience = (state, action, reward, next_state, done)
            self.memory.append(experience)
            self.system_memories[system_type].append(experience)
    
    def act(self, state, possible_actions):
        """
        Selecciona una acción según la política epsilon-greedy.
        Con verificación de índices.
        """
        if len(possible_actions) == 0:
            return -1  # No hay acciones posibles
        
        if np.random.rand() <= self.epsilon:
            return np.random.choice(len(possible_actions))
        
        state_vector = self._process_state(state)
        act_values = self.model.predict(np.array([state_vector]), verbose=0)[0]
        
        # Filtrar acciones posibles y válidas
        filtered_actions = [(i, act_values[i]) for i in range(min(len(act_values), len(possible_actions)))]
        
        if not filtered_actions:
            return np.random.choice(len(possible_actions))
        
        return max(filtered_actions, key=lambda x: x[1])[0]
    
    def _process_state(self, state):
        """
        Procesa el estado en un vector para la red.
        Con normalización y protección de dimensiones.
        """
        if state is None:
            return np.zeros(self.state_size)
        
        # Vector base con las características principales
        base_vector = []
        
        # Características básicas del EV
        ev_features = state["ev_features"]
        base_vector.extend(ev_features[:7])  # 7 características principales del EV
        
        # Características del sistema
        system_features = [
            state.get("avg_available_spots", 0.5),
            state.get("avg_available_chargers", 0.5),
            state.get("min_price", 0.5),
            state.get("avg_price", 0.5),
            state.get("min_transformer_capacity", 0.5),
            state.get("avg_transformer_capacity", 0.5),
            state.get("system_type", 0) / 20.0,
            state.get("n_spots_total", 10) / 100.0,
            state.get("n_chargers_total", 5) / 50.0,
            state.get("transformer_limit", 50) / 200.0,
            state.get("max_charger_power", 10) / 100.0
        ]
        base_vector.extend(system_features)
        
        # Características adicionales si están disponibles
        if "battery_capacity" in state:
            base_vector.append(state["battery_capacity"])
        
        if "charging_urgency" in state:
            base_vector.append(state["charging_urgency"])
        
        if "system_demand_ratio" in state:
            base_vector.append(state["system_demand_ratio"])
        
        if "time_remaining" in state:
            base_vector.append(state["time_remaining"])
        
        # Asegurar la dimensión correcta
        all_features = np.array(base_vector)
        if len(all_features) < self.state_size:
            # Rellenar con ceros
            padding = np.zeros(self.state_size - len(all_features))
            all_features = np.concatenate([all_features, padding])
        elif len(all_features) > self.state_size:
            # Truncar
            all_features = all_features[:self.state_size]
        
        return all_features
    
    def replay(self, batch_size, system_specific=False, system_type=0):
        """
        Entrena la red con experiencias pasadas.
        Con verificación de índices y protección contra errores.
        """
        try:
            if system_specific and len(self.system_memories[system_type]) >= batch_size:
                memory = self.system_memories[system_type]
            elif len(self.memory) >= batch_size:
                memory = self.memory
            else:
                return
            
            minibatch = random.sample(memory, batch_size)
            states = []
            targets = []
            
            for state, action, reward, next_state, done in minibatch:
                # Verificar índice válido
                if action >= self.action_size:
                    continue
                    
                state_vector = self._process_state(state)
                target = reward
                
                if not done:
                    next_state_vector = self._process_state(next_state)
                    target = reward + self.gamma * np.amax(
                        self.model.predict(np.array([next_state_vector]), verbose=0)[0]
                    )
                
                target_f = self.model.predict(np.array([state_vector]), verbose=0)
                target_f[0][action] = target
                
                states.append(state_vector)
                targets.append(target_f[0])
            
            if states:
                self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0, batch_size=32)
                
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
        except Exception as e:
            print(f"Error en replay: {e}")
            # No fallar, simplemente no actualizar en este paso
    
    def load(self, name):
        """Carga los pesos del modelo desde un archivo."""
        try:
            self.model.load_weights(name)
            return True
        except Exception as e:
            print(f"Error al cargar modelo desde {name}: {e}")
            return False
    
    def save(self, name):
        """Guarda los pesos del modelo en un archivo."""
        try:
            self.model.save_weights(name)
            return True
        except Exception as e:
            print(f"Error al guardar modelo en {name}: {e}")
            return False

#########################################
# Funciones de entrenamiento generalizado
#########################################
def train_generalized_agent(systems, episodes_per_system=30, batch_size=64, model_path="ev_scheduler_generalized_model.h5", 
                       checkpoint_dir="./checkpoints", checkpoint_frequency=5, resume_from_checkpoint=True,
                       patience=5):
    """
    Entrena un agente RL generalizado con múltiples sistemas.
    Guarda el modelo entrenado para uso futuro y checkpoints periódicos.
    Incluye early stopping para detener el entrenamiento cuando no hay mejoras.
    
    Args:
        systems: Diccionario con configuraciones de sistemas
        episodes_per_system: Episodios de entrenamiento por sistema
        batch_size: Tamaño del batch para experience replay
        model_path: Ruta donde guardar el modelo entrenado
        checkpoint_dir: Directorio para guardar los checkpoints
        checkpoint_frequency: Cada cuántos episodios guardar un checkpoint
        resume_from_checkpoint: Si se debe reanudar desde el último checkpoint disponible
        patience: Número de episodios sin mejora antes de detener el entrenamiento (early stopping)
        
    Returns:
        El agente entrenado
    """
    import os
    import glob
    from datetime import datetime
    import numpy as np
    
    # Crear directorio de checkpoints si no existe
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Definir dimensiones del modelo (suficientemente grandes para cualquier caso)
    state_size = 24
    action_size = 150
    
    # Crear agente
    agent = GeneralizedDQNAgent(state_size, action_size)
    
    # Orden de entrenamiento: sistemas más simples primero
    system_order = sorted(systems.keys())
    
    # Variables para seguimiento de progreso y checkpoints
    current_system_idx = 0
    current_episode = 0
    
    # Buscar el checkpoint más reciente si resume_from_checkpoint es True
    latest_checkpoint = None
    if resume_from_checkpoint:
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_sys_*_ep_*.h5"))
        if checkpoint_files:
            # Ordenar por fecha de modificación (el más reciente al final)
            checkpoint_files.sort(key=os.path.getmtime)
            latest_checkpoint = checkpoint_files[-1]
            
            # Extraer índice de sistema y episodio del nombre del archivo
            try:
                # Formato esperado: checkpoint_sys_X_ep_Y.h5
                filename = os.path.basename(latest_checkpoint)
                parts = filename.split('_')
                current_system_idx = int(parts[2])
                current_episode = int(parts[4].split('.')[0])
                
                print(f"Encontrado checkpoint: {latest_checkpoint}")
                print(f"Reanudando desde sistema {current_system_idx}, episodio {current_episode}")
                
                # Cargar pesos del modelo desde el checkpoint
                agent.load(latest_checkpoint)
                
                # Avanzar al siguiente episodio
                current_episode += 1
                
                # Si ya se completaron todos los episodios para este sistema, avanzar al siguiente
                if current_episode >= episodes_per_system:
                    current_system_idx += 1
                    current_episode = 0
            except (IndexError, ValueError) as e:
                print(f"Error al analizar nombre de checkpoint, iniciando desde cero: {e}")
                current_system_idx = 0
                current_episode = 0
    
    # Si no se encontró checkpoint o no se quiere reanudar, intentar cargar un modelo pre-entrenado
    if latest_checkpoint is None:
        try:
            print(f"Intentando cargar modelo pre-entrenado desde {model_path}...")
            model_loaded = agent.load(model_path)
            if model_loaded:
                print("Modelo pre-entrenado cargado exitosamente.")
                # Ajustar epsilon para menos exploración
                agent.epsilon = 0.2
        except Exception as e:
            print(f"No se pudo cargar modelo pre-entrenado: {e}")
            print("Entrenando nuevo modelo generalizado...")
    
    # Archivo de registro para el progreso del entrenamiento
    log_path = os.path.join(checkpoint_dir, "training_log.txt")
    with open(log_path, "a") as log_file:
        log_file.write(f"\n\n--- Nueva sesión de entrenamiento: {datetime.now()} ---\n")
        if latest_checkpoint:
            log_file.write(f"Reanudando desde checkpoint: {latest_checkpoint}\n")
        
        # Entrenar en cada sistema, comenzando desde el índice actual
        for system_idx in range(current_system_idx, len(system_order)):
            system_num = system_order[system_idx]
            config = systems[system_num]
            
            log_message = f"\n{'='*50}\nEntrenando en sistema {system_num} ({system_idx+1}/{len(systems)})\n{'='*50}"
            print(log_message)
            log_file.write(f"{log_message}\n")
            
            env = EVChargingEnv(config)
            
            # Comenzar desde el episodio actual para el primer sistema, desde 0 para los demás
            start_episode = current_episode if system_idx == current_system_idx else 0
            
            log_message = f"Ejecutando episodios de entrenamiento desde {start_episode+1} hasta {episodes_per_system}..."
            print(log_message)
            log_file.write(f"{log_message}\n")
            
            # Variables para early stopping
            best_reward = -float('inf')
            no_improvement_count = 0
            best_model_path = os.path.join(checkpoint_dir, f"best_model_sys_{system_idx}.h5")
            
            for e in range(start_episode, episodes_per_system):
                state = env.reset()
                total_reward = 0
                done = False
                
                # Contadores para diagnóstico
                skip_count = 0
                assign_count = 0
                
                while not done:
                    possible_actions = env._get_possible_actions(state)
                    action = agent.act(state, possible_actions)
                    
                    if action == -1:  # No hay acciones posibles
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
                    
                    # Entrenar con batches
                    if len(agent.system_memories[system_num]) > batch_size:
                        agent.replay(batch_size, system_specific=True, system_type=system_num)
                    
                    if len(agent.memory) > batch_size:
                        agent.replay(batch_size)
                
                log_message = f"Episodio {e+1}/{episodes_per_system}, Recompensa: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}, EVs asignados: {assign_count}, EVs saltados: {skip_count}"
                print(log_message)
                log_file.write(f"{log_message}\n")
                
                # Guardar checkpoint cada checkpoint_frequency episodios o al final de cada sistema
                if (e + 1) % checkpoint_frequency == 0 or e == episodes_per_system - 1:
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_sys_{system_idx}_ep_{e+1}.h5")
                    agent.save(checkpoint_path)
                    log_message = f"Checkpoint guardado en {checkpoint_path}"
                    print(log_message)
                    log_file.write(f"{log_message}\n")
                    
                    # También guardar la versión actual como modelo principal
                    agent.save(model_path)
                    log_file.write(f"Modelo principal actualizado en {model_path}\n")
                    
                    # Flush para asegurar que se escriba en caso de interrupción
                    log_file.flush()
                
                # Early stopping: comprobar si la recompensa ha mejorado
                if total_reward > best_reward:
                    best_reward = total_reward
                    no_improvement_count = 0
                    # Guardar el mejor modelo para este sistema
                    agent.save(best_model_path)
                    log_message = f"Nuevo mejor modelo guardado con recompensa: {best_reward:.2f}"
                    print(log_message)
                    log_file.write(f"{log_message}\n")
                else:
                    no_improvement_count += 1
                    log_message = f"No hay mejora en el episodio. Contador de paciencia: {no_improvement_count}/{patience}"
                    print(log_message)
                    log_file.write(f"{log_message}\n")
                    
                    # Aplicar early stopping si no hay mejora durante 'patience' episodios
                    if no_improvement_count >= patience:
                        log_message = f"Early stopping activado tras {patience} episodios sin mejora. Cargando mejor modelo..."
                        print(log_message)
                        log_file.write(f"{log_message}\n")
                        
                        # Cargar el mejor modelo antes de pasar al siguiente sistema
                        agent.load(best_model_path)
                        break
    
    # Guardar el modelo entrenado final
    print(f"\nGuardando modelo generalizado final en {model_path}...")
    agent.save(model_path)
    
    return agent

def visualize_solution(solution, config):
    """
    Visualiza los gráficos de carga y parqueo para una solución dada.
    
    Args:
        solution (dict): Diccionario de solución en el formato {ev_id: [ [t_start, t_end, charger, slot, power], ... ], ...}.
        config (dict): Configuración del sistema que debe contener al menos:
                       - "times": lista de tiempos
                       - "dt": delta tiempo
                       - "n_spots": número de plazas de parqueo
                       - "prices": lista de precios (para plot de carga, si se desea)
    """
    # Llamamos a la función de plot para el perfil de carga.
    plot_charging_schedule(solution, config)
    
    # Calculamos el tiempo total (tiempo final + dt) para el diagrama de parqueo.
    total_time = config["times"][-1] + config["dt"]
    plot_parking_schedule(solution, total_time, config["dt"], config["n_spots"])
    
def visualize_solution_from_file(solution_file, config_file):
    """
    Carga la solución y la configuración desde archivos JSON y visualiza los gráficos.
    
    Args:
        solution_file (str): Ruta del archivo JSON que contiene la solución.
        config_file (str): Ruta del archivo JSON que contiene la configuración del sistema.
    """
    import json
    config = load_data(config_file)
    with open(solution_file, "r") as f:
        data = json.load(f)
    # Si el archivo tiene la clave "schedule", la usamos; de lo contrario, se asume que data es el schedule.
    if "schedule" in data:
        solution = data["schedule"]
    else:
        solution = data
    visualize_solution(solution, config)


def analyze_parking_overflow(config):
    """
    Analiza cuándo hay más vehículos queriendo entrar que slots disponibles.
    
    Args:
        config: Configuración del sistema
        
    Returns:
        Un diccionario con análisis de capacidad y vehículos rechazados
    """
    # Obtener datos del sistema
    arrivals = config["arrivals"]
    times = config["times"]
    prices = config["prices"]
    dt = config["dt"]
    n_spots = config["n_spots"]
    
    # Mapear datos de vehículos
    ev_data = {
        arr["id"]: {
            "arrival": arr["arrival_time"],
            "departure": arr["departure_time"],
            "required_energy": arr["required_energy"]
        } for arr in arrivals
    }
    
    # Analizar ocupación por intervalo horario
    hourly_occupancy = {}
    for t_idx, t in enumerate(times):
        # Vehículos presentes en este intervalo
        evs_in_interval = [
            ev_id for ev_id, data in ev_data.items()
            if data["arrival"] <= t < data["departure"]
        ]
        
        hourly_occupancy[t] = {
            "total_evs": len(evs_in_interval),
            "capacity": n_spots,
            "overflow": max(0, len(evs_in_interval) - n_spots),
            "ev_ids": evs_in_interval,
            "time_index": t_idx
        }
    
    # Encontrar períodos con sobrecarga
    overflow_periods = {
        t: data for t, data in hourly_occupancy.items()
        if data["overflow"] > 0
    }
    
    # Si hay sobrecarga, simular una priorización simple
    rejected_evs = set()
    ev_priority = {}
    
    if overflow_periods:
        # Calcular métricas de prioridad para cada EV
        for ev_id, data in ev_data.items():
            stay_duration = data["departure"] - data["arrival"]
            charging_intensity = data["required_energy"] / stay_duration if stay_duration > 0 else float('inf')
            
            # Mayor prioridad para vehículos con mayor necesidad de carga por unidad de tiempo
            # y que permanecen menos tiempo
            priority_score = charging_intensity / stay_duration if stay_duration > 0 else 0
            ev_priority[ev_id] = priority_score
        
        # Simular asignación de slots en períodos con sobrecarga
        for t, data in sorted(overflow_periods.items()):
            evs_present = data["ev_ids"]
            # Ordenar por prioridad (mayor a menor)
            prioritized_evs = sorted(evs_present, key=lambda ev: ev_priority.get(ev, 0), reverse=True)
            
            # Asignar slots a los primeros n_spots vehículos
            assigned_evs = set(prioritized_evs[:n_spots]) 
            
            # Los demás son rechazados
            rejected_at_t = set(prioritized_evs[n_spots:])
            rejected_evs.update(rejected_at_t)
    
    # Calcular ingresos potenciales perdidos
    lost_revenue = 0
    rejected_details = {}
    
    for ev_id in rejected_evs:
        data = ev_data[ev_id]
        # Períodos donde este EV estaría presente
        ev_times = [t for t in times if data["arrival"] <= t < data["departure"]]
        
        if ev_times:
            # Calcular precio promedio durante su estancia
            avg_price = sum(prices[times.index(t)] for t in ev_times) / len(ev_times)
            
            # Calcular ingreso potencial (asumiendo margen del 50%)
            potential_revenue = data["required_energy"] * avg_price * 0.5
            lost_revenue += potential_revenue
            
            rejected_details[ev_id] = {
                "arrival": data["arrival"],
                "departure": data["departure"],
                "energy_required": data["required_energy"],
                "potential_revenue": potential_revenue
            }
    
    # Compilar reporte
    report = {
        "total_evs": len(ev_data),
        "parking_capacity": n_spots,
        "overflow_periods": len(overflow_periods),
        "overflow_details": overflow_periods,
        "rejected_evs": len(rejected_evs),
        "rejected_ev_ids": list(rejected_evs),
        "rejected_details": rejected_details,
        "total_lost_revenue": lost_revenue
    }
    
    return report


def modify_milp_for_capacity_constraints(config):
    """
    Modifica la configuración para manejar escenarios donde hay más vehículos
    que slots disponibles.
    
    Este enfoque pre-filtra vehículos para no intentar satisfacer demandas imposibles.
    
    Args:
        config: Configuración original
        
    Returns:
        Configuración modificada con cantidad manejable de vehículos
    """
    # Analizar capacidad
    capacity_report = analyze_parking_overflow(config)
    
    # Si no hay problemas de capacidad, devolver configuración original
    if capacity_report["rejected_evs"] == 0:
        return config, capacity_report
    
    # Crear una copia de la configuración original
    import copy
    modified_config = copy.deepcopy(config)
    
    # Filtrar los vehículos rechazados
    rejected_ids = set(capacity_report["rejected_ev_ids"])
    modified_config["arrivals"] = [
        ev for ev in modified_config["arrivals"]
        if ev["id"] not in rejected_ids
    ]
    
    print(f"Configuración modificada: {len(modified_config['arrivals'])}/{len(config['arrivals'])} vehículos incluidos")
    print(f"Se excluyeron {capacity_report['rejected_evs']} vehículos por limitaciones de capacidad")
    print(f"Ingreso potencial perdido por vehículos rechazados: ${capacity_report['total_lost_revenue']:.2f}")
    
    return modified_config, capacity_report

#########################################
# Funciones de generación de solución
#########################################
def generate_rl_solution(config, agent=None, model_path="ev_scheduler_generalized_model.h5"):
    """
    Genera una solución de scheduling utilizando el agente RL generalizado.
    Carga un modelo pre-entrenado si no se proporciona un agente.
    
    Args:
        config: Configuración del sistema
        agent: Agente RL pre-entrenado (opcional)
        model_path: Ruta al modelo generalizado pre-entrenado
        
    Returns:
        schedule_rl: lista de tuplas (ev_id, t_slot, charger_id, slot, power)
    """
    print("Generando solución RL...")
    
    # Si no se proporciona un agente, intentar cargar el modelo generalizado
    if agent is None:
        print(f"Intentando cargar modelo generalizado desde {model_path}...")
        state_size = 24  # Suficientemente grande para cualquier caso
        action_size = 150  # Suficientemente grande para cualquier caso
        
        agent = GeneralizedDQNAgent(state_size, action_size)
        
        # Intentar cargar el modelo pre-entrenado
        model_loaded = agent.load(model_path)
        
        if not model_loaded:
            print("No se pudo cargar el modelo generalizado. Entrenando uno básico...")
            # Entrenar un agente básico específico para este sistema
            system_num = config.get("test_number", 0)
            systems = {system_num: config}
            agent = train_generalized_agent(systems, episodes_per_system=30, model_path=model_path)
    
    # En sistemas muy grandes, procesar por lotes para evitar problemas de memoria
    if len(config['arrivals']) > 200:
        print("Sistema grande detectado. Procesando por lotes...")
        return generate_rl_solution_batched(config, agent)
    
    # Procesamiento estándar
    env = EVChargingEnv(config)
    state = env.reset()
    done = False
    
    # Configurar agente en modo evaluación (menos exploración)
    original_epsilon = agent.epsilon
    agent.epsilon = 0.05  # Poca exploración para generar soluciones de calidad
    
    while not done:
        possible_actions = env._get_possible_actions(state)
        action = agent.act(state, possible_actions)
        
        if action == -1 or action >= len(possible_actions):
            # No hay acciones posibles o índice inválido
            break
        
        state, _, done = env.step(action)
    
    # Restaurar epsilon original
    agent.epsilon = original_epsilon
    
    # Obtener el schedule generado
    schedule_rl = env.get_schedule()
    
    unique_evs = len(set(entry[0] for entry in schedule_rl))
    print(f"Solución RL generada: {len(schedule_rl)} asignaciones para {unique_evs}/{len(config['arrivals'])} EVs ({unique_evs/len(config['arrivals'])*100:.2f}%)")
    
    return schedule_rl

def generate_rl_solution_batched(config, agent):
    """
    Versión por lotes de la generación de solución RL para sistemas grandes.
    """
    all_evs = config['arrivals'].copy()
    batch_size = min(50, max(10, len(all_evs) // 10))  # Tamaño adaptativo
    
    # Crear lotes
    ev_batches = []
    for i in range(0, len(all_evs), batch_size):
        batch = all_evs[i:i+batch_size]
        ev_batches.append(batch)
    
    print(f"Procesando en {len(ev_batches)} lotes de hasta {batch_size} EVs...")
    
    all_schedules = []
    
    for i, batch in enumerate(ev_batches):
        print(f"Procesando lote {i+1}/{len(ev_batches)}...")
        
        # Crear configuración temporal para este lote
        batch_config = config.copy()
        batch_config['arrivals'] = batch
        
        # Generar solución para este lote
        env = EVChargingEnv(batch_config)
        state = env.reset()
        done = False
        
        # Configurar agente en modo evaluación
        original_epsilon = agent.epsilon
        agent.epsilon = 0.05
        
        while not done:
            possible_actions = env._get_possible_actions(state)
            action = agent.act(state, possible_actions)
            
            if action == -1 or action >= len(possible_actions):
                break
            
            state, _, done = env.step(action)
        
        # Restaurar epsilon
        agent.epsilon = original_epsilon
        
        # Obtener el schedule generado para este lote
        batch_schedule = env.get_schedule()
        all_schedules.extend(batch_schedule)
        
        unique_evs = len(set(entry[0] for entry in batch_schedule))
        print(f"Lote {i+1}: {len(batch_schedule)} asignaciones para {unique_evs}/{len(batch)} EVs")
    
    # Combinar todos los lotes
    return all_schedules

def save_schedule_to_json(schedule, extra_info=None, file_path="resultados.json"):
    # Convertir las tuplas a listas para serialización
    schedule_serializable = {
        str(ev_id): [list(interval) for interval in intervals]
        for ev_id, intervals in schedule.items()
    }
    if extra_info is not None:
        data_to_save = {
            "schedule": schedule_serializable,
            "extra_info": extra_info
        }
    else:
        data_to_save = schedule_serializable
    with open(file_path, "w") as f:
        json.dump(data_to_save, f, indent=4)
    print(f"Resultados guardados en {file_path}")


#########################################
# Modelo MILP para refinamiento
#########################################
def solve_ev_schedule(json_path, penalty_unmet=1000.0, rl_schedule=None, time_limit=None):
    """
    Resuelve el problema de scheduling de carga con MILP, usando una solución RL inicial.
    
    Se agregan variables de holgura para relajar:
      - Restricción de capacidad de cada cargador
      - Restricción del límite del transformador
      - Restricción de capacidad del parqueadero
      - Restricción de asignación de slot (para garantizar que se asigne un slot, aunque sea con violación)
    
    Estas holguras se penalizan fuertemente en la función objetivo para que solo se utilicen si es absolutamente necesario.
    
    Args:
        json_path: Ruta al archivo JSON o configuración directa.
        penalty_unmet: Penalización por demanda insatisfecha (por kWh no entregado).
        rl_schedule: Solución RL inicial para warm start.
        time_limit: Límite de tiempo para resolver MILP (segundos).
        
    Returns:
        model: Modelo MILP resuelto.
        schedule: Diccionario con la solución optimizada. Las claves son EV IDs y los valores listas de tuplas 
                  (t_start, t_end, charger_id, slot, power).
        rejected_details: Diccionario con, para cada EV no completamente atendido, la energía requerida,
                           energía entregada, energía no cubierta y la penalización calculada.
    """
    # Coeficientes de penalización para las holguras
    M_slack = 1e32

    # Cargar datos
    if isinstance(json_path, dict):
        data = json_path
    else:
        data = load_data(json_path)
        
    times = data["times"]
    prices = data["prices"]
    arrivals = data["arrivals"]
    chargers = data["chargers"]
    station_limit = data["station_limit"]
    dt = data["dt"]
    n_spots = data["n_spots"]

    T = range(len(times))
    EVs = [arr["id"] for arr in arrivals]
    arrival_time = {arr["id"]: arr["arrival_time"] for arr in arrivals}
    departure_time = {arr["id"]: arr["departure_time"] for arr in arrivals}
    required_energy = {arr["id"]: arr["required_energy"] for arr in arrivals}

    charger_ids = [c["charger_id"] for c in chargers]
    max_charger_power = {c["charger_id"]: c["power"] for c in chargers}

    # Crear problema MILP
    model = LpProblem("Partial_Charging_MinCost", LpMinimize)

    # Variables de decisión principales: x y y
    x = {}  # Potencia asignada
    y = {}  # Variable binaria: conexión a cargador
    for i in EVs:
        for t in T:
            for c in charger_ids:
                x[(i, t, c)] = LpVariable(f"x_{i}_{t}_{c}", lowBound=0)
                y[(i, t, c)] = LpVariable(f"y_{i}_{t}_{c}", cat="Binary")
                # Inicialización con solución RL si se proporciona
                if rl_schedule is not None:
                    for entry in rl_schedule:
                        if entry[0] == i and entry[1] == t and entry[2] == c:
                            x[(i, t, c)].setInitialValue(entry[4])
                            y[(i, t, c)].setInitialValue(1)
                            break

    # Variable para energía no cubierta
    u = { i: LpVariable(f"u_{i}", lowBound=0) for i in EVs }

    # Variable binaria: EV estacionado en intervalo t
    z = {}
    for i in EVs:
        for t in T:
            if arrival_time[i] <= times[t] < departure_time[i]:
                z[(i, t)] = LpVariable(f"z_{i}_{t}", cat="Binary")
            else:
                z[(i, t)] = 0

    # Variables binarias: asignación de slot en cada intervalo para cada EV
    w = {}
    for i in EVs:
        for t in T:
            if arrival_time[i] <= times[t] < departure_time[i]:
                for s in range(n_spots):
                    w[(i, t, s)] = LpVariable(f"w_{i}_{t}_{s}", cat="Binary")

    # --- Variables de holgura (slack) ---
    # 1. Para capacidad de cada cargador: s_charger[(t,c)] >= 0
    s_charger = {}
    for t in T:
        for c in charger_ids:
            s_charger[(t, c)] = LpVariable(f"s_charger_{t}_{c}", lowBound=0)
    # 2. Para el límite del transformador: s_transformer[t] >= 0
    s_transformer = {t: LpVariable(f"s_transformer_{t}", lowBound=0) for t in T}
    # 3. Para capacidad del parqueadero: s_parking[t] >= 0
    s_parking = {t: LpVariable(f"s_parking_{t}", lowBound=0) for t in T}
    # 4. Para asignación de slot: s_slot[(i,t)] >= 0 para cada EV y t en ventana
    s_slot = {}
    for i in EVs:
        for t in T:
            if arrival_time[i] <= times[t] < departure_time[i]:
                s_slot[(i, t)] = LpVariable(f"s_slot_{i}_{t}", lowBound=0)

    # Función objetivo: minimizar costo de carga, penalización por energía no cubierta
    # y penalizaciones por las holguras.
    model += (
        lpSum(prices[t] * dt * x[(i, t, c)]
              for i in EVs for t in T for c in charger_ids)
        + penalty_unmet * lpSum(u[i] for i in EVs)
        + M_slack * ( lpSum(s_charger[(t, c)] for t in T for c in charger_ids)
                     + lpSum(s_transformer[t] for t in T)
                     + lpSum(s_parking[t] for t in T)
                     + lpSum(s_slot[(i, t)] for i in EVs for t in T if arrival_time[i] <= times[t] < departure_time[i])
        ),
        "MinCost_Partial"
    )

    # Restricción 1: Carga entregada + energía no cubierta = energía requerida
    for i in EVs:
        model += (
            lpSum(x[(i, t, c)] * dt for t in T for c in charger_ids if times[t] >= arrival_time[i] and times[t] < departure_time[i])
            + u[i]
            == required_energy[i]
        ), f"PartialCharge_EV_{i}"

    # Restricción 2: Capacidad de cada cargador (con slack)
    for t in T:
        for c in charger_ids:
            model += (
                lpSum(x[(i, t, c)] for i in EVs) <= max_charger_power[c] + s_charger[(t, c)]
            ), f"ChargerCap_t{t}_c{c}"

    # Restricción 3: Límite del transformador (con slack)
    for t in T:
        model += (
            lpSum(x[(i, t, c)] for i in EVs for c in charger_ids) <= station_limit + s_transformer[t]
        ), f"StationCap_t{t}"

    # Restricción 4: No cargar fuera de la ventana de disponibilidad
    for i in EVs:
        for t in T:
            if not (arrival_time[i] <= times[t] < departure_time[i]):
                for c in charger_ids:
                    model += x[(i, t, c)] == 0, f"NoChargeOutside_{i}_{t}_{c}"

    # Restricción 5: Asignación única de cargador en cada intervalo
    for i in EVs:
        for t in T:
            model += lpSum(y[(i, t, c)] for c in charger_ids) <= 1, f"UniqueCharger_EV_{i}_t{t}"

    # Restricción 6: Vinculación entre x e y
    for i in EVs:
        for t in T:
            for c in charger_ids:
                model += x[(i, t, c)] <= y[(i, t, c)] * max_charger_power[c], f"Link_x_y_EV_{i}_t{t}_c{c}"

    # Restricción 7: Vinculación entre z e y
    for i in EVs:
        for t in T:
            if isinstance(z[(i, t)], LpVariable):
                model += z[(i, t)] >= lpSum(y[(i, t, c)] for c in charger_ids) / len(charger_ids), f"Link_z_y_EV_{i}_t{t}"

    # Restricción 8: Capacidad del parqueadero (con slack)
    for t in T:
        model += lpSum(z[(i, t)] for i in EVs if isinstance(z[(i, t)], LpVariable)) <= n_spots + s_parking[t], f"ParkingCap_t{t}"

    # Restricción 9: Asignación de slot de parqueo (con slack)
    for i in EVs:
        for t in T:
            if arrival_time[i] <= times[t] < departure_time[i]:
                model += lpSum(w[(i, t, s)] for s in range(n_spots)) + s_slot[(i, t)] == 1, f"AssignSlot_EV_{i}_t{t}"

    # Restricción 10: Unicidad de slot en cada intervalo
    for t in T:
        for s in range(n_spots):
            evs_in_interval = [i for i in EVs if arrival_time[i] <= times[t] < departure_time[i]]
            if evs_in_interval:
                model += lpSum(w[(i, t, s)] for i in evs_in_interval) <= 1, f"SlotUnique_t{t}_s{s}"

    # Configurar tiempo límite y gap, si se especifica
    if time_limit:
        print(f"Configurando MILP con tiempo límite de {time_limit} segundos y gap de 0.01 (1%)")
        solver = pulp.PULP_CBC_CMD(msg=True, options=[f"sec {time_limit}", "timeMode elapsed", "ratioGap 0.01"])
    else:
        solver = pulp.PULP_CBC_CMD(msg=True, options=["ratioGap 0.01"])

    # Resolver el modelo
    start_time_model = time.time()
    try:
        model.solve(solver)
        solve_time = time.time() - start_time_model
        print("Status:", LpStatus[model.status])
        # Si el modelo sigue siendo infactible, ya tendremos holguras, pero seguimos
        print("Objective value (Total Cost):", model.objective.value())
        print("Tiempo de resolución MILP:", solve_time, "segundos")
    except Exception as e:
        print(f"Error al resolver MILP: {e}")
        print("Devolviendo solución RL original")
        return None, {} if rl_schedule is None else convert_rl_schedule_to_dict(rl_schedule, times, dt)

    # Extraer solución: construir schedule con asignaciones de tiempo, cargador, slot y potencia
    schedule = defaultdict(list)
    for i in EVs:
        for t in T:
            for c in charger_ids:
                var_val = x[(i, t, c)].varValue
                if var_val is not None and var_val > 1e-4:
                    for s in range(n_spots):
                        if (i, t, s) in w and w[(i, t, s)].varValue is not None and w[(i, t, s)].varValue > 0.5:
                            schedule[i].append((times[t], times[t] + dt, c, s, var_val))
                            break

    # Generar reporte de EVs parcialmente o no atendidos
    rejected_details = {}
    for i in EVs:
        delivered_energy = sum([entry[4] for entry in schedule[i]]) if i in schedule else 0
        unmet_energy = required_energy[i] - delivered_energy
        if unmet_energy > 1e-4:
            penalty_cost = penalty_unmet * unmet_energy
            rejected_details[i] = {
                "required_energy": required_energy[i],
                "delivered_energy": delivered_energy,
                "unmet_energy": unmet_energy,
                "penalty_cost": penalty_cost
            }

    return model, schedule, rejected_details



def convert_rl_schedule_to_dict(rl_schedule, times, dt):
    """Convierte una lista de asignaciones RL a un diccionario para visualización."""
    schedule_dict = defaultdict(list)
    for (ev_id, t_idx, charger_id, slot, power) in rl_schedule:
        t_start = times[t_idx]
        t_end = t_start + dt
        schedule_dict[ev_id].append((t_start, t_end, charger_id, slot, power))
    return schedule_dict

#########################################
# Funciones para visualización
#########################################

def check_training_progress(checkpoint_dir="./checkpoints"):
    """
    Verifica el progreso del entrenamiento revisando los checkpoints disponibles.
    
    Args:
        checkpoint_dir: Directorio donde se guardan los checkpoints
        
    Returns:
        Un reporte sobre el estado del entrenamiento y los checkpoints disponibles
    """
    import os
    import glob
    import time
    from datetime import datetime
    
    print(f"\nVerificando progreso de entrenamiento en {checkpoint_dir}...")
    
    # Verificar si el directorio existe
    if not os.path.exists(checkpoint_dir):
        print(f"El directorio {checkpoint_dir} no existe. No hay checkpoints disponibles.")
        return None
    
    # Buscar archivos de checkpoint
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_sys_*_ep_*.h5"))
    
    if not checkpoint_files:
        print("No se encontraron archivos de checkpoint.")
        return None
    
    # Ordenar por fecha de modificación
    checkpoint_files.sort(key=os.path.getmtime)
    
    # Obtener información sobre los checkpoints
    checkpoints_info = []
    systems_completed = set()
    max_system_idx = -1
    
    for cp_file in checkpoint_files:
        # Extraer información del nombre
        filename = os.path.basename(cp_file)
        try:
            parts = filename.split('_')
            system_idx = int(parts[2])
            episode = int(parts[4].split('.')[0])
            
            max_system_idx = max(max_system_idx, system_idx)
            
            # Obtener stats del archivo
            file_size_mb = os.path.getsize(cp_file) / (1024 * 1024)
            mod_time = datetime.fromtimestamp(os.path.getmtime(cp_file))
            
            # Si tenemos el último episodio de un sistema, considerarlo completado
            if "final" in filename or episode >= 30:  # Asumiendo 30 como número típico de episodios
                systems_completed.add(system_idx)
                
            checkpoints_info.append({
                "file": cp_file,
                "system": system_idx,
                "episode": episode,
                "size_mb": file_size_mb,
                "modified": mod_time
            })
        except (IndexError, ValueError):
            # Ignorar archivos con formato incorrecto
            continue
    
    # Encontrar el checkpoint más reciente
    latest_checkpoint = checkpoints_info[-1] if checkpoints_info else None
    
    # Ver log de entrenamiento si existe
    log_path = os.path.join(checkpoint_dir, "training_log.txt")
    recent_log_entries = []
    
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            lines = f.readlines()
            # Mostrar las últimas 10 líneas no vacías del log
            recent_log_entries = [line.strip() for line in lines[-20:] if line.strip()]
    
    # Imprimir reporte
    print("\n=== REPORTE DE PROGRESO DE ENTRENAMIENTO ===")
    print(f"Total de checkpoints: {len(checkpoints_info)}")
    print(f"Sistemas completados: {len(systems_completed)} (hasta sistema {max_system_idx})")
    
    if latest_checkpoint:
        print(f"\nCheckpoint más reciente:")
        print(f"  Sistema: {latest_checkpoint['system']}")
        print(f"  Episodio: {latest_checkpoint['episode']}")
        print(f"  Fecha: {latest_checkpoint['modified']}")
        print(f"  Archivo: {os.path.basename(latest_checkpoint['file'])}")
    
    if recent_log_entries:
        print("\nEntradas recientes del log de entrenamiento:")
        for entry in recent_log_entries:
            print(f"  {entry}")
    
    return {
        "checkpoints": checkpoints_info,
        "latest": latest_checkpoint,
        "systems_completed": systems_completed,
        "max_system": max_system_idx
    }


def plot_training_progress(checkpoint_dir="./checkpoints"):
    """
    Genera gráficas de progreso del entrenamiento a partir del log.
    
    Args:
        checkpoint_dir: Directorio donde se guarda el log de entrenamiento
    """
    import os
    import re
    import matplotlib.pyplot as plt
    import numpy as np
    
    log_path = os.path.join(checkpoint_dir, "training_log.txt")
    
    if not os.path.exists(log_path):
        print(f"No se encontró el archivo de log en {log_path}")
        return
    
    # Patrones para extraer información
    system_pattern = re.compile(r"Entrenando en sistema (\d+)")
    episode_pattern = re.compile(r"Episodio (\d+)/\d+, Recompensa: ([-\d.]+), Epsilon: ([\d.]+)")
    
    # Estructuras para almacenar datos
    systems = {}
    current_system = None
    
    with open(log_path, 'r') as f:
        for line in f:
            # Buscar cambio de sistema
            system_match = system_pattern.search(line)
            if system_match:
                current_system = int(system_match.group(1))
                if current_system not in systems:
                    systems[current_system] = {
                        "episodes": [],
                        "rewards": [],
                        "epsilons": []
                    }
                continue
            
            # Buscar información de episodio
            if current_system is not None:
                episode_match = episode_pattern.search(line)
                if episode_match:
                    episode = int(episode_match.group(1))
                    reward = float(episode_match.group(2))
                    epsilon = float(episode_match.group(3))
                    
                    systems[current_system]["episodes"].append(episode)
                    systems[current_system]["rewards"].append(reward)
                    systems[current_system]["epsilons"].append(epsilon)
    
    if not systems:
        print("No se encontraron datos de entrenamiento en el log.")
        return
    
    # Crear visualizaciones
    plt.figure(figsize=(12, 10))
    
    # Gráfica de recompensas
    plt.subplot(2, 1, 1)
    for system, data in sorted(systems.items()):
        if data["episodes"]:
            plt.plot(data["episodes"], data["rewards"], marker='o', linestyle='-', label=f"Sistema {system}")
    
    plt.title("Recompensas por episodio")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa")
    plt.grid(True)
    plt.legend()
    
    # Gráfica de epsilon
    plt.subplot(2, 1, 2)
    for system, data in sorted(systems.items()):
        if data["episodes"]:
            plt.plot(data["episodes"], data["epsilons"], marker='x', linestyle='-', label=f"Sistema {system}")
    
    plt.title("Valor de epsilon por episodio")
    plt.xlabel("Episodio")
    plt.ylabel("Epsilon")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # Guardar y mostrar
    plt.savefig(os.path.join(checkpoint_dir, "training_progress.png"))
    print(f"Gráfica de progreso guardada en {os.path.join(checkpoint_dir, 'training_progress.png')}")
    plt.show()
    
    
def plot_charging_schedule(schedule, config):
    """
    Grafica los perfiles de carga de los EVs (sin límite de visualización).
    """
    times = config["times"]
    dt = config["dt"]
    prices = config["prices"]
    
    # Crear una figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Crear una paleta de colores para los EVs
    # Usar una paleta con suficientes colores o colores cíclicos
    num_evs = len(schedule)
    cmap = plt.cm.get_cmap('tab20', 20)  # Usamos 20 colores base que se repetirán
    
    # Para cada EV, graficar su perfil de carga
    for idx, (ev_id, intervals) in enumerate(schedule.items()):
        # Eliminar el límite de 20 EVs
        # Ordenar los intervalos por tiempo
        intervals_sorted = sorted(intervals, key=lambda x: x[0])
        
        # Extraer tiempos y potencias
        time_points = [interval[0] for interval in intervals_sorted]
        power_values = [interval[4] for interval in intervals_sorted]
        
        # Graficar el perfil de carga (con colores cíclicos)
        ax1.step(time_points, power_values, where='post', 
                label=f'EV {ev_id}', color=cmap(idx % 20))
    
    # Graficar los precios de energía
    ax2.step(times, prices, where='post', color='red', label='Precio de energía')
    ax2.set_xlabel('Tiempo (horas)')
    ax2.set_ylabel('Precio ($/kWh)')
    ax2.grid(True)
    
    # Configuración de los ejes
    ax1.set_title(f'Perfiles de carga de vehículos eléctricos ({num_evs} EVs)')
    ax1.set_ylabel('Potencia (kW)')
    ax1.grid(True)
    
    # Manejar la leyenda según la cantidad de EVs
    if num_evs <= 30:  # Para un número razonable de EVs, mostrar la leyenda
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    else:
        # Para muchos EVs, omitir la leyenda pero indicar el total
        ax1.text(0.99, 0.99, f'Total: {num_evs} EVs', 
                 transform=ax1.transAxes, ha='right', va='top',
                 bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('charging_profiles.png')
    plt.show()
    
def plot_parking_schedule(schedule, total_time, dt, n_spots):
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("tab20")
    ev_colors = {}
    
    # Usar TODOS los EVs para la visualización
    evs_to_show = list(schedule.keys())
    
    for idx, ev in enumerate(evs_to_show):
        ev_colors[ev] = cmap(idx % 20)
        for (t_start, t_end, charger, slot, power) in schedule[ev]:
            ax.broken_barh([(t_start, dt)], (slot - 0.4, 0.8),
                           facecolors=ev_colors[ev])
            # Mostrar la etiqueta del EV en cada slot
            ax.text(t_start + dt/2, slot, f"EV {ev}", color='white',
                    ha="center", va="center", fontsize=8)
    
    ax.set_xlabel("Tiempo (horas)")
    ax.set_ylabel("Plazas de parqueo")
    ax.set_title("Asignación de EVs a plazas de parqueo en el tiempo")
    ax.set_xlim(0, total_time)
    ax.set_ylim(-0.5, n_spots - 0.5)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('parking_assignment_all_slots.png')
    plt.show()


#########################################
# Código principal
#########################################
import os
import json
import tempfile
import time
import glob
import pulp
from collections import defaultdict

# Supongamos que ya tienes definidas todas las funciones:
# load_data, load_all_test_systems, GeneralizedDQNAgent,
# generate_rl_solution, convert_rl_schedule_to_dict, solve_ev_schedule,
# save_schedule_to_json, check_training_progress, plot_training_progress,
# plot_charging_schedule, plot_parking_schedule, train_generalized_agent
# (estas funciones son las que ya has ido definiendo en tu código)

def load_rl_schedule_from_json(file_path, config):
    """
    Carga el archivo JSON con la solución RL (guardado en formato diccionario)
    y lo convierte en una lista de tuplas en el formato original.
    """
    rl_data = json.load(open(file_path))
    # Si el JSON tiene la clave "schedule", usamos ese diccionario
    if "schedule" in rl_data:
        rl_data = rl_data["schedule"]
    rl_schedule = []
    # rl_data es un diccionario: key (EV id) -> list de intervalos [t_start, t_end, charger, slot, power]
    for ev_id_str, intervals in rl_data.items():
        for interval in intervals:
            try:
                t_idx = config["times"].index(interval[0])
            except ValueError:
                t_idx = 0
            try:
                ev_id = int(ev_id_str)
            except:
                ev_id = ev_id_str
            rl_schedule.append((ev_id, t_idx, interval[2], interval[3], interval[4]))
    return rl_schedule


#####################
def calculate_solution_cost(schedule, prices, dt):
    """
    Calcula el costo total de una solución.
    
    Args:
        schedule: Diccionario con la programación {ev_id: [(t_start, t_end, charger, slot, power),...]}
        prices: Lista de precios por intervalo
        dt: Delta de tiempo (horas)
        
    Returns:
        float: Costo total de la solución
    """
    total_cost = 0
    times_dict = {t: p for t, p in enumerate(prices)}
    
    for ev_id, intervals in schedule.items():
        for interval in intervals:
            t_start, t_end, _, _, power = interval
            # Encontrar el índice de tiempo correspondiente
            t_idx = None
            for idx, t in enumerate(times_dict):
                if abs(t - t_start) < 1e-5:
                    t_idx = idx
                    break
            
            if t_idx is not None:
                energy = power * dt
                price = prices[t_idx]
                total_cost += energy * price
    
    return total_cost

def assess_infeasibility_from_zero_energy(milp_data):
    """
    Evalúa si el modelo original era infactible basado en vehículos 
    que recibieron EXACTAMENTE 0 energía.
    
    Args:
        milp_data: Datos de solución MILP
        
    Returns:
        dict: Información de factibilidad
    """
    # Extraer información extra si está disponible
    extra_info_milp = milp_data.get("extra_info", {})
    rejected_details = extra_info_milp.get("rejected_details", {})
    
    # Filtrar solo vehículos con entrega de energía exactamente igual a 0
    zero_energy_evs = [
        ev_id for ev_id, details in rejected_details.items()
        if details.get("delivered_energy", 0) == 0
    ]
    
    # El modelo es infactible solo si hay vehículos que no recibieron NADA de energía
    model_infeasible = len(zero_energy_evs) > 0
    
    # También verificamos señales explícitas de infactibilidad
    explicit_infeasible = extra_info_milp.get("original_model_infeasible", False)
    
    return {
        "modelo_original_infactible": model_infeasible or explicit_infeasible,
        "razon_infactibilidad": "Vehículos con 0 energía" if len(zero_energy_evs) > 0 else 
                              "Explícitamente marcado como infactible" if explicit_infeasible else
                              "Ninguna (modelo factible)",
        "num_vehiculos_cero_energia": len(zero_energy_evs),
        "vehiculos_cero_energia": zero_energy_evs,
        "num_vehiculos_energia_parcial": len(rejected_details) - len(zero_energy_evs)
    }

def generate_results_table_zero_energy_focus(results_dir, data_dir):
    """
    Genera una tabla de resultados enfocada en detectar vehículos que recibieron 0 energía.
    
    Args:
        results_dir: Directorio con los resultados
        data_dir: Directorio con los datos originales
        
    Returns:
        DataFrame: Tabla con los resultados
    """
    import pandas as pd
    import os
    import json
    import glob
    
    # Buscar todas las soluciones MILP disponibles
    milp_files = glob.glob(os.path.join(results_dir, "resultados_milp_instancia_*.json"))
    
    # Crear lista para almacenar resultados
    results = []
    
    for milp_file in milp_files:
        # Extraer número de instancia del nombre de archivo
        filename = os.path.basename(milp_file)
        instance_num = filename.replace("resultados_milp_instancia_", "").replace(".json", "")
        
        # Ruta al archivo RL correspondiente
        rl_file = os.path.join(results_dir, f"resultados_rl_instancia_{instance_num}.json")
        
        # Cargar datos
        try:
            with open(milp_file, 'r') as f:
                milp_data = json.load(f)
            
            # Cargar datos RL si están disponibles
            rl_data = None
            if os.path.exists(rl_file):
                with open(rl_file, 'r') as f:
                    rl_data = json.load(f)
            
            # Evaluar infactibilidad basada en entregas de 0 energía
            infeasibility_info = assess_infeasibility_from_zero_energy(milp_data)
            
            # Extraer información de capacidad
            capacity_info = get_parking_capacity(instance_num, data_dir)
            
            # Extraer información de rechazos y energía
            extra_info_milp = milp_data.get("extra_info", {})
            rejected_details = extra_info_milp.get("rejected_details", {})
            
            # Calcular energía requerida y entregada totales
            total_required_energy = sum(ev_info.get("required_energy", 0) for ev_info in rejected_details.values())
            total_delivered_energy = sum(ev_info.get("delivered_energy", 0) for ev_info in rejected_details.values())
            total_unmet_energy = sum(ev_info.get("unmet_energy", 0) for ev_info in rejected_details.values())
            
            # Verificar si hay algún vehículo con 0 energía entregada
            has_zero_energy = any(ev_info.get("delivered_energy", 0) == 0 for ev_info in rejected_details.values())
            
            # Agregar a resultados
            results.append({
                "instancia": instance_num,
                "modelo_original_infactible": infeasibility_info["modelo_original_infactible"],
                "razon_infactibilidad": infeasibility_info["razon_infactibilidad"],
                "num_vehiculos": capacity_info["n_evs"],
                "num_spots": capacity_info["n_spots"],
                "ratio_ev_spots": capacity_info["n_evs"] / max(1, capacity_info["n_spots"]),
                "vehiculos_energia_cero": infeasibility_info["num_vehiculos_cero_energia"],
                "vehiculos_energia_parcial": infeasibility_info["num_vehiculos_energia_parcial"],
                "vehiculos_con_rechazos": len(rejected_details),
                "energia_requerida": total_required_energy,
                "energia_entregada": total_delivered_energy,
                "energia_no_satisfecha": total_unmet_energy,
                "porcentaje_energia_satisfecha": (total_delivered_energy / total_required_energy * 100) if total_required_energy > 0 else 100
            })
            
        except Exception as e:
            print(f"Error procesando instancia {instance_num}: {e}")
            continue
    
    # Convertir a DataFrame
    df = pd.DataFrame(results)
    
    # Ordenar por número de instancia
    df = df.sort_values("instancia")
    
    return df

def get_parking_capacity(test_number, data_dir="./data"):
    """
    Obtiene la capacidad de parqueadero para un sistema de prueba específico.
    
    Args:
        test_number: Número del sistema de prueba
        data_dir: Directorio de datos
        
    Returns:
        dict: Información de capacidad del sistema
    """
    import os
    import json
    
    config_file = os.path.join(data_dir, f"test_system_{test_number}.json")
    if not os.path.exists(config_file):
        return {"n_spots": 0, "n_chargers": 0, "station_limit": 0, "n_evs": 0}
    
    try:
        with open(config_file, 'r') as f:
            data = json.load(f)
        
        parking_config = data.get("parking_config", {})
        return {
            "n_spots": parking_config.get("n_spots", 0),
            "n_chargers": len(parking_config.get("chargers", [])),
            "station_limit": parking_config.get("transformer_limit", 0),
            "n_evs": len(data.get("arrivals", []))
        }
    except Exception as e:
        print(f"Error al leer configuración: {e}")
        return {"n_spots": 0, "n_chargers": 0, "station_limit": 0, "n_evs": 0}


def run_mode_6_zero_energy_focus(results_dir, data_dir):
    """
    Ejecuta el modo 6 con enfoque en detectar infactibilidad basada en vehículos con 0 energía
    
    Args:
        results_dir: Directorio con los resultados
        data_dir: Directorio con los datos originales
    """
    import pandas as pd
    import os
    
    print("\nMODO 6: ANÁLISIS DE INFACTIBILIDAD POR VEHÍCULOS CON 0 ENERGÍA")
    print("-----------------------------------------------------------")
    
    if not os.path.exists(results_dir):
        print(f"El directorio de resultados {results_dir} no existe.")
        return
    
    print("Analizando resultados para determinar infactibilidad basada en entregas de 0 energía...")
    
    # Generar tabla de resultados
    results_df = generate_results_table_zero_energy_focus(results_dir, data_dir)
    
    if results_df.empty:
        print("No se encontraron resultados para analizar.")
        return
    
    # Contar instancias factibles e infactibles
    num_infactible = results_df['modelo_original_infactible'].sum()
    num_factible = len(results_df) - num_infactible
    
    # Mostrar resumen
    print(f"\nSe analizaron {len(results_df)} instancias:")
    print(f"- Modelo original factible: {num_factible} ({num_factible/len(results_df)*100:.1f}%)")
    print(f"- Modelo original infactible (vehículos con 0 energía): {num_infactible} ({num_infactible/len(results_df)*100:.1f}%)")
    
    # Guardar tabla en formato CSV
    output_file = os.path.join(results_dir, "analisis_infactibilidad_zero_energy.csv")
    results_df.to_csv(output_file, index=False)
    print(f"\nAnálisis guardado en: {output_file}")
    
    # Intentar guardar en Excel
    try:
        excel_file = os.path.join(results_dir, "analisis_infactibilidad_zero_energy.xlsx")
        results_df.to_excel(excel_file, index=False)
        print(f"Análisis guardado en formato Excel: {excel_file}")
    except Exception as e:
        pass
    
    # Mostrar tabla en pantalla
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print("\nANÁLISIS DE INFACTIBILIDAD POR VEHÍCULOS CON 0 ENERGÍA:")
    print(results_df)
    
    # Mostrar casos infactibles con más detalle
    if num_infactible > 0:
        print("\nDETALLE DE INSTANCIAS INFACTIBLES (VEHÍCULOS CON 0 ENERGÍA):")
        infactible_df = results_df[results_df['modelo_original_infactible']].sort_values('vehiculos_energia_cero', ascending=False)
        for _, row in infactible_df.iterrows():
            print(f"Instancia {row['instancia']}:")
            print(f"  Vehículos totales: {row['num_vehiculos']}, Spots: {row['num_spots']}, Ratio: {row['ratio_ev_spots']:.2f}")
            print(f"  Vehículos con 0 energía: {row['vehiculos_energia_cero']}")
            print(f"  Vehículos con energía parcial: {row['vehiculos_energia_parcial']}")
            print(f"  Energía requerida total: {row['energia_requerida']:.2f} kWh")
            print(f"  Energía entregada: {row['energia_entregada']:.2f} kWh ({row['porcentaje_energia_satisfecha']:.1f}%)")
            print()
    
    # También mostrar casos con entregas parciales pero sin infactibilidad
    partial_cases = results_df[(~results_df['modelo_original_infactible']) & (results_df['vehiculos_energia_parcial'] > 0)]
    if len(partial_cases) > 0:
        print("\nINSTANCIAS CON ENTREGAS PARCIALES (SIN INFACTIBILIDAD):")
        for _, row in partial_cases.iterrows():
            print(f"Instancia {row['instancia']}:")
            print(f"  Vehículos con entregas parciales: {row['vehiculos_energia_parcial']}")
            print(f"  Porcentaje de energía satisfecha: {row['porcentaje_energia_satisfecha']:.1f}%")
            print()

if __name__ == "__main__":
    print("Sistema de optimización de carga de vehículos eléctricos")
    print("--------------------------------------------------------")
    
    # Determinar el modo de operación
    print("\nModos disponibles:")
    print("1. Entrenamiento generalizado (crear modelo para todos los sistemas)")
    print("2. Solución para una instancia específica (usando modelo generalizado)")
    print("3. Verificar progreso de entrenamiento y checkpoints")
    print("4. Procesar TODAS las instancias automáticamente (RL y MILP)")
    print("5. Visualizar solución MILP desde archivo")
    print("6. Generar tabla de resultados comparativa (solo JSON)")
    
    mode = input("Seleccione modo (1-4): ")
    
    # Definir rutas básicas
    model_path = "ev_scheduler_generalized_model.h5"
    data_dir = "./data"
    checkpoint_dir = "./checkpoints"
    results_dir = "./results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    if mode == "1":
        # MODO 1: ENTRENAMIENTO GENERALIZADO
        print("\nMODO: ENTRENAMIENTO GENERALIZADO")
        print("--------------------------------")
        
        print("Cargando todos los sistemas disponibles...")
        all_systems = load_all_test_systems(data_dir)
        if not all_systems:
            print("No se encontraron sistemas para entrenar. Verifique la carpeta de datos.")
            exit(1)
        print(f"Se cargaron {len(all_systems)} sistemas.")
        episodes = int(input("Número de episodios por sistema (recomendado: 30-50): "))
        checkpoint_frequency = int(input("Guardar checkpoint cada cuántos episodios (recomendado: 5): "))
        resume = input("¿Desea reanudar desde el último checkpoint? (s/n): ").lower() == 's'
        
        start_time = time.time()
        agent = train_generalized_agent(
            all_systems, 
            episodes_per_system=episodes, 
            model_path=model_path,
            checkpoint_dir=checkpoint_dir,
            checkpoint_frequency=checkpoint_frequency,
            resume_from_checkpoint=resume
        )
        train_time = time.time() - start_time
        
        print(f"\nEntrenamiento completado en {train_time:.2f} segundos.")
        print(f"Modelo generalizado guardado en: {model_path}")
        print(f"Checkpoints guardados en: {checkpoint_dir}")
    
    elif mode == "2":
        # MODO 2: SOLUCIÓN PARA UNA INSTANCIA
        print("\nMODO: SOLUCIÓN PARA UNA INSTANCIA")
        print("--------------------------------")
        
        json_file = input("Introduzca el nombre del archivo JSON (ej: test_system_1.json): ")
        config = None
        try_paths = [
            json_file,
            f"{data_dir}/{json_file}",
            f"{data_dir}/test_system_{json_file}.json" if json_file.isdigit() else None
        ]
        for path in try_paths:
            if path is None:
                continue
            try:
                config = load_data(path)
                print(f"Sistema cargado exitosamente desde {path}")
                break
            except Exception as e:
                print(f"No se pudo cargar desde {path}: {e}")
        if config is None:
            print("No se pudo cargar el sistema. Verifique la ruta y el formato.")
            exit(1)
        
        print(f"\nInformación del sistema:")
        print(f"- Número de vehículos: {len(config['arrivals'])}")
        print(f"- Plazas de estacionamiento: {config['n_spots']}")
        print(f"- Cargadores: {len(config['chargers'])}")
        print(f"- Límite del transformador: {config['station_limit']} kW")
        has_brand_info = any('brand' in ev for ev in config['arrivals'])
        has_battery_info = any('battery_capacity' in ev for ev in config['arrivals'])
        if has_brand_info or has_battery_info:
            print("- Información adicional detectada:")
            if has_brand_info:
                print("  * Marcas de vehículos")
            if has_battery_info:
                print("  * Capacidades de batería")
        
        # 1. Generar solución RL
        print("\n1. GENERACIÓN DE SOLUCIÓN RL")
        print("----------------------------")
        state_size = 24
        action_size = 150
        agent = GeneralizedDQNAgent(state_size, action_size)
        model_loaded = agent.load(model_path)
        if not model_loaded:
            print(f"No se pudo cargar el modelo generalizado desde {model_path}.")
            print("Se recomienda entrenar primero un modelo generalizado (Modo 1).")
            print("Generando solución con un agente básico...")
        else:
            print(f"Modelo generalizado cargado desde {model_path}.")
        
        start_time_rl = time.time()
        rl_schedule = generate_rl_solution(config, agent, model_path=model_path)
        rl_time = time.time() - start_time_rl
        print(f"Solución RL generada en {rl_time:.2f} segundos.")
        instance_id = config.get("test_number", "unknown")
        filename_rl = os.path.join(results_dir, f"resultados_rl_instancia_{instance_id}.json")
        rl_schedule_dict = convert_rl_schedule_to_dict(rl_schedule, config["times"], config["dt"])
        # Guardamos también el tiempo de ejecución del RL en el extra_info
        extra_info_rl = {"rl_time": rl_time}
        save_schedule_to_json(rl_schedule_dict, extra_info=extra_info_rl, file_path=filename_rl)
        
        # 2. Refinamiento con MILP
        print("\n2. REFINAMIENTO CON MILP")
        time_limit = 600 if len(config['arrivals']) > 200 else 300
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            temp_json = f.name
            json.dump({
                "test_number": config.get("test_number", 0),
                "energy_prices": [{"time": t, "price": p} for t, p in zip(config["times"], config["prices"])],
                "arrivals": config["arrivals"],
                "parking_config": {
                    "n_spots": config["n_spots"],
                    "chargers": config["chargers"],
                    "transformer_limit": config["station_limit"]
                }
            }, f)
        print("Refinando solución con MILP (esto puede tardar)...")
        start_time_milp = time.time()
        model, milp_schedule, rejected_details = solve_ev_schedule(temp_json, penalty_unmet=1000.0, rl_schedule=rl_schedule, time_limit=time_limit)
        milp_time = time.time() - start_time_milp
        try:
            os.remove(temp_json)
        except:
            pass
        if model is not None and milp_schedule:
            filename_milp = os.path.join(results_dir, f"resultados_milp_instancia_{instance_id}.json")
            # Extraemos también tiempos y sumamos total
            extra_info_milp = {
                "rejected_details": rejected_details,
                "milp_time": milp_time,
                "total_time": rl_time + milp_time
            }
            save_schedule_to_json(milp_schedule, extra_info=extra_info_milp, file_path=filename_milp)
            print(f"Solución MILP y reporte guardados en {filename_milp}.")
        else:
            print("MILP no generó una solución mejorada. Se mantendrá la solución RL como resultado final.")
    
    elif mode == "3":
        # MODO 3: VERIFICAR PROGRESO DE ENTRENAMIENTO
        print("\nMODO: VERIFICAR PROGRESO DE ENTRENAMIENTO")
        print("----------------------------------------")
        progress_info = check_training_progress(checkpoint_dir)
        if progress_info:
            if input("\n¿Desea visualizar el progreso gráficamente? (s/n): ").lower() == 's':
                plot_training_progress(checkpoint_dir)
            if input("\n¿Desea eliminar checkpoints antiguos para liberar espacio? (s/n): ").lower() == 's':
                checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_sys_*_ep_*.h5"))
                checkpoint_files.sort(key=os.path.getmtime)
                if len(checkpoint_files) <= 5:
                    print("No hay suficientes checkpoints para eliminar (menos de 5).")
                else:
                    to_keep = set(checkpoint_files[-3:])
                    systems = {}
                    for cp in checkpoint_files:
                        filename = os.path.basename(cp)
                        try:
                            parts = filename.split('_')
                            system_idx = int(parts[2])
                            if system_idx not in systems:
                                systems[system_idx] = cp
                                to_keep.add(cp)
                        except:
                            pass
                    deleted_count = 0
                    freed_space = 0
                    for cp in checkpoint_files:
                        if cp not in to_keep:
                            try:
                                size = os.path.getsize(cp) / (1024 * 1024)
                                os.remove(cp)
                                deleted_count += 1
                                freed_space += size
                                print(f"Eliminado: {os.path.basename(cp)} ({size:.2f} MB)")
                            except Exception as e:
                                print(f"Error al eliminar {cp}: {e}")
                    print(f"\nSe eliminaron {deleted_count} checkpoints antiguos, liberando {freed_space:.2f} MB")
            print("\nOperación completada.")
        else:
            print("No se encontró información de checkpoints.")
        print("\n¡Proceso completado con éxito!")
    
    elif mode == "4":
        # MODO 4: PROCESAR TODAS LAS INSTANCIAS AUTOMÁTICAMENTE
        print("\nMODO: PROCESAR TODAS LAS INSTANCIAS AUTOMÁTICAMENTE")
        print("------------------------------------------------------")
        print("Cargando todos los sistemas disponibles...")
        all_systems = load_all_test_systems(data_dir)
        if not all_systems:
            print("No se encontraron sistemas para procesar. Verifique la carpeta de datos.")
            exit(1)
        print(f"Se encontraron {len(all_systems)} sistemas.")
        
        state_size = 24
        action_size = 150
        agent = GeneralizedDQNAgent(state_size, action_size)
        if not agent.load(model_path):
            print(f"No se pudo cargar el modelo generalizado desde {model_path}.")
            print("Se recomienda entrenar primero un modelo generalizado (Modo 1).")
            exit(1)
        
        for test_number, config in all_systems.items():
            print(f"\nProcesando instancia con test_number: {test_number}")
            
            filename_rl = os.path.join(results_dir, f"resultados_rl_instancia_{test_number}.json")
            filename_milp = os.path.join(results_dir, f"resultados_milp_instancia_{test_number}.json")
            
            # Si ya existe la solución MILP, saltamos la instancia.
            if os.path.exists(filename_milp):
                print(f"Instancia {test_number} ya procesada (MILP existe), saltando.")
                continue
            
            # Variable para almacenar el tiempo de la solución RL
            rl_time_val = 0
            if os.path.exists(filename_rl):
                print(f"Instancia {test_number}: Se encontró solución RL guardada. Cargando...")
                rl_schedule = load_rl_schedule_from_json(filename_rl, config)
                # Intentamos cargar el tiempo de ejecución RL desde el JSON si existe.
                try:
                    data_rl = json.load(open(filename_rl))
                    if "extra_info" in data_rl and "rl_time" in data_rl["extra_info"]:
                        rl_time_val = data_rl["extra_info"]["rl_time"]
                except Exception as e:
                    print("No se pudo leer el tiempo de RL guardado, se asigna 0.", e)
            else:
                print(f"Instancia {test_number}: No se encontró solución RL guardada, generando RL...")
                start_time_rl = time.time()
                rl_schedule = generate_rl_solution(config, agent, model_path=model_path)
                rl_time_val = time.time() - start_time_rl
                rl_schedule_dict = convert_rl_schedule_to_dict(rl_schedule, config["times"], config["dt"])
                extra_info_rl = {"rl_time": rl_time_val}
                save_schedule_to_json(rl_schedule_dict, extra_info=extra_info_rl, file_path=filename_rl)
                print(f"Instancia {test_number}: Solución RL generada en {rl_time_val:.2f} segundos y guardada en {filename_rl}.")
            
            # Refinamiento con MILP
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
                temp_json = f.name
                json.dump({
                    "test_number": config.get("test_number", 0),
                    "energy_prices": [{"time": t, "price": p} for t, p in zip(config["times"], config["prices"])],
                    "arrivals": config["arrivals"],
                    "parking_config": {
                        "n_spots": config["n_spots"],
                        "chargers": config["chargers"],
                        "transformer_limit": config["station_limit"]
                    }
                }, f)
            print(f"Instancia {test_number}: Refinando solución con MILP (esto puede tardar)...")
            time_limit_val = 600 if len(config['arrivals']) > 200 else 300
            start_time_milp = time.time()
            model, milp_schedule, rejected_details = solve_ev_schedule(temp_json, penalty_unmet=1000.0, rl_schedule=rl_schedule, time_limit=time_limit_val)
            milp_time = time.time() - start_time_milp
            try:
                os.remove(temp_json)
            except:
                pass
            if model is not None and milp_schedule:
                extra_info_milp = {
                    "rejected_details": rejected_details,
                    "milp_time": milp_time,
                    "total_time": milp_time + rl_time_val
                }
                save_schedule_to_json(milp_schedule, extra_info=extra_info_milp, file_path=filename_milp)
                print(f"Instancia {test_number}: Solución MILP y reporte guardados en {filename_milp}.")
            else:
                print(f"Instancia {test_number}: MILP no generó una solución mejorada. Se mantendrá la solución RL como resultado final.")
        print("\nProcesamiento de todas las instancias completado.")

    elif mode == "5":
        # Opción 5: Visualizar solución MILP desde archivo
        instance_id = input("Ingrese el número de instancia (ej: 10): ").strip()
        solution_file = os.path.join(results_dir, f"resultados_milp_instancia_{instance_id}.json")
        config_file = os.path.join(data_dir, f"test_system_{instance_id}.json")
        if not os.path.exists(solution_file):
            print(f"No se encontró el archivo de solución MILP: {solution_file}")
        elif not os.path.exists(config_file):
            print(f"No se encontró el archivo de configuración: {config_file}")
        else:
            visualize_solution_from_file(solution_file, config_file)
    elif mode =="6":

        run_mode_6_zero_energy_focus(results_dir, data_dir)
    else:
        print("Modo no válido. Saliendo.")
        exit(1)
