import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, defaultdict
import pickle
import os

class DQNNetwork(nn.Module):
    """
    Red neuronal DQN con arquitectura Dueling implementada en PyTorch.
    """
    def __init__(self, state_size, action_size, dueling=True):
        super(DQNNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.dueling = dueling
        
        # Capas compartidas
        self.fc1 = nn.Linear(state_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 32)
        
        if self.dueling:
            # Dueling DQN: separar value y advantage streams
            self.value_stream = nn.Linear(32, 16)
            self.value_output = nn.Linear(16, 1)
            
            self.advantage_stream = nn.Linear(32, 16)
            self.advantage_output = nn.Linear(16, action_size)
        else:
            # DQN simple
            self.output = nn.Linear(32, action_size)
    
    def forward(self, x):
        # Asegurar que x tenga la forma correcta
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        # Capas compartidas
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        
        if self.dueling:
            # Dueling DQN
            value = F.relu(self.value_stream(x))
            value = self.value_output(value)
            
            advantage = F.relu(self.advantage_stream(x))
            advantage = self.advantage_output(advantage)
            
            # Combinar value y advantage
            # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
            
            return q_values
        else:
            # DQN simple
            return self.output(x)

class EnhancedDQNAgentPyTorch:
    """
    Agente DQN mejorado implementado en PyTorch con soporte automático para GPU.
    """
    def __init__(self, state_size, action_size, learning_rate=0.0005, 
                 gamma=0.95, epsilon=0.9, epsilon_min=0.05, epsilon_decay=0.99,
                 memory_size=5000, batch_size=32, target_update_freq=50,
                 dueling_network=True):
        
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.dueling_network = dueling_network
        self.target_update_freq = target_update_freq
        self.target_update_counter = 0
        self.steps = 0
        
        # Detectar dispositivo (GPU/CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DQN Agent usando: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
            print(f"Memoria GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Crear redes neuronales
        self.q_network = DQNNetwork(state_size, action_size, dueling_network).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size, dueling_network).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Inicializar target network
        self.update_target_network()
        
        # Memoria específica por sistema
        self.system_memories = defaultdict(lambda: deque(maxlen=2000))
        
        print(f"Red neuronal creada con {sum(p.numel() for p in self.q_network.parameters())} parámetros")
    
    def update_target_network(self):
        """Actualiza el target network copiando pesos del main network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done, system_type=0):
        """Almacena experiencia en memoria."""
        try:
            action = int(action)
            reward = float(reward)
            done = bool(done)
            system_type = int(system_type)
        except (TypeError, ValueError):
            return
            
        if not (0 <= action < self.action_size):
            return
            
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)
        self.system_memories[system_type].append(experience)
    
    def act(self, state, possible_actions, verbose=False):
        """Selecciona acción usando epsilon-greedy."""
        if len(possible_actions) == 0:
            if verbose:
                print("      No hay acciones posibles")
            return -1
        
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(len(possible_actions))
            if verbose:
                print(f"      Acción aleatoria: {action} (epsilon: {self.epsilon:.3f})")
            return action
        
        try:
            state_vector = self._process_state(state)
            
            # Convertir a tensor y mover a dispositivo
            state_tensor = torch.FloatTensor(state_vector).to(self.device)
            
            # Modo evaluación para predicción
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            
            # Filtrar solo acciones posibles
            q_values_np = q_values.cpu().numpy().flatten()
            filtered_actions = [(i, q_values_np[i]) for i in range(min(len(q_values_np), len(possible_actions)))]
            
            if not filtered_actions:
                action = np.random.choice(len(possible_actions))
                if verbose:
                    print(f"      Fallback aleatorio: {action}")
                return action
            
            action = max(filtered_actions, key=lambda x: x[1])[0]
            if verbose:
                q_value = filtered_actions[action][1]
                print(f"      Acción DQN: {action} (Q-value: {q_value:.3f})")
            return action
            
        except Exception as e:
            if verbose:
                print(f"      Error en act(): {e}")
            return np.random.choice(len(possible_actions))
    
    def _safe_scalar_from_list(self, value, default=0.0):
        """Convierte valor a escalar de manera segura."""
        if value is None:
            return default
        
        if isinstance(value, (list, np.ndarray)):
            if len(value) == 0:
                return default
            try:
                arr = np.array(value)
                if arr.size == 0:
                    return default
                return float(np.mean(arr))
            except (TypeError, ValueError):
                return default
        
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    
    def _process_state(self, state):
        """Procesa el estado para la red neuronal."""
        if state is None:
            return np.zeros(self.state_size, dtype=np.float32)
        
        base_vector = []
        
        try:
            # Características del vehículo
            ev_features = state.get("ev_features", [])
            if isinstance(ev_features, list) and len(ev_features) > 0:
                for i in range(7):
                    if i < len(ev_features):
                        val = self._safe_scalar_from_list(ev_features[i])
                        base_vector.append(val)
                    else:
                        base_vector.append(0.0)
            else:
                base_vector.extend([0.0] * 7)
            
            # Características del sistema
            system_features = [
                self._safe_scalar_from_list(state.get("avg_available_spots", 0.5)),
                self._safe_scalar_from_list(state.get("avg_available_chargers", 0.5)),
                self._safe_scalar_from_list(state.get("min_price", 0.5)),
                self._safe_scalar_from_list(state.get("avg_price", 0.5)),
                self._safe_scalar_from_list(state.get("min_transformer_capacity", 0.5)),
                self._safe_scalar_from_list(state.get("avg_transformer_capacity", 0.5)),
                self._safe_scalar_from_list(state.get("system_type", 0)) / 20.0,
                self._safe_scalar_from_list(state.get("n_spots_total", 10)) / 100.0,
                self._safe_scalar_from_list(state.get("n_chargers_total", 5)) / 50.0,
                self._safe_scalar_from_list(state.get("transformer_limit", 50)) / 200.0,
                self._safe_scalar_from_list(state.get("max_charger_power", 10)) / 100.0
            ]
            base_vector.extend(system_features)
            
            # Características adicionales
            additional_features = [
                "battery_capacity", "charging_urgency", "system_demand_ratio", 
                "time_remaining", "min_charge_rate", "max_charge_rate", 
                "priority", "willingness_to_pay", "efficiency", "compatible_chargers_ratio"
            ]
            
            for feature in additional_features:
                if feature in state:
                    if feature == "battery_capacity":
                        base_vector.append(self._safe_scalar_from_list(state[feature]) / 100.0)
                    elif feature in ["min_charge_rate", "ac_charge_rate"]:
                        base_vector.append(self._safe_scalar_from_list(state[feature]) / 50.0)
                    elif feature in ["max_charge_rate", "dc_charge_rate"]:
                        base_vector.append(self._safe_scalar_from_list(state[feature]) / 350.0)
                    elif feature == "priority":
                        base_vector.append(self._safe_scalar_from_list(state[feature]) / 3.0)
                    elif feature == "willingness_to_pay":
                        base_vector.append(self._safe_scalar_from_list(state[feature]) / 1.5)
                    else:
                        base_vector.append(self._safe_scalar_from_list(state[feature]))
            
            # Características de listas
            list_features = ["spot_availability", "charger_availability", "relevant_prices", "transformer_capacity"]
            for feature in list_features:
                if feature in state:
                    base_vector.append(self._safe_scalar_from_list(state[feature]))
            
            # Asegurar que todos son escalares
            base_vector = [self._safe_scalar_from_list(x) for x in base_vector]
            
            # Convertir a numpy array
            all_features = np.array(base_vector, dtype=np.float32)
            
            # Asegurar dimensión correcta
            if len(all_features) < self.state_size:
                padding = np.zeros(self.state_size - len(all_features), dtype=np.float32)
                all_features = np.concatenate([all_features, padding])
            elif len(all_features) > self.state_size:
                all_features = all_features[:self.state_size]
            
            # Limpiar NaN e infinitos
            all_features = np.nan_to_num(all_features, nan=0.0, posinf=1.0, neginf=0.0)
            
            return all_features
            
        except Exception as e:
            print(f"Error en _process_state: {e}")
            return np.zeros(self.state_size, dtype=np.float32)
    
    def replay(self, system_specific=False, system_type=0):
        """Entrena la red con experiencia pasada."""
        try:
            # Seleccionar memoria
            if system_specific and len(self.system_memories[system_type]) >= self.batch_size:
                samples = random.sample(self.system_memories[system_type], self.batch_size)
            else:
                if len(self.memory) < self.batch_size:
                    return
                samples = random.sample(self.memory, self.batch_size)
            
            if not samples:
                return
            
            # Procesar batch
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []
            
            for state, action, reward, next_state, done in samples:
                if not isinstance(action, (int, np.integer)) or action >= self.action_size:
                    continue
                
                try:
                    state_vector = self._process_state(state)
                    next_state_vector = self._process_state(next_state)
                    
                    if state_vector is None or next_state_vector is None:
                        continue
                    
                    states.append(state_vector)
                    actions.append(action)
                    rewards.append(float(reward))
                    next_states.append(next_state_vector)
                    dones.append(done)
                    
                except Exception as e:
                    continue
            
            if len(states) == 0:
                return
            
            # Convertir a tensores
            states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
            actions_tensor = torch.LongTensor(actions).to(self.device)
            rewards_tensor = torch.FloatTensor(rewards).to(self.device)
            next_states_tensor = torch.FloatTensor(np.array(next_states)).to(self.device)
            dones_tensor = torch.BoolTensor(dones).to(self.device)
            
            # Calcular Q-values actuales
            self.q_network.train()
            current_q_values = self.q_network(states_tensor)
            current_q_values = current_q_values.gather(1, actions_tensor.unsqueeze(1))
            
            # Calcular Q-values target (Double DQN)
            with torch.no_grad():
                # Usar main network para seleccionar acciones
                next_q_values_main = self.q_network(next_states_tensor)
                next_actions = next_q_values_main.argmax(1)
                
                # Usar target network para evaluar
                next_q_values_target = self.target_network(next_states_tensor)
                next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1))
                
                # Calcular targets
                target_q_values = rewards_tensor + (self.gamma * next_q_values.squeeze() * ~dones_tensor)
            
            # Calcular loss y optimizar
            loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping para estabilidad
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Decrementar epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Actualizar target network
            self.steps += 1
            self.target_update_counter += 1
            if self.target_update_counter >= self.target_update_freq:
                self.update_target_network()
                self.target_update_counter = 0
                
        except Exception as e:
            print(f"Error en replay: {e}")
    
    def save(self, filepath):
        """Guarda el modelo."""
        try:
            torch.save({
                'q_network_state_dict': self.q_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'steps': self.steps
            }, filepath)
            return True
        except Exception as e:
            print(f"Error al guardar modelo: {e}")
            return False
    
    def load(self, filepath):
        """Carga el modelo."""
        try:
            if not os.path.exists(filepath):
                return False
                
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.steps = checkpoint.get('steps', 0)
            
            return True
        except Exception as e:
            print(f"Error al cargar modelo: {e}")
            return False