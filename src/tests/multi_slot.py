class MultiSlotPredictor:
    """Usa el modelo existente para predecir múltiples slots por timestep"""
    
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.max_decisions_per_timestep = 10  # Límite de seguridad
        
    def predict_multiple_slots(self, state):
        """Predice para múltiples slots usando el modelo existente"""
        
        decisions_made = []
        current_state = state.copy()
        
        for decision_round in range(self.max_decisions_per_timestep):
            
            # 1. Obtener acciones posibles del estado actual
            possible_actions = self.env._get_possible_actions(current_state)
            
            if not possible_actions or len(possible_actions) <= 1:  # Solo no_action
                break
                
            # 2. Usar el agente para predecir la mejor acción
            action_idx = self.agent.act(current_state, possible_actions, verbose=False)
            
            if action_idx == -1 or action_idx >= len(possible_actions):
                break
                
            selected_action = possible_actions[action_idx]
            
            # 3. Si es no_action, terminar
            if selected_action.get("type") == "no_action" or selected_action.get("skip", False):
                break
                
            # 4. Simular la acción en el estado (SIN avanzar tiempo)
            updated_state = self._simulate_action_on_state(current_state, selected_action)
            
            if updated_state is None:  # Acción inválida
                break
                
            # 5. Guardar decisión y actualizar estado
            decisions_made.append(selected_action)
            current_state = updated_state
            
            # 6. Verificar si ya no hay más vehículos o spots disponibles
            if not current_state.get("evs_present") or not current_state.get("available_spots"):
                break
                
        print(f"Decisiones tomadas en este timestep: {len(decisions_made)}")
        return decisions_made
    
    def _simulate_action_on_state(self, state, action):
        """Simula una acción en el estado sin modificar el environment"""
        
        updated_state = state.copy()
        
        if action["type"] == "assign_parking":
            ev_id = action["ev_id"]
            spot = action["spot"]
            
            # Verificar que la acción es válida
            if ev_id not in updated_state["evs_present"]:
                return None
            if spot not in updated_state["available_spots"]:
                return None
                
            # Actualizar estado
            updated_state["available_spots"].remove(spot)
            updated_state["occupied_spots"].append(spot)
            
            # Marcar vehículo como asignado
            if ev_id in updated_state["evs_present"]:
                updated_state["evs_present"].remove(ev_id)
                
            # Actualizar detalles del vehículo
            if "ev_details" in updated_state and ev_id in updated_state["ev_details"]:
                updated_state["ev_details"][ev_id]["current_spot"] = spot
                
        elif action["type"] == "assign_charging":
            ev_id = action["ev_id"]
            spot = action["spot"]
            charger = action["charger"]
            
            # Verificaciones
            if ev_id not in updated_state["evs_present"]:
                return None
            if spot not in updated_state["available_spots"]:
                return None
            if charger not in updated_state["available_chargers"]:
                return None
                
            # Actualizar estado
            updated_state["available_spots"].remove(spot)
            updated_state["available_chargers"].remove(charger)
            updated_state["occupied_spots"].append(spot)
            updated_state["occupied_chargers"].append(charger)
            
            # Marcar vehículo como asignado
            if ev_id in updated_state["evs_present"]:
                updated_state["evs_present"].remove(ev_id)
                
            # Actualizar detalles del vehículo
            if "ev_details" in updated_state and ev_id in updated_state["ev_details"]:
                updated_state["ev_details"][ev_id]["current_spot"] = spot
                updated_state["ev_details"][ev_id]["is_charging"] = True
                
        elif action["type"] == "start_charging":
            ev_id = action["ev_id"]
            charger = action["charger"]
            
            # Verificar que el vehículo ya tiene spot
            if "ev_details" not in updated_state or ev_id not in updated_state["ev_details"]:
                return None
            if updated_state["ev_details"][ev_id].get("current_spot") is None:
                return None
            if charger not in updated_state["available_chargers"]:
                return None
                
            # Actualizar estado
            updated_state["available_chargers"].remove(charger)
            updated_state["occupied_chargers"].append(charger)
            updated_state["ev_details"][ev_id]["is_charging"] = True
            
        # Otros tipos de acciones...
        
        return updated_state


class EnhancedEnvironmentWrapper:
    """Wrapper que usa el MultiSlotPredictor"""
    
    def __init__(self, original_env, agent):
        self.env = original_env
        self.predictor = MultiSlotPredictor(agent, original_env)
        
    def step_with_multiple_decisions(self):
        """Step que toma múltiples decisiones por timestep"""
        
        # 1. Obtener estado actual
        state = self.env._get_state()
        if state is None:
            return None, 0, True
            
        # 2. Predecir múltiples acciones
        decisions = self.predictor.predict_multiple_slots(state)
        
        # 3. Ejecutar todas las decisiones en el environment real
        total_reward = 0
        
        for decision in decisions:
            # Convertir decisión a action_idx
            possible_actions = self.env._get_possible_actions(self.env._get_state())
            
            # Encontrar índice de la acción
            action_idx = -1
            for i, possible_action in enumerate(possible_actions):
                if self._actions_match(decision, possible_action):
                    action_idx = i
                    break
                    
            if action_idx != -1:
                next_state, reward, done = self.env.step(action_idx)
                total_reward += reward
                
                if done:
                    break
            else:
                # Si no encuentra la acción, hacer no_action
                next_state, reward, done = self.env.step(0)  # Asumiendo que 0 es no_action
                
        # 4. Avanzar tiempo UNA vez (no por cada decisión)
        final_state = self.env._get_state()
        is_done = self.env.current_time_idx >= len(self.env.times)
        
        return final_state, total_reward, is_done
    
    def _actions_match(self, decision, possible_action):
        """Verifica si dos acciones son equivalentes"""
        
        if decision.get("type") != possible_action.get("type"):
            return False
            
        # Comparar campos relevantes según el tipo
        if decision["type"] in ["assign_parking", "assign_charging"]:
            return (decision.get("ev_id") == possible_action.get("ev_id") and
                    decision.get("spot") == possible_action.get("spot"))
                    
        elif decision["type"] == "start_charging":
            return (decision.get("ev_id") == possible_action.get("ev_id") and
                    decision.get("charger") == possible_action.get("charger"))
                    
        return True