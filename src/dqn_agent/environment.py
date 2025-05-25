import numpy as np
from collections import deque, defaultdict
from typing import Dict

class EVChargingEnv:
    """
    Entorno de simulación mejorado para la carga de vehículos eléctricos con RL.
    Incorpora nuevas características y restricciones para sistemas avanzados.
    OPTIMIZADO: Recompensas balanceadas y aprendizaje más eficiente.
    """
    def __init__(self, config):
        """Inicializa el entorno con la configuración del sistema."""
        self.ENERGY_SATISFACTION_WEIGHT = 1.0 # Actualmente es implícito, pero puedes hacerlo explícito si la satisfacción se multiplica por un peso
        self.PENALTY_FOR_SKIPPED_VEHICLE = 50.0 # ¡Este es el que hay que reducir!
        self.ENERGY_COST_WEIGHT = 0.1 # Peso para el costo de energía
        self.REWARD_FOR_ASSIGNED_VEHICLE = 20.0 # ¡Nueva constante! Recompensa explícita por cada vehículo asignado
        self.config = config
        self.process_config()
        self.reset()
        
    
    def process_config(self):
        """Procesa la configuración para extraer parámetros."""
        # Parámetros básicos del sistema
        self.times = self.config["times"]
        self.prices = self.config["prices"]
        self.arrivals = self.config["arrivals"]
        self.chargers = self.config["parking_config"]["chargers"]
        self.station_limit = self.config["parking_config"]["transformer_limit"]
        self.dt = self.config.get("dt", 0.25)  # Default 15 min
        self.n_spots = self.config["parking_config"]["n_spots"]
        self.test_number = self.config.get("test_number", 0)
        
        # Detectar características adicionales disponibles
        self.has_brand_info = any('brand' in ev for ev in self.arrivals)
        self.has_battery_info = any('battery_capacity' in ev for ev in self.arrivals)
        self.has_priority_info = any('priority' in ev for ev in self.arrivals)
        self.has_willingness_info = any('willingness_to_pay' in ev for ev in self.arrivals)
        self.has_efficiency_info = any('efficiency' in ev for ev in self.arrivals)
        self.has_charge_rate_info = any('min_charge_rate' in ev for ev in self.arrivals)
        
        # Normalización de precios
        self.min_price = min(self.prices)
        self.max_price = max(self.prices)
        self.normalized_prices = [(p - self.min_price) / (self.max_price - self.min_price + 1e-6) 
                                 for p in self.prices]
        
        # Cálculo de estadísticas del sistema
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
        
        # Información adicional avanzada
        if self.has_battery_info:
            self.battery_capacity = {arr["id"]: arr.get("battery_capacity", 40) 
                                   for arr in self.arrivals}
        
        if self.has_brand_info:
            self.brands = {arr["id"]: arr.get("brand", "Unknown") for arr in self.arrivals}
            # Mapeo numérico para las marcas
            unique_brands = list(set(self.brands.values()))
            self.brand_to_id = {brand: i/max(1, len(unique_brands)) 
                               for i, brand in enumerate(unique_brands)}
        
        if self.has_priority_info:
            self.priority = {arr["id"]: arr.get("priority", 1) for arr in self.arrivals}
            self.max_priority = max(self.priority.values())
        
        if self.has_willingness_info:
            self.willingness_to_pay = {arr["id"]: arr.get("willingness_to_pay", 1.0) 
                                     for arr in self.arrivals}
            
        if self.has_efficiency_info:
            self.efficiency = {arr["id"]: arr.get("efficiency", 0.9) for arr in self.arrivals}
            
        if self.has_charge_rate_info:
            self.min_charge_rate = {arr["id"]: arr.get("min_charge_rate", 3.5) 
                                   for arr in self.arrivals}
            self.max_charge_rate = {arr["id"]: arr.get("max_charge_rate", 50) 
                                   for arr in self.arrivals}
            self.ac_charge_rate = {arr["id"]: arr.get("ac_charge_rate", 7) 
                                  for arr in self.arrivals}
            self.dc_charge_rate = {arr["id"]: arr.get("dc_charge_rate", 50) 
                                  for arr in self.arrivals}
        
        # Información de cargadores
        self.charger_ids = [c["charger_id"] for c in self.chargers]
        self.max_charger_power_dict = {c["charger_id"]: c["power"] for c in self.chargers}
        
        # Tipos de cargadores y compatibilidad
        self.charger_type = {}
        self.compatible_vehicles = {}
        
        for c in self.chargers:
            charger_id = c["charger_id"]
            self.charger_type[charger_id] = c.get("type", "AC")
            
            if "compatible_vehicles" in c:
                self.compatible_vehicles[charger_id] = c["compatible_vehicles"]
            else:
                # Por defecto compatible con todos
                self.compatible_vehicles[charger_id] = list(set(self.brands.values())) if self.has_brand_info else ["All"]
        
        # Mapeo de compatibilidad EV-Cargador
        self.ev_charger_compatible = {}
        for ev_id in self.ev_ids:
            self.ev_charger_compatible[ev_id] = []
            
            if self.has_brand_info:
                ev_brand = self.brands.get(ev_id, "")
                
                for charger_id in self.charger_ids:
                    # Verificar compatibilidad por marca
                    compatible = True
                    if ev_brand and charger_id in self.compatible_vehicles:
                        # Extraer marca base sin detalles
                        base_brand = ev_brand.split()[0] if " " in ev_brand else ev_brand
                        compatible = False
                        for comp_brand in self.compatible_vehicles[charger_id]:
                            if comp_brand in ev_brand or base_brand in comp_brand:
                                compatible = True
                                break
                    
                    if compatible:
                        # Verificar límites de potencia
                        charger_power = self.max_charger_power_dict[charger_id]
                        charger_type = self.charger_type.get(charger_id, "AC")
                        
                        if self.has_charge_rate_info:
                            # Verificar compatibilidad con tasas de carga
                            if charger_type == "AC" and charger_power <= self.ac_charge_rate.get(ev_id, 7):
                                self.ev_charger_compatible[ev_id].append(charger_id)
                            elif charger_type == "DC" and charger_power <= self.dc_charge_rate.get(ev_id, 50):
                                self.ev_charger_compatible[ev_id].append(charger_id)
                        else:
                            # Sin info de tasa de carga, asumir compatible
                            self.ev_charger_compatible[ev_id].append(charger_id)
            else:
                # Sin info de marca, asumir todos los cargadores son compatibles
                self.ev_charger_compatible[ev_id] = self.charger_ids
    
    def _build_ev_presence_map(self):
        """Construye un mapa de qué vehículos están presentes en cada intervalo."""
        presence_map = defaultdict(set)
        
        for ev_id in self.ev_ids:
            arrival_time = self.arrival_time[ev_id]
            departure_time = self.departure_time[ev_id]
            
            for t_idx, time_val in enumerate(self.times):
                if arrival_time <= time_val < departure_time:
                    presence_map[t_idx].add(ev_id)
        
        return presence_map
    
    def _get_state(self):
        """
        Obtiene el estado actual del entorno para el RL.
        RESTRICCIÓN MÁS FUERTE: Salta automáticamente períodos sin vehículos.
        Evita completamente planificar en horarios donde no hay vehículos presentes.
        """
        # Verificar inicialización
        if not hasattr(self, 'current_time_idx'):
            raise RuntimeError("Environment not properly initialized. Call reset() first.")
        
        # RESTRICCIÓN FUERTE: Saltar automáticamente todos los intervalos sin vehículos
        max_skips = len(self.times)  # Evitar bucles infinitos
        skips_made = 0
        
        while self.current_time_idx < len(self.times) and skips_made < max_skips:
            current_time = self.times[self.current_time_idx]
            
            # Obtener vehículos presentes que aún necesitan asignación
            evs_present_now = []
            for ev_id in self.ev_ids:
                # El vehículo está presente en este intervalo
                if (self.arrival_time[ev_id] <= current_time < self.departure_time[ev_id] and
                    ev_id not in self.evs_processed):
                    
                    # Verificar si aún necesita energía
                    energy_needed = self.required_energy[ev_id] - self.energy_delivered[ev_id]
                    if energy_needed > 0.01:  # Threshold para considerar que aún necesita carga
                        evs_present_now.append(ev_id)
            
            # Si HAY vehículos, procesar normalmente
            if evs_present_now:
                break
            
            # Si NO HAY vehículos, saltar automáticamente SIN penalización
            #print(f"⚡ Auto-skip: No hay vehículos en tiempo {current_time:.2f}h (idx: {self.current_time_idx})")
            self.current_time_idx += 1
            skips_made += 1
        
        # Si llegamos al final del tiempo, terminamos
        if self.current_time_idx >= len(self.times):
            #print(f"Simulación terminada: {len(self.times)} intervalos procesados")
            return None
        
        # AQUÍ YA SABEMOS QUE HAY VEHÍCULOS PRESENTES
        representative_ev = self._select_representative_ev(evs_present_now)
        if representative_ev is None:
            # Esto no debería pasar, pero por seguridad
            #print(f"Warning: No se pudo seleccionar vehículo representativo en t={current_time:.2f}h")
            self.current_time_idx += 1
            return self._get_state()
        
        ev_id = representative_ev
        
        # Calcular los intervalos FUTUROS en los que este EV estará presente
        # (solo desde el tiempo actual hacia adelante)
        ev_time_indices = [i for i in range(self.current_time_idx, len(self.times))
                        if self.arrival_time[ev_id] <= self.times[i] < self.departure_time[ev_id]]
        
        if not ev_time_indices:
            # Este EV ya no tiene tiempo disponible, marcarlo como procesado
            self.evs_processed.add(ev_id)
            #print(f"EV {ev_id} procesado: sin tiempo restante")
            return self._get_state()
        
        # Log de información útil para debugging
        #print(f"Procesando EV {ev_id} en t={current_time:.2f}h, "
            #f"{len(evs_present_now)} EVs presentes, "
            #f"{len(ev_time_indices)} intervalos futuros")
        
        # -- Cálculo de características del EV --
        
        # Características básicas (normalizadas)
        stay_duration = self.departure_time[ev_id] - self.arrival_time[ev_id]
        energy_requirement_normalized = self.required_energy[ev_id] / self.max_required_energy
        stay_duration_normalized = stay_duration / max(self.times)
        energy_delivered_ratio = self.energy_delivered[ev_id] / self.required_energy[ev_id] if self.required_energy[ev_id] > 0 else 0
        
        # Tiempo restante desde AHORA (no desde llegada)
        time_remaining_from_now = (self.departure_time[ev_id] - self.times[self.current_time_idx]) / max(self.times)
        
        ev_features = [
            self.arrival_time[ev_id] / max(self.times),  # Tiempo de llegada normalizado
            self.departure_time[ev_id] / max(self.times),  # Tiempo de salida normalizado
            energy_requirement_normalized,  # Energía requerida normalizada
            len(ev_time_indices) / len(self.times),  # Proporción de tiempo RESTANTE disponible
            energy_delivered_ratio,  # Proporción de energía ya entregada
            self.current_time_idx / len(self.times),  # Tiempo actual normalizado
            time_remaining_from_now   # Tiempo restante normalizado
        ]
        
        # Características avanzadas del vehículo
        if self.has_battery_info:
            battery_capacity_normalized = self.battery_capacity[ev_id] / 100.0
            ev_features.append(battery_capacity_normalized)
        
        if self.has_brand_info:
            brand_id = self.brand_to_id[self.brands[ev_id]]
            ev_features.append(brand_id)
            
        if self.has_priority_info:
            priority_normalized = self.priority[ev_id] / self.max_priority
            ev_features.append(priority_normalized)
            
        if self.has_willingness_info:
            willingness_normalized = self.willingness_to_pay[ev_id] / 1.5
            ev_features.append(willingness_normalized)
            
        if self.has_efficiency_info:
            ev_features.append(self.efficiency[ev_id])
            
        if self.has_charge_rate_info:
            min_rate_normalized = self.min_charge_rate[ev_id] / 50.0
            max_rate_normalized = self.max_charge_rate[ev_id] / 350.0
            ac_rate_normalized = self.ac_charge_rate[ev_id] / 50.0
            dc_rate_normalized = self.dc_charge_rate[ev_id] / 350.0
            
            ev_features.extend([
                min_rate_normalized,
                max_rate_normalized,
                ac_rate_normalized,
                dc_rate_normalized
            ])
        
        # -- Cálculo de características del entorno (solo intervalos futuros) --
        
        # Disponibilidad de spots y cargadores
        available_spots_ratio = []
        available_chargers_ratio = []
        compatible_chargers_ratio = []
        
        for t in ev_time_indices:
            # Ratio de spots disponibles
            ratio_spots = (self.n_spots - len(self.occupied_spots[t])) / self.n_spots
            available_spots_ratio.append(ratio_spots)
            
            # Ratio de cargadores disponibles
            ratio_chargers = (len(self.charger_ids) - len(self.occupied_chargers[t])) / len(self.charger_ids)
            available_chargers_ratio.append(ratio_chargers)
            
            # Ratio de cargadores compatibles disponibles
            if hasattr(self, 'ev_charger_compatible'):
                compatible_chargers = self.ev_charger_compatible.get(ev_id, self.charger_ids)
                occupied_compatible = len([c for c in compatible_chargers if c in self.occupied_chargers[t]])
                ratio_compatible = (len(compatible_chargers) - occupied_compatible) / max(1, len(compatible_chargers))
                compatible_chargers_ratio.append(ratio_compatible)
        
        # Precios de energía normalizados en los intervalos relevantes (futuros)
        relevant_prices = [self.normalized_prices[t] for t in ev_time_indices]
        
        # Capacidad restante del transformador en cada intervalo (normalizada)
        transformer_capacity_ratio = [(self.station_limit - self.power_used[t]) / self.station_limit 
                                    for t in ev_time_indices]
        
        # Valores agregados para simplificar el estado
        avg_available_spots = np.mean(available_spots_ratio) if available_spots_ratio else 0.0
        avg_available_chargers = np.mean(available_chargers_ratio) if available_chargers_ratio else 0.0
        min_price = np.min(relevant_prices) if relevant_prices else 0.5
        avg_price = np.mean(relevant_prices) if relevant_prices else 0.5
        min_transformer_capacity = np.min(transformer_capacity_ratio) if transformer_capacity_ratio else 0.0
        avg_transformer_capacity = np.mean(transformer_capacity_ratio) if transformer_capacity_ratio else 0.0
        
        if hasattr(self, 'ev_charger_compatible'):
            avg_compatible_chargers = np.mean(compatible_chargers_ratio) if compatible_chargers_ratio else 0.0
        else:
            avg_compatible_chargers = 1.0  # Todos compatibles por defecto
        
        # Calcular métricas adicionales útiles
        total_energy_needed = sum(max(0, self.required_energy[id] - self.energy_delivered[id]) 
                                for id in self.ev_ids if id not in self.evs_processed)
        
        # Capacidad restante del sistema (solo tiempo futuro)
        remaining_time_steps = len(self.times) - self.current_time_idx
        total_energy_capacity = self.total_charging_capacity * remaining_time_steps * self.dt
        system_demand_ratio = min(1.0, total_energy_needed / (total_energy_capacity + 1e-6))
        
        # Urgencia de carga ACTUAL
        time_remaining = (self.departure_time[ev_id] - self.times[self.current_time_idx])
        energy_needed = (self.required_energy[ev_id] - self.energy_delivered[ev_id]) / self.max_required_energy
        charging_urgency = energy_needed / (max(1e-6, time_remaining / max(self.times)))
        charging_urgency = min(1.0, charging_urgency)  # Normalizado entre 0 y 1
        
        # Competencia por recursos (cuántos otros EVs necesitan asignación ahora)
        evs_competing = len([ev for ev in evs_present_now if ev != ev_id])
        competition_pressure = min(1.0, evs_competing / max(1, self.n_spots))
        
        # Estado completo más detallado
        state = {
            "ev_features": ev_features,
            "spot_availability": available_spots_ratio,
            "charger_availability": available_chargers_ratio,
            "compatible_chargers_ratio": compatible_chargers_ratio if hasattr(self, 'ev_charger_compatible') else [1.0] * len(ev_time_indices),
            "relevant_prices": relevant_prices,
            "transformer_capacity": transformer_capacity_ratio,
            "time_indices": ev_time_indices,
            
            # Características agregadas para facilitar la generalización
            "avg_available_spots": avg_available_spots,
            "avg_available_chargers": avg_available_chargers,
            "avg_compatible_chargers": avg_compatible_chargers,
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
            
            # NUEVAS características temporales y de competencia
            "current_time_idx": self.current_time_idx,
            "current_time_normalized": self.current_time_idx / len(self.times),
            "system_demand_ratio": system_demand_ratio,
            "charging_urgency": charging_urgency,
            "time_remaining": time_remaining_from_now,
            "competition_pressure": competition_pressure,
            "evs_present_count": len(evs_present_now),
            "representative_ev": ev_id
        }
        
        # Añadir propiedades de vehículo si están disponibles
        if self.has_battery_info:
            state["battery_capacity"] = self.battery_capacity[ev_id] / 100.0
            
        if self.has_priority_info:
            state["priority"] = self.priority[ev_id] / self.max_priority
            
        if self.has_willingness_info:
            state["willingness_to_pay"] = self.willingness_to_pay[ev_id] / 1.5
            
        if self.has_efficiency_info:
            state["efficiency"] = self.efficiency[ev_id]
            
        if self.has_charge_rate_info:
            state["min_charge_rate"] = self.min_charge_rate[ev_id] / 50.0
            state["max_charge_rate"] = self.max_charge_rate[ev_id] / 350.0
            state["ac_charge_rate"] = self.ac_charge_rate[ev_id] / 50.0
            state["dc_charge_rate"] = self.dc_charge_rate[ev_id] / 350.0
        
        return state

    def _build_waiting_schedule(self):
        """
        Construye un mapping de qué vehículos están esperando asignación en cada intervalo.
        """
        waiting_schedule = defaultdict(set)
        
        for ev_id in self.ev_ids:
            arrival_time = self.arrival_time[ev_id]
            departure_time = self.departure_time[ev_id]
            
            for t_idx, time_val in enumerate(self.times):
                if arrival_time <= time_val < departure_time:
                    waiting_schedule[t_idx].add(ev_id)
        
        return waiting_schedule

    def _select_representative_ev(self, evs_present):
        """
        Selecciona el vehículo más adecuado para representar el estado actual.
        Prioriza por urgencia, luego por prioridad, luego por orden de llegada.
        """
        if not evs_present:
            return None
        
        current_time = self.times[self.current_time_idx]
        
        # Calcular scoring para cada EV
        ev_scores = []
        for ev_id in evs_present:
            # Factor de urgencia (energía necesaria / tiempo restante)
            energy_needed = self.required_energy[ev_id] - self.energy_delivered[ev_id]
            time_remaining = max(1e-6, self.departure_time[ev_id] - current_time)
            urgency_score = energy_needed / time_remaining
            
            # Factor de prioridad
            priority_score = self.priority.get(ev_id, 1) if self.has_priority_info else 1
            
            # Factor de willingness to pay
            willingness_score = self.willingness_to_pay.get(ev_id, 1) if self.has_willingness_info else 1
            
            # Score combinado (normalizado)
            combined_score = (urgency_score * 0.5 + priority_score * 0.3 + willingness_score * 0.2)
            
            ev_scores.append((ev_id, combined_score))
        
        # Seleccionar el EV con mayor score
        ev_scores.sort(key=lambda x: x[1], reverse=True)
        return ev_scores[0][0]
    
    def _get_possible_actions(self, state):
        """
        Determina las acciones posibles dado el estado actual.
        RESTRICCIÓN MÁS FUERTE: Solo genera acciones cuando realmente hay vehículos y recursos.
        OPTIMIZADO: Reduce drásticamente el espacio de acciones para acelerar entrenamiento.
        """
        if state is None:
            return []
        
        # Verificar que tenemos un vehículo representativo válido
        if "representative_ev" not in state:
            return [{"skip": True}]
        
        ev_id = state["representative_ev"]
        current_time_idx = state.get("current_time_idx", 0)
        time_indices = state["time_indices"]
        
        # VERIFICACIÓN ADICIONAL: Asegurar que el vehículo realmente esté presente AHORA
        current_time = self.times[current_time_idx]
        if not (self.arrival_time[ev_id] <= current_time < self.departure_time[ev_id]):
            #print(f" Warning: EV {ev_id} no está presente en tiempo {current_time:.2f}h")
            return [{"skip": True}]
        
        # Verificar que el vehículo aún necesita energía
        energy_needed = self.required_energy[ev_id] - self.energy_delivered[ev_id]
        if energy_needed <= 0.01:  # Threshold mínimo
            #print(f" EV {ev_id} ya está completamente cargado")
            return [{"skip": True}]
        
        # Verificar recursos disponibles EN EL TIEMPO ACTUAL
        current_available_spots = [s for s in range(self.n_spots) 
                                if s not in self.occupied_spots[current_time_idx]]
        current_available_chargers = [c for c in self.charger_ids 
                                    if c not in self.occupied_chargers[current_time_idx]]
        
        # Si no hay recursos, solo skip
        if not current_available_spots or not current_available_chargers:
            #print(f" Sin recursos: spots={len(current_available_spots)}, chargers={len(current_available_chargers)}")
            return [{"skip": True}]
        
        # VERIFICACIÓN DE TIEMPO SUFICIENTE: Verificar que haya tiempo suficiente para carga mínima
        time_remaining = self.departure_time[ev_id] - current_time
        min_power = self.min_charge_rate[ev_id] if self.has_charge_rate_info else 3.5
        min_energy_possible = min_power * time_remaining
        
        if min_energy_possible < 0.1:  # Menos de 0.1 kWh posible
            #print(f" EV {ev_id}: tiempo insuficiente ({time_remaining:.2f}h) para carga mínima")
            return [{"skip": True}]
        
        # VERIFICACIÓN DE VIABILIDAD ECONÓMICA: Asegurar que vale la pena cargar
        if energy_needed < 0.5:  # Menos de 0.5 kWh necesario
            #print(f" EV {ev_id}: energía necesaria muy baja ({energy_needed:.2f} kWh)")
            return [{"skip": True}]
        
        # Solo ahora generar acciones reales
        actions = []
        skip_action = {"skip": True}
        
        # Calcular potencia ideal
        available_time = len(time_indices) * self.dt
        if available_time <= 0:
            return [skip_action]
        
        ideal_power = energy_needed / available_time
        max_power = self.max_charge_rate[ev_id] if self.has_charge_rate_info else 50
        
        # OPTIMIZACIÓN FUERTE: Máximo 2 spots para reducir aún más el espacio de acciones
        spots_to_try = current_available_spots[:2] if len(current_available_spots) > 2 else current_available_spots
        
        #print(f" Generando acciones para EV {ev_id}: {len(spots_to_try)} spots, energía_necesaria={energy_needed:.2f}kWh")
        
        for spot in spots_to_try:
            # Determinar cargadores compatibles y disponibles AHORA
            if hasattr(self, 'ev_charger_compatible'):
                compatible_chargers = [c for c in self.ev_charger_compatible.get(ev_id, self.charger_ids)
                                    if c in current_available_chargers]
            else:
                compatible_chargers = current_available_chargers
            
            if not compatible_chargers:
                continue  # No hay cargadores compatibles para este EV
            
            # SIMPLIFICACIÓN EXTREMA: Solo 2 niveles de potencia para reducir complejidad
            available_powers = sorted(set(self.max_charger_power_dict[c] for c in compatible_chargers))
            
            if len(available_powers) >= 2:
                power_levels = [available_powers[0], available_powers[-1]]  # Solo min y max
            else:
                power_levels = available_powers
            
            # Agregar potencia ideal si es factible y está en rango
            if min_power <= ideal_power <= max_power:
                suitable_chargers = [c for c in compatible_chargers 
                                if self.max_charger_power_dict[c] >= ideal_power]
                if suitable_chargers:
                    power_levels.append(ideal_power)
            
            # Filtrar y eliminar duplicados
            power_levels = sorted(set(p for p in power_levels if min_power <= p <= max_power))
            
            for required_power in power_levels:
                # Verificar factibilidad de esta asignación
                action_feasible = self._check_action_feasibility(
                    ev_id, spot, required_power, current_time_idx, time_indices, compatible_chargers
                )
                
                if action_feasible:
                    actions.append(action_feasible)
        
        # Siempre incluir skip como opción
        actions.append(skip_action)
        
        #print(f" Generadas {len(actions)-1} acciones + skip para EV {ev_id} en t={current_time:.2f}h")
        
        return actions

    def _check_action_feasibility(self, ev_id, spot, required_power, current_time_idx, time_indices, compatible_chargers):
        """
        Verificación de factibilidad MÁS ESTRICTA.
        Solo permite acciones que realmente tienen sentido económico y energético.
        """
        # Verificar capacidad del transformador EN EL TIEMPO ACTUAL
        if self.power_used[current_time_idx] + required_power > self.station_limit:
            return None  # Excede capacidad del transformador
        
        # Buscar el mejor cargador para esta potencia en el tiempo actual
        suitable_chargers = [c for c in compatible_chargers 
                            if (c not in self.occupied_chargers[current_time_idx] and
                                self.max_charger_power_dict[c] >= required_power)]
        
        if not suitable_chargers:
            return None  # No hay cargadores disponibles y adecuados
        
        # Seleccionar el cargador con menor desperdicio de potencia
        best_charger = min(suitable_chargers, 
                        key=lambda c: self.max_charger_power_dict[c] - required_power)
        
        # VERIFICACIÓN ADICIONAL: Asegurar que la asignación tenga sentido económico/energético
        energy_to_deliver = min(required_power * self.dt, 
                            self.required_energy[ev_id] - self.energy_delivered[ev_id])
        
        if energy_to_deliver < 0.01:  # Menos de 0.01 kWh
            return None  # No vale la pena la asignación
        
        # VERIFICACIÓN DE EFICIENCIA: Evitar asignaciones con mucho desperdicio de potencia
        power_waste_ratio = (self.max_charger_power_dict[best_charger] - required_power) / self.max_charger_power_dict[best_charger]
        if power_waste_ratio > 0.8:  # Más del 80% de desperdicio
            return None  # Demasiado ineficiente
        
        action = {
            "skip": False,
            "ev_id": ev_id,
            "spot": spot,
            "charger": best_charger,
            "power": required_power,
            "time_idx": current_time_idx,
            "duration": 1,  # Solo para este intervalo de tiempo
            "energy_to_deliver": energy_to_deliver
        }
        
        return action
    
    def step(self, action_idx):
        """
        Ejecuta una acción y avanza el entorno al siguiente estado.
        CORREGIDO: Funciona con enfoque temporal consistente.
        OPTIMIZADO: Recompensas balanceadas para mejor aprendizaje.
        """
        state = self._get_state()
        if state is None:
            return None, 0, True

        actions = self._get_possible_actions(state)

        # Protección contra índices inválidos
        if action_idx < 0 or action_idx >= len(actions):
            action_idx = len(actions) - 1

        action = actions[action_idx]

        # CAMBIO PRINCIPAL: Usar el vehículo representativo del estado
        if "representative_ev" not in state:
            return None, 0, True
        
        ev_id = state["representative_ev"]
        current_time_idx = state.get("current_time_idx", 0)

        if action["skip"]:
            # Calcular penalización por saltar
            energy_deficit = self.required_energy[ev_id] - self.energy_delivered[ev_id]
            priority_factor = self.priority[ev_id] if self.has_priority_info else 1.0
            urgency_factor = min(2.0, energy_deficit / self.avg_required_energy)
            
            # Penalización reducida si no hay recursos disponibles
            if (len(self.occupied_spots[current_time_idx]) >= self.n_spots or 
                len(self.occupied_chargers[current_time_idx]) >= len(self.charger_ids)):
                # Penalización menor si no hay recursos
                reward = -self.PENALTY_FOR_SKIPPED_VEHICLE * 0.3 * priority_factor * urgency_factor
            else:
                # Penalización normal por saltar cuando hay recursos
                reward = -self.PENALTY_FOR_SKIPPED_VEHICLE * priority_factor * urgency_factor
            
            # Avanzar al siguiente intervalo de tiempo
            self.current_time_idx += 1
            
            return self._get_state(), reward, self.current_time_idx >= len(self.times)

        # NUEVA ESTRUCTURA: Manejar acción de asignación inmediata
        if "charging_profile" in action:
            # Compatibilidad con estructura legacy
            reward = self._execute_legacy_action(action, ev_id, state)
        else:
            # Nueva estructura simplificada
            reward = self._execute_immediate_action(action, ev_id, state)

        return self._get_state(), reward, self.current_time_idx >= len(self.times)

    def _execute_immediate_action(self, action, ev_id, state):
        """
        Ejecuta una acción inmediata (solo para el tiempo actual).
        NUEVO: Maneja la estructura de acción simplificada.
        """
        spot = action["spot"]
        charger = action["charger"]
        power = action["power"]
        time_idx = action["time_idx"]
        energy_to_deliver = action["energy_to_deliver"]
        
        # Registrar asignación para este intervalo
        self.charging_schedule.append((ev_id, time_idx, charger, spot, power))
        self.occupied_spots[time_idx].add(spot)
        self.occupied_chargers[time_idx].add(charger)
        self.power_used[time_idx] += power
        
        # Calcular energía efectiva considerando eficiencias
        efficiency = self.efficiency[ev_id] if self.has_efficiency_info else 0.9
        charger_eff = 0.95  # Eficiencia del cargador
        effective_energy = energy_to_deliver * efficiency * charger_eff
        
        # Actualizar energía entregada
        remaining_needed = max(0, self.required_energy[ev_id] - self.energy_delivered[ev_id])
        actual_energy = min(effective_energy, remaining_needed)
        self.energy_delivered[ev_id] += actual_energy
        
        # Calcular costo
        current_price = self.prices[time_idx]
        energy_cost = actual_energy * current_price
        
        # Calcular recompensa
        reward = self._calculate_assignment_reward(ev_id, actual_energy, energy_cost, power, state)
        
        # Verificar si el vehículo está completamente cargado
        satisfaction_ratio = self.energy_delivered[ev_id] / self.required_energy[ev_id]
        if satisfaction_ratio >= 0.99:  # 99% cargado
            self.evs_processed.add(ev_id)
            #print(f"EV {ev_id} completamente cargado ({satisfaction_ratio:.1%})")
        
        # Avanzar al siguiente intervalo
        self.current_time_idx += 1
        
        return reward

    def _execute_legacy_action(self, action, ev_id, state):
        """
        Ejecuta una acción con estructura legacy (charging_profile).
        COMPATIBILIDAD: Para mantener funcionamiento con código existente.
        """
        spot = action["spot"]
        charging_profile = action["charging_profile"]

        total_cost = 0
        total_energy = 0

        for t, (charger, power) in charging_profile.items():
            self.charging_schedule.append((ev_id, t, charger, spot, power))
            self.occupied_spots[t].add(spot)
            self.occupied_chargers[t].add(charger)
            self.power_used[t] += power

            efficiency = self.efficiency[ev_id] if self.has_efficiency_info else 0.9
            charger_eff = 0.95
            effective_energy = power * self.dt * efficiency * charger_eff
            remaining_needed = max(0, self.required_energy[ev_id] - self.energy_delivered[ev_id])
            actual_energy = min(effective_energy, remaining_needed)
            self.energy_delivered[ev_id] += actual_energy
            total_energy += effective_energy
            total_cost += effective_energy * self.prices[t]

        energy_satisfaction_ratio = min(1.0, self.energy_delivered[ev_id] / self.required_energy[ev_id])
        satisfaction_reward = self.ENERGY_SATISFACTION_WEIGHT * energy_satisfaction_ratio * 100

        if energy_satisfaction_ratio >= 0.95:
            satisfaction_reward += self.REWARD_FOR_ASSIGNED_VEHICLE

        if total_energy > 0:
            avg_price_paid = total_cost / total_energy
            price_penalty = -self.ENERGY_COST_WEIGHT * (avg_price_paid - self.min_price) / (self.max_price - self.min_price + 1e-6)
        else:
            price_penalty = 0

        priority_multiplier = 1.0
        if self.has_priority_info:
            priority_multiplier = 0.5 + 0.5 * (self.priority[ev_id] / self.max_priority)

        reward = satisfaction_reward * priority_multiplier + price_penalty
        
        # Marcar vehículo como procesado si está completo
        if energy_satisfaction_ratio >= 0.99:
            self.evs_processed.add(ev_id)
        
        # Con enfoque legacy, avanzamos varios intervalos
        self.current_time_idx = max(charging_profile.keys()) + 1 if charging_profile else self.current_time_idx + 1

        return reward

    def _calculate_assignment_reward(self, ev_id, actual_energy, energy_cost, power, state):
        """
        Calcula recompensa por asignación exitosa con el nuevo enfoque.
        """
        # Recompensa base por energía entregada
        energy_satisfaction = min(1.0, self.energy_delivered[ev_id] / self.required_energy[ev_id])
        satisfaction_reward = self.ENERGY_SATISFACTION_WEIGHT * energy_satisfaction * 100
        
        # Bonus por completar la carga
        if energy_satisfaction >= 0.95:
            satisfaction_reward += self.REWARD_FOR_ASSIGNED_VEHICLE
        
        # Penalización por costo (normalizada)
        price_penalty = -self.ENERGY_COST_WEIGHT * energy_cost
        
        # Bonus por eficiencia (usar potencia adecuada)
        efficiency_bonus = 10 if actual_energy > 0 else 0
        
        # Factor de prioridad
        priority_multiplier = 1.0
        if self.has_priority_info:
            priority_multiplier = 0.5 + 0.5 * (self.priority[ev_id] / self.max_priority)
        
        # Factor de urgencia (bonus por atender vehículos urgentes)
        urgency_bonus = 0
        if "charging_urgency" in state:
            urgency_bonus = state["charging_urgency"] * 20
        
        # Factor de competencia (bonus por atender cuando hay mucha demanda)
        competition_bonus = 0
        if "competition_pressure" in state:
            competition_bonus = state["competition_pressure"] * 15
        
        total_reward = ((satisfaction_reward + efficiency_bonus + urgency_bonus + competition_bonus) * 
                    priority_multiplier + price_penalty)
        
        return total_reward

    def reset(self):
        """
        Reinicia el entorno al estado inicial.
        CORREGIDO: Inicialización consistente con enfoque temporal.
        """
        # Estado temporal del entorno
        self.current_time_idx = 0  # ← Enfoque temporal
        self.evs_processed = set()  # ← Tracking de EVs procesados
        
        # Estados del entorno
        self.spot_assignments = {}
        self.charging_schedule = []
        self.occupied_spots = {t: set() for t in range(len(self.times))}
        self.occupied_chargers = {t: set() for t in range(len(self.times))}
        self.power_used = {t: 0 for t in range(len(self.times))}
        self.energy_delivered = {ev_id: 0 for ev_id in self.ev_ids}
        
        # Tracking de vehículos por intervalo
        self.evs_present_at_time = self._build_ev_presence_map()
        self.evs_waiting_assignment = {t: set() for t in range(len(self.times))}
        self.evs_assigned = set()
        
        # AGREGADO: Inicializar evs_waiting_by_time aquí (movido desde _get_state)
        self.evs_waiting_by_time = self._build_waiting_schedule()
        
        # Inicializar vehículos esperando asignación
        for t in range(len(self.times)):
            for ev_id in self.evs_present_at_time[t]:
                if ev_id not in self.evs_assigned:
                    self.evs_waiting_assignment[t].add(ev_id)
        
        return self._get_state()

    def _get_possible_actions_legacy(self, state):
        """
        VERSIÓN LEGACY para compatibilidad con código existente que espera charging_profile.
        Esta versión mantiene la estructura original pero con las correcciones temporales.
        """
        if state is None:
            return []
        
        if "representative_ev" not in state:
            return [{"skip": True}]
        
        ev_id = state["representative_ev"]
        time_indices = state["time_indices"]
        
        # Verificar que el vehículo aún necesita energía
        energy_needed = self.required_energy[ev_id] - self.energy_delivered[ev_id]
        if energy_needed <= 0.01:
            return [{"skip": True}]
        
        # Determinar qué slots están disponibles durante TODO el período futuro
        consistent_spots = list(range(self.n_spots))
        for t in time_indices:
            consistent_spots = [s for s in consistent_spots if s not in self.occupied_spots[t]]
        
        actions = []
        
        # Opción de saltar
        skip_action = {"skip": True}
        
        # Restricciones específicas del vehículo
        min_power = self.min_charge_rate[ev_id] if self.has_charge_rate_info else 3.5
        max_power = self.max_charge_rate[ev_id] if self.has_charge_rate_info else 50
        
        # Calcular la potencia requerida ideal (energía/tiempo disponible)
        available_time = len(time_indices) * self.dt
        
        if available_time <= 0:
            return [skip_action]  # No hay tiempo, solo podemos saltar
            
        ideal_power = energy_needed / available_time
        
        # OPTIMIZACIÓN: Limitar a máximo 3 spots para reducir complejidad
        spots_to_try = consistent_spots[:3] if len(consistent_spots) > 3 else consistent_spots
        
        # Para cada spot disponible, generamos acciones posibles
        for spot in spots_to_try:
            # OPTIMIZACIÓN: Solo 3 niveles de potencia en lugar de muchos
            power_levels = []
            
            # Determinar cargadores compatibles
            if hasattr(self, 'ev_charger_compatible'):
                compatible_chargers = self.ev_charger_compatible.get(ev_id, self.charger_ids)
            else:
                compatible_chargers = self.charger_ids
                
            available_powers = sorted(set(self.max_charger_power_dict[c] 
                                        for c in compatible_chargers))
            
            # Solo usar 3 niveles: mínimo, medio, máximo
            if len(available_powers) >= 3:
                power_levels = [available_powers[0], available_powers[len(available_powers)//2], available_powers[-1]]
            else:
                power_levels = available_powers
            
            # Agregar potencia ideal si está en rango
            if min_power <= ideal_power <= max_power:
                power_levels.append(ideal_power)
            
            # Eliminar duplicados y filtrar por rango
            power_levels = sorted(set(p for p in power_levels if min_power <= p <= max_power))
            
            for required_power in power_levels:
                # Verificamos si hay suficiente capacidad de cargador y transformador
                feasible_charging = True
                charger_assignments = {}
                
                for t in time_indices:
                    # Buscamos un cargador compatible y disponible con suficiente potencia
                    if hasattr(self, 'ev_charger_compatible'):
                        available_chargers = [c for c in self.ev_charger_compatible.get(ev_id, self.charger_ids) 
                                            if c not in self.occupied_chargers[t] and 
                                            self.max_charger_power_dict[c] >= required_power and
                                            self.power_used[t] + required_power <= self.station_limit]
                    else:
                        available_chargers = [c for c in self.charger_ids 
                                            if c not in self.occupied_chargers[t] and 
                                            self.max_charger_power_dict[c] >= required_power and
                                            self.power_used[t] + required_power <= self.station_limit]
                    
                    if not available_chargers:
                        feasible_charging = False
                        break
                    
                    # Asignamos el cargador con la potencia más adecuada (menor desperdicio)
                    best_charger = min(available_chargers, 
                                    key=lambda c: self.max_charger_power_dict[c] - required_power)
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
    
    def get_schedule(self):
        """Retorna el schedule de carga completo generado."""
        return self.charging_schedule
    
    def get_energy_satisfaction_metrics(self):
        """Retorna métricas detalladas sobre satisfacción de energía."""
        metrics = {}
        
        # Energía total requerida y entregada
        total_required = sum(self.required_energy.values())
        total_delivered = sum(self.energy_delivered.values())
        
        # Porcentaje total de satisfacción
        total_satisfaction_pct = (total_delivered / total_required) * 100 if total_required > 0 else 100
        
        metrics["total_required_energy"] = total_required
        metrics["total_delivered_energy"] = total_delivered
        metrics["total_satisfaction_pct"] = total_satisfaction_pct
        
        # Métricas por vehículo
        ev_metrics = {}
        for ev_id in self.ev_ids:
            required = self.required_energy[ev_id]
            delivered = self.energy_delivered[ev_id]
            satisfaction = delivered / required if required > 0 else 1.0
            
            ev_metrics[ev_id] = {
                "required_energy": required,
                "delivered_energy": delivered,
                "satisfaction": satisfaction,
                "priority": self.priority[ev_id] if self.has_priority_info else 1,
                "willingness": self.willingness_to_pay[ev_id] if self.has_willingness_info else 1.0
            }
        
        metrics["ev_metrics"] = ev_metrics
        
        # Métricas agrupadas por prioridad si disponible
        if self.has_priority_info:
            by_priority = {}
            for priority in range(1, self.max_priority + 1):
                priority_evs = [ev for ev in self.ev_ids if self.priority[ev] == priority]
                if not priority_evs:
                    continue
                    
                required = sum(self.required_energy[ev] for ev in priority_evs)
                delivered = sum(self.energy_delivered[ev] for ev in priority_evs)
                satisfaction = delivered / required if required > 0 else 1.0
                
                by_priority[priority] = {
                    "count": len(priority_evs),
                    "required_energy": required,
                    "delivered_energy": delivered,
                    "satisfaction": satisfaction
                }
            
            metrics["by_priority"] = by_priority
        
        return metrics
    
    def update_reward_weights(self, weights_dict: Dict[str, float]):
        """Actualiza pesos de recompensa dinámicamente"""
        self.ENERGY_SATISFACTION_WEIGHT = weights_dict.get(
            "energy_satisfaction_weight", self.ENERGY_SATISFACTION_WEIGHT
        )
        self.ENERGY_COST_WEIGHT = weights_dict.get(
            "energy_cost_weight", self.ENERGY_COST_WEIGHT
        )
        self.PENALTY_FOR_SKIPPED_VEHICLE = weights_dict.get(
            "penalty_skipped_vehicle", self.PENALTY_FOR_SKIPPED_VEHICLE
        )
        self.REWARD_FOR_ASSIGNED_VEHICLE = weights_dict.get(
            "reward_assigned_vehicle", self.REWARD_FOR_ASSIGNED_VEHICLE
        )