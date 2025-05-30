import numpy as np
from collections import deque, defaultdict
from typing import Dict

class EVChargingEnv:
    """
    Enhanced simulation environment for Electric Vehicle (EV) charging with Reinforcement Learning (RL).
    Incorporates new features and constraints for advanced systems.
    OPTIMIZED: Balanced rewards and more efficient learning.
    """
    def __init__(self, config):
        """
        Initializes the environment with the system configuration.

        Args:
            config (dict): A dictionary containing the environment's configuration parameters.
        """
        self.ENERGY_SATISFACTION_WEIGHT = 1.0  
        self.PENALTY_FOR_SKIPPED_VEHICLE = 50.0 
        self.ENERGY_COST_WEIGHT = 0.1 
        self.REWARD_FOR_ASSIGNED_VEHICLE = 20.0  
        self.config = config
        self.process_config()
        self.reset()
        
    def process_config(self):
        """
        Processes the configuration to extract parameters.
        """
        # Basic system parameters
        self.times = self.config["times"]
        self.prices = self.config["prices"]
        self.arrivals = self.config["arrivals"]
        self.chargers = self.config["parking_config"]["chargers"]
        self.station_limit = self.config["parking_config"]["transformer_limit"]
        self.dt = self.config.get("dt", 0.25)  # Default 15 min
        self.n_spots = self.config["parking_config"]["n_spots"]
        self.test_number = self.config.get("test_number", 0)
        
        # Detect available additional features
        self.has_brand_info = any('brand' in ev for ev in self.arrivals)
        self.has_battery_info = any('battery_capacity' in ev for ev in self.arrivals)
        self.has_priority_info = any('priority' in ev for ev in self.arrivals)
        self.has_willingness_info = any('willingness_to_pay' in ev for ev in self.arrivals)
        self.has_efficiency_info = any('efficiency' in ev for ev in self.arrivals)
        self.has_charge_rate_info = any('min_charge_rate' in ev for ev in self.arrivals)
        
        # Price normalization
        self.min_price = min(self.prices)
        self.max_price = max(self.prices)
        self.normalized_prices = [(p - self.min_price) / (self.max_price - self.min_price + 1e-6) 
                                 for p in self.prices]
        
        # System statistics calculation
        self.max_charger_power = max(c["power"] for c in self.chargers)
        self.total_charging_capacity = sum(c["power"] for c in self.chargers)
        self.avg_required_energy = np.mean([arr["required_energy"] for arr in self.arrivals])
        self.max_required_energy = max(arr["required_energy"] for arr in self.arrivals)
        self.avg_stay_duration = np.mean([(arr["departure_time"] - arr["arrival_time"]) 
                                         for arr in self.arrivals])
        
        # Mappings for easier access
        self.ev_ids = [arr["id"] for arr in self.arrivals]
        self.arrival_time = {arr["id"]: arr["arrival_time"] for arr in self.arrivals}
        self.departure_time = {arr["id"]: arr["departure_time"] for arr in self.arrivals}
        self.required_energy = {arr["id"]: arr["required_energy"] for arr in self.arrivals}
        
        # Additional advanced information
        if self.has_battery_info:
            self.battery_capacity = {arr["id"]: arr.get("battery_capacity", 40) 
                                     for arr in self.arrivals}
        
        if self.has_brand_info:
            self.brands = {arr["id"]: arr.get("brand", "Unknown") for arr in self.arrivals}
            # Numeric mapping for brands
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
        
        # Charger information
        self.charger_ids = [c["charger_id"] for c in self.chargers]
        self.max_charger_power_dict = {c["charger_id"]: c["power"] for c in self.chargers}
        
        # Charger types and compatibility
        self.charger_type = {}
        self.compatible_vehicles = {}
        
        for c in self.chargers:
            charger_id = c["charger_id"]
            self.charger_type[charger_id] = c.get("type", "AC")
            
            if "compatible_vehicles" in c:
                self.compatible_vehicles[charger_id] = c["compatible_vehicles"]
            else:
                # Default to compatible with all
                self.compatible_vehicles[charger_id] = list(set(self.brands.values())) if self.has_brand_info else ["All"]
        
        # EV-Charger compatibility mapping
        self.ev_charger_compatible = {}
        for ev_id in self.ev_ids:
            self.ev_charger_compatible[ev_id] = []
            
            if self.has_brand_info:
                ev_brand = self.brands.get(ev_id, "")
                
                for charger_id in self.charger_ids:
                    # Check compatibility by brand
                    compatible = True
                    if ev_brand and charger_id in self.compatible_vehicles:
                        # Extract base brand without details
                        base_brand = ev_brand.split()[0] if " " in ev_brand else ev_brand
                        compatible = False
                        for comp_brand in self.compatible_vehicles[charger_id]:
                            if comp_brand in ev_brand or base_brand in comp_brand:
                                compatible = True
                                break
                    
                    if compatible:
                        # Check power limits
                        charger_power = self.max_charger_power_dict[charger_id]
                        charger_type = self.charger_type.get(charger_id, "AC")
                        
                        if self.has_charge_rate_info:
                            # Check compatibility with charge rates
                            if charger_type == "AC" and charger_power <= self.ac_charge_rate.get(ev_id, 7):
                                self.ev_charger_compatible[ev_id].append(charger_id)
                            elif charger_type == "DC" and charger_power <= self.dc_charge_rate.get(ev_id, 50):
                                self.ev_charger_compatible[ev_id].append(charger_id)
                        else:
                            # No charge rate info, assume compatible
                            self.ev_charger_compatible[ev_id].append(charger_id)
            else:
                # No brand info, assume all chargers are compatible
                self.ev_charger_compatible[ev_id] = self.charger_ids
    
    def _build_ev_presence_map(self):
        """
        Builds a map of which vehicles are present at each interval.

        Returns:
            defaultdict[int, set]: A dictionary where keys are time indices and values are sets of EV IDs present at that time.
        """
        presence_map = defaultdict(set)
        
        for ev_id in self.ev_ids:
            arrival_time = self.arrival_time[ev_id]
            departure_time = self.departure_time[ev_id]
            
            for t_idx, time_val in enumerate(self.times):
                if arrival_time <= time_val < departure_time:
                    presence_map[t_idx].add(ev_id)
        
        return presence_map
    
    def _get_state(self):
        if not hasattr(self, 'current_time_idx'):
            raise RuntimeError("Environment not properly initialized. Call reset() first.")
        
        max_skips = len(self.times)
        skips_made = 0
        
        while self.current_time_idx < len(self.times) and skips_made < max_skips:
            current_time = self.times[self.current_time_idx]
            
            evs_present_now = []
            for ev_id in self.ev_ids:
                if (self.arrival_time[ev_id] <= current_time < self.departure_time[ev_id] and
                        ev_id not in self.evs_processed):
                    energy_needed = self.required_energy[ev_id] - self.energy_delivered[ev_id]
                    if energy_needed > 0.01:
                        evs_present_now.append(ev_id)
            
            if evs_present_now:
                break
            
            self.current_time_idx += 1
            skips_made += 1
        
        if self.current_time_idx >= len(self.times):
            return None
        
        representative_ev = self._select_representative_ev(evs_present_now)
        if representative_ev is None:
            self.current_time_idx += 1
            return self._get_state()
        
        ev_id = representative_ev
        
        ev_time_indices = [i for i in range(self.current_time_idx, len(self.times))
                            if self.arrival_time[ev_id] <= self.times[i] < self.departure_time[ev_id]]
        
        if not ev_time_indices:
            self.evs_processed.add(ev_id)
            return self._get_state()
        
        stay_duration = self.departure_time[ev_id] - self.arrival_time[ev_id]
        energy_requirement_normalized = self.required_energy[ev_id] / self.max_required_energy
        stay_duration_normalized = stay_duration / max(self.times)
        energy_delivered_ratio = self.energy_delivered[ev_id] / self.required_energy[ev_id] if self.required_energy[ev_id] > 0 else 0
        
        time_remaining_from_now = (self.departure_time[ev_id] - self.times[self.current_time_idx]) / max(self.times)
        
        ev_features = [
            self.arrival_time[ev_id] / max(self.times),
            self.departure_time[ev_id] / max(self.times),
            energy_requirement_normalized,
            len(ev_time_indices) / len(self.times),
            energy_delivered_ratio,
            self.current_time_idx / len(self.times),
            time_remaining_from_now
        ]
        
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
        
        available_spots_ratio = []
        available_chargers_ratio = []
        compatible_chargers_ratio = []
        
        for t in ev_time_indices:
            ratio_spots = (self.n_spots - len(self.occupied_spots[t])) / self.n_spots
            available_spots_ratio.append(ratio_spots)
            
            ratio_chargers = (len(self.charger_ids) - len(self.occupied_chargers[t])) / len(self.charger_ids)
            available_chargers_ratio.append(ratio_chargers)
            
            if hasattr(self, 'ev_charger_compatible'):
                compatible_chargers = self.ev_charger_compatible.get(ev_id, self.charger_ids)
                occupied_compatible = len([c for c in compatible_chargers if c in self.occupied_chargers[t]])
                ratio_compatible = (len(compatible_chargers) - occupied_compatible) / max(1, len(compatible_chargers))
                compatible_chargers_ratio.append(ratio_compatible)
        
        relevant_prices = [self.normalized_prices[t] for t in ev_time_indices]
        
        transformer_capacity_ratio = [(self.station_limit - self.power_used[t]) / self.station_limit 
                                    for t in ev_time_indices]
        
        avg_available_spots = np.mean(available_spots_ratio) if available_spots_ratio else 0.0
        avg_available_chargers = np.mean(available_chargers_ratio) if available_chargers_ratio else 0.0
        min_price = np.min(relevant_prices) if relevant_prices else 0.5
        avg_price = np.mean(relevant_prices) if relevant_prices else 0.5
        min_transformer_capacity = np.min(transformer_capacity_ratio) if transformer_capacity_ratio else 0.0
        avg_transformer_capacity = np.mean(transformer_capacity_ratio) if transformer_capacity_ratio else 0.0
        
        if hasattr(self, 'ev_charger_compatible'):
            avg_compatible_chargers = np.mean(compatible_chargers_ratio) if compatible_chargers_ratio else 0.0
        else:
            avg_compatible_chargers = 1.0
        
        total_energy_needed = sum(max(0, self.required_energy[id] - self.energy_delivered[id]) 
                                for id in self.ev_ids if id not in self.evs_processed)
        
        remaining_time_steps = len(self.times) - self.current_time_idx
        total_energy_capacity = self.total_charging_capacity * remaining_time_steps * self.dt
        system_demand_ratio = min(1.0, total_energy_needed / (total_energy_capacity + 1e-6))
        
        time_remaining = (self.departure_time[ev_id] - self.times[self.current_time_idx])
        energy_needed = (self.required_energy[ev_id] - self.energy_delivered[ev_id]) / self.max_required_energy
        charging_urgency = energy_needed / (max(1e-6, time_remaining / max(self.times)))
        charging_urgency = min(1.0, charging_urgency)
        
        evs_competing = len([ev for ev in evs_present_now if ev != ev_id])
        competition_pressure = min(1.0, evs_competing / max(1, self.n_spots))
        
        current_available_spots = [s for s in range(self.n_spots) 
                                if s not in self.occupied_spots[self.current_time_idx]]
        current_available_chargers = [c for c in self.charger_ids 
                                    if c not in self.occupied_chargers[self.current_time_idx]]
        
        state = {
            "ev_features": ev_features,
            "spot_availability": available_spots_ratio,
            "charger_availability": available_chargers_ratio,
            "compatible_chargers_ratio": compatible_chargers_ratio if hasattr(self, 'ev_charger_compatible') else [1.0] * len(ev_time_indices),
            "relevant_prices": relevant_prices,
            "transformer_capacity": transformer_capacity_ratio,
            "time_indices": ev_time_indices,
            
            "avg_available_spots": avg_available_spots,
            "avg_available_chargers": avg_available_chargers,
            "avg_compatible_chargers": avg_compatible_chargers,
            "min_price": min_price,
            "avg_price": avg_price,
            "min_transformer_capacity": min_transformer_capacity,
            "avg_transformer_capacity": avg_transformer_capacity,
            
            "current_time_idx": self.current_time_idx,
            "current_time_normalized": self.current_time_idx / len(self.times),
            "system_demand_ratio": system_demand_ratio,
            "charging_urgency": charging_urgency,
            "time_remaining": time_remaining_from_now,
            "competition_pressure": competition_pressure,
            "evs_present_count": len(evs_present_now),
            "representative_ev": ev_id,
            
            "evs_present": evs_present_now,
            "available_spots": current_available_spots,
            "available_chargers": current_available_chargers,
            "occupied_spots": list(self.occupied_spots[self.current_time_idx]),
            "occupied_chargers": list(self.occupied_chargers[self.current_time_idx]),
            "current_price": self.normalized_prices[self.current_time_idx],
            "transformer_capacity_used": self.power_used[self.current_time_idx],
            "transformer_capacity_available": self.station_limit - self.power_used[self.current_time_idx],
        }
        
        ev_details = {}
        for present_ev_id in evs_present_now:
            energy_needed_ev = self.required_energy[present_ev_id] - self.energy_delivered[present_ev_id]
            energy_progress = self.energy_delivered[present_ev_id] / self.required_energy[present_ev_id]
            time_remaining_ev = (self.departure_time[present_ev_id] - current_time) / max(self.times)
            urgency = energy_needed_ev / max(1e-6, self.departure_time[present_ev_id] - current_time)
            
            current_assignment = None
            if hasattr(self, 'spot_assignments') and present_ev_id in self.spot_assignments:
                if self.current_time_idx in self.spot_assignments[present_ev_id]:
                    current_assignment = self.spot_assignments[present_ev_id][self.current_time_idx]
            
            is_charging = any(entry[0] == present_ev_id and entry[1] == self.current_time_idx 
                            for entry in self.charging_schedule)
            
            ev_detail = {
                "energy_needed": energy_needed_ev / self.max_required_energy,
                "energy_progress": energy_progress,
                "time_remaining": time_remaining_ev,
                "urgency": min(1.0, urgency),
                "current_spot": current_assignment,
                "is_charging": is_charging
            }
            
            if self.has_priority_info:
                ev_detail["priority"] = self.priority[present_ev_id] / self.max_priority
            if self.has_willingness_info:
                ev_detail["willingness_to_pay"] = self.willingness_to_pay[present_ev_id] / 1.5
            if self.has_battery_info:
                ev_detail["battery_capacity"] = self.battery_capacity[present_ev_id] / 100.0
            if hasattr(self, 'ev_charger_compatible'):
                compatible_chargers = self.ev_charger_compatible.get(present_ev_id, self.charger_ids)
                available_compatible = [c for c in compatible_chargers if c in current_available_chargers]
                ev_detail["compatible_chargers_available"] = len(available_compatible)
            
            ev_details[present_ev_id] = ev_detail
        
        state["ev_details"] = ev_details
        
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
        Builds a mapping of which vehicles are waiting for assignment at each interval.

        Returns:
            defaultdict[int, set]: A dictionary where keys are time indices and values are sets of EV IDs waiting at that time.
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
        Selects the most suitable vehicle to represent the current state.
        Prioritizes by urgency, then by priority, then by arrival order.

        Args:
            evs_present (list): A list of EV IDs currently present.

        Returns:
            str or None: The ID of the selected representative EV, or None if no EVs are present.
        """
        if not evs_present:
            return None
        
        current_time = self.times[self.current_time_idx]
        
        # Calculate scoring for each EV
        ev_scores = []
        for ev_id in evs_present:
            # Urgency factor (energy needed / time remaining)
            energy_needed = self.required_energy[ev_id] - self.energy_delivered[ev_id]
            time_remaining = max(1e-6, self.departure_time[ev_id] - current_time)
            urgency_score = energy_needed / time_remaining
            
            # Priority factor
            priority_score = self.priority.get(ev_id, 1) if self.has_priority_info else 1
            
            # Willingness to pay factor
            willingness_score = self.willingness_to_pay.get(ev_id, 1) if self.has_willingness_info else 1
            
            # Combined (normalized) score
            combined_score = (urgency_score * 0.5 + priority_score * 0.3 + willingness_score * 0.2)
            
            ev_scores.append((ev_id, combined_score))
        
        # Select the EV with the highest score
        ev_scores.sort(key=lambda x: x[1], reverse=True)
        return ev_scores[0][0]
    
    def _get_possible_actions(self, state):
        """
        Determines the possible actions given the current state.
        STRONGER CONSTRAINT: Only generates actions when there are actually vehicles and resources.
        OPTIMIZED: Drastically reduces the action space to speed up training.

        Args:
            state (dict or None): The current state of the environment.

        Returns:
            list: A list of dictionaries, where each dictionary represents a possible action.
        """
        
        if state is None:
            return []
        
        evs_present = state["evs_present"]
        available_spots = state["available_spots"]
        available_chargers = state["available_chargers"]
        
        if not evs_present:
            return [{"type": "no_action"}]
        
        actions = []
        
        actions.append({"type": "no_action"})
        
        for ev_id in evs_present:
            ev_detail = state["ev_details"][ev_id]
            
            if ev_detail["current_spot"] is None:
                for spot in available_spots:
                    actions.append({
                        "type": "assign_parking",
                        "ev_id": ev_id,
                        "spot": spot,
                        "charging": False
                    })
                    
                    for charger in available_chargers:
                        if hasattr(self, 'ev_charger_compatible'):
                            compatible_chargers = self.ev_charger_compatible.get(ev_id, self.charger_ids)
                            if charger not in compatible_chargers:
                                continue
                        
                        min_power = self.min_charge_rate[ev_id] if self.has_charge_rate_info else 3.5
                        max_power = min(self.max_charger_power_dict[charger],
                                    self.max_charge_rate[ev_id] if self.has_charge_rate_info else 50)
                        
                        if self.power_used[self.current_time_idx] + max_power <= self.station_limit:
                            actions.append({
                                "type": "assign_charging",
                                "ev_id": ev_id,
                                "spot": spot,
                                "charger": charger,
                                "power": max_power,
                                "charging": True
                            })
            
            else:
                if not ev_detail["is_charging"] and ev_detail["energy_progress"] < 0.95:
                    for charger in available_chargers:
                        if hasattr(self, 'ev_charger_compatible'):
                            compatible_chargers = self.ev_charger_compatible.get(ev_id, self.charger_ids)
                            if charger not in compatible_chargers:
                                continue
                        
                        max_power = min(self.max_charger_power_dict[charger],
                                    self.max_charge_rate[ev_id] if self.has_charge_rate_info else 50)
                        
                        if self.power_used[self.current_time_idx] + max_power <= self.station_limit:
                            actions.append({
                                "type": "start_charging",
                                "ev_id": ev_id,
                                "charger": charger,
                                "power": max_power
                            })
                
                if ev_detail["is_charging"] or ev_detail["energy_progress"] >= 0.95:
                    for new_spot in available_spots:
                        actions.append({
                            "type": "move_vehicle",
                            "ev_id": ev_id,
                            "new_spot": new_spot,
                            "charging": False
                        })
                
                if state["current_time_normalized"] >= 0.8 or ev_detail["energy_progress"] >= 0.95:
                    actions.append({
                        "type": "release_vehicle",
                        "ev_id": ev_id
                    })
        
        if len(evs_present) > 1:
            for i, ev1 in enumerate(evs_present):
                for ev2 in evs_present[i+1:]:
                    ev1_detail = state["ev_details"][ev1]
                    ev2_detail = state["ev_details"][ev2]
                    
                    if (ev1_detail["current_spot"] is not None and 
                        ev2_detail["current_spot"] is not None):
                        
                        urgency_diff = abs(ev1_detail["urgency"] - ev2_detail["urgency"])
                        if urgency_diff > 0.3:
                            actions.append({
                                "type": "swap_vehicles",
                                "ev1": ev1,
                                "ev2": ev2
                            })
        
        return actions


    def _check_action_feasibility(self, ev_id, spot, required_power, current_time_idx, time_indices, compatible_chargers):
        """
        Stricter feasibility check.
        Only allows actions that are economically and energetically sensible.

        Args:
            ev_id (int): The ID of the electric vehicle.
            spot (int): The parking spot ID.
            required_power (float): The power required for charging.
            current_time_idx (int): The current time index.
            time_indices (list): A list of relevant time indices.
            compatible_chargers (list): A list of chargers compatible with the EV.

        Returns:
            dict or None: The feasible action dictionary if an action is feasible, otherwise None.
        """
        # Check transformer capacity at the current time
        if self.power_used[current_time_idx] + required_power > self.station_limit:
            return None  # Exceeds transformer capacity

        # Find the best charger for this power at the current time
        suitable_chargers = [c for c in compatible_chargers
                                if (c not in self.occupied_chargers[current_time_idx] and
                                    self.max_charger_power_dict[c] >= required_power)]

        if not suitable_chargers:
            return None  # No suitable chargers available

        # Select the charger with the least power waste
        best_charger = min(suitable_chargers,
                            key=lambda c: self.max_charger_power_dict[c] - required_power)

        # ADDITIONAL VERIFICATION: Ensure the assignment makes economic/energetic sense
        energy_to_deliver = min(required_power * self.dt,
                                self.required_energy[ev_id] - self.energy_delivered[ev_id])

        if energy_to_deliver < 0.01:  # Less than 0.01 kWh
            return None  # Assignment is not worthwhile

        # EFFICIENCY VERIFICATION: Avoid assignments with too much power waste
        power_waste_ratio = (self.max_charger_power_dict[best_charger] - required_power) / self.max_charger_power_dict[best_charger]
        if power_waste_ratio > 0.8:  # More than 80% waste
            return None  # Too inefficient

        action = {
            "skip": False,
            "ev_id": ev_id,
            "spot": spot,
            "charger": best_charger,
            "power": required_power,
            "time_idx": current_time_idx,
            "duration": 1,  # Only for this time interval
            "energy_to_deliver": energy_to_deliver
        }

        return action

    def step(self, action_idx):
        """
        Executes an action and advances the environment to the next state.
        Corrected: Functions with consistent temporal approach.
        Optimized: Balanced rewards for better learning.

        Args:
            action_idx (int): The index of the action to execute from the list of possible actions.

        Returns:
            tuple: A tuple containing the new state, the reward, and a boolean indicating if the episode is done.
        """
        state = self._get_state()
        if state is None:
            return None, 0, True

        actions = self._get_possible_actions(state)

        if action_idx < 0 or action_idx >= len(actions):
            action_idx = 0

        action = actions[action_idx]
        reward = 0

        if action["type"] == "no_action":
            reward = self._calculate_no_action_reward(state)
        
        elif action["type"] == "assign_parking":
            reward = self._execute_assign_parking(action, state)
        
        elif action["type"] == "assign_charging":
            reward = self._execute_assign_charging(action, state)
        
        elif action["type"] == "start_charging":
            reward = self._execute_start_charging(action, state)
        
        elif action["type"] == "move_vehicle":
            reward = self._execute_move_vehicle(action, state)
        
        elif action["type"] == "release_vehicle":
            reward = self._execute_release_vehicle(action, state)
        
        elif action["type"] == "swap_vehicles":
            reward = self._execute_swap_vehicles(action, state)

        self.current_time_idx += 1

        return self._get_state(), reward, self.current_time_idx >= len(self.times)

    def _calculate_no_action_reward(self, state):
        evs_present = state["evs_present"]
        available_spots = state["available_spots"]
        
        if len(available_spots) == 0:
            return 0
        
        unassigned_count = sum(1 for ev_id in evs_present 
                            if state["ev_details"][ev_id]["current_spot"] is None)
        
        # Count assigned vehicles for occupancy calculation
        assigned_count = len(evs_present) - unassigned_count
        occupancy_rate = assigned_count / self.n_spots
        
        base_penalty = -10 * unassigned_count
        
        # OCCUPANCY PENALTIES - Force high utilization
        if occupancy_rate < 0.4:  # <40% occupancy = severe penalty
            occupancy_penalty = -(0.4 - occupancy_rate) * 400  # Up to -160
        elif occupancy_rate < 0.5:  # 40-50% = medium penalty  
            occupancy_penalty = -(0.5 - occupancy_rate) * 200  # Up to -20
        else:
            occupancy_penalty = 0
        
        # WASTE PENALTY - Having spots but unassigned urgent vehicles
        urgent_unassigned = sum(1 for ev_id in evs_present 
                            if (state["ev_details"][ev_id]["current_spot"] is None and 
                                state["ev_details"][ev_id]["urgency"] > 0.6))
        waste_penalty = -urgent_unassigned * 50
        
        return base_penalty + occupancy_penalty + waste_penalty

    def _execute_assign_parking(self, action, state):
        ev_id = action["ev_id"]
        spot = action["spot"]
        
        if spot not in state["available_spots"]:
            return -50
        
        if ev_id not in self.spot_assignments:
            self.spot_assignments[ev_id] = {}
        self.spot_assignments[ev_id][self.current_time_idx] = spot
        self.occupied_spots[self.current_time_idx].add(spot)
        
        base_reward = self.REWARD_FOR_ASSIGNED_VEHICLE * 0.3
        
        ev_detail = state["ev_details"][ev_id]
        if self.has_priority_info and "priority" in ev_detail:
            base_reward *= (0.5 + 0.5 * ev_detail["priority"])
        
        # GLOBAL OCCUPANCY REWARDS
        evs_present = state["evs_present"]
        assigned_count = sum(1 for ev in evs_present 
                            if (ev == ev_id or state["ev_details"][ev]["current_spot"] is not None))
        occupancy_rate = assigned_count / self.n_spots
        
        if occupancy_rate >= 0.8:  # 80%+ occupancy
            occupancy_bonus = 100 + (occupancy_rate - 0.8) * 200
        elif occupancy_rate >= 0.5:  # 50-80% occupancy  
            occupancy_bonus = (occupancy_rate - 0.5) * 100
        else:  # <50% occupancy
            occupancy_bonus = -(0.5 - occupancy_rate) * 300
        
        # EFFICIENCY BONUS for serving urgent vehicles
        urgency_bonus = 0
        if ev_detail["urgency"] > 0.7:
            urgency_bonus = 25
        
        return base_reward + occupancy_bonus + urgency_bonus

    def _execute_assign_charging(self, action, state):
        ev_id = action["ev_id"]
        spot = action["spot"]
        charger = action["charger"]
        power = action["power"]
        
        if spot not in state["available_spots"] or charger not in state["available_chargers"]:
            return -50
        
        if self.power_used[self.current_time_idx] + power > self.station_limit:
            return -100
        
        if ev_id not in self.spot_assignments:
            self.spot_assignments[ev_id] = {}
        self.spot_assignments[ev_id][self.current_time_idx] = spot
        
        self.charging_schedule.append((ev_id, self.current_time_idx, charger, spot, power))
        self.occupied_spots[self.current_time_idx].add(spot)
        self.occupied_chargers[self.current_time_idx].add(charger)
        self.power_used[self.current_time_idx] += power
        
        efficiency = self.efficiency[ev_id] if self.has_efficiency_info else 0.9
        charger_eff = 0.95
        energy_delivered = power * self.dt * efficiency * charger_eff
        
        remaining_needed = max(0, self.required_energy[ev_id] - self.energy_delivered[ev_id])
        actual_energy = min(energy_delivered, remaining_needed)
        self.energy_delivered[ev_id] += actual_energy
        
        current_price = self.prices[self.current_time_idx]
        energy_cost = actual_energy * current_price
        
        satisfaction_reward = self.ENERGY_SATISFACTION_WEIGHT * (actual_energy / self.max_required_energy) * 100
        parking_reward = self.REWARD_FOR_ASSIGNED_VEHICLE * 0.5
        cost_penalty = -self.ENERGY_COST_WEIGHT * energy_cost
        
        ev_detail = state["ev_details"][ev_id]
        urgency_bonus = ev_detail["urgency"] * 30
        
        priority_multiplier = 1.0
        if self.has_priority_info and "priority" in ev_detail:
            priority_multiplier = 0.5 + 0.5 * ev_detail["priority"]
        
        base_total = (satisfaction_reward + parking_reward + urgency_bonus) * priority_multiplier + cost_penalty
        
        # GLOBAL OCCUPANCY AND EFFICIENCY REWARDS
        evs_present = state["evs_present"]
        assigned_spots = sum(1 for ev in evs_present 
                            if (ev == ev_id or state["ev_details"][ev]["current_spot"] is not None))
        
        # Current charger usage after this action
        occupied_chargers_count = len(state["occupied_chargers"]) + 1
        charger_utilization = occupied_chargers_count / len(self.charger_ids)
        
        spot_occupancy = assigned_spots / self.n_spots
        
        # OCCUPANCY BONUS
        if spot_occupancy >= 0.8:
            occupancy_bonus = 120 + (spot_occupancy - 0.8) * 200
        elif spot_occupancy >= 0.5:
            occupancy_bonus = (spot_occupancy - 0.5) * 120
        else:
            occupancy_bonus = -(0.5 - spot_occupancy) * 300
        
        # CHARGER EFFICIENCY BONUS
        if charger_utilization >= 0.8:
            charger_bonus = 50
        elif charger_utilization >= 0.5:
            charger_bonus = 25
        else:
            charger_bonus = 0
        
        total_reward = base_total + occupancy_bonus + charger_bonus
        
        satisfaction_ratio = self.energy_delivered[ev_id] / self.required_energy[ev_id]
        if satisfaction_ratio >= 0.99:
            self.evs_processed.add(ev_id)
        
        return total_reward

    def _execute_start_charging(self, action, state):
        ev_id = action["ev_id"]
        charger = action["charger"]
        power = action["power"]
        
        if charger not in state["available_chargers"]:
            return -50
        
        if self.power_used[self.current_time_idx] + power > self.station_limit:
            return -100
        
        current_spot = state["ev_details"][ev_id]["current_spot"]
        if current_spot is None:
            return -75
        
        self.charging_schedule.append((ev_id, self.current_time_idx, charger, current_spot, power))
        self.occupied_chargers[self.current_time_idx].add(charger)
        self.power_used[self.current_time_idx] += power
        
        efficiency = self.efficiency[ev_id] if self.has_efficiency_info else 0.9
        charger_eff = 0.95
        energy_delivered = power * self.dt * efficiency * charger_eff
        
        remaining_needed = max(0, self.required_energy[ev_id] - self.energy_delivered[ev_id])
        actual_energy = min(energy_delivered, remaining_needed)
        self.energy_delivered[ev_id] += actual_energy
        
        current_price = self.prices[self.current_time_idx]
        energy_cost = actual_energy * current_price
        
        satisfaction_reward = self.ENERGY_SATISFACTION_WEIGHT * (actual_energy / self.max_required_energy) * 100
        cost_penalty = -self.ENERGY_COST_WEIGHT * energy_cost
        
        ev_detail = state["ev_details"][ev_id]
        urgency_bonus = ev_detail["urgency"] * 30
        
        priority_multiplier = 1.0
        if self.has_priority_info and "priority" in ev_detail:
            priority_multiplier = 0.5 + 0.5 * ev_detail["priority"]
        
        base_reward = (satisfaction_reward + urgency_bonus) * priority_multiplier + cost_penalty
        
        # CHARGER UTILIZATION BONUS
        occupied_chargers_count = len(state["occupied_chargers"]) + 1
        charger_utilization = occupied_chargers_count / len(self.charger_ids)
        
        if charger_utilization >= 0.8:
            charger_bonus = 40
        elif charger_utilization >= 0.6:
            charger_bonus = 20
        else:
            charger_bonus = 0
        
        total_reward = base_reward + charger_bonus
        
        satisfaction_ratio = self.energy_delivered[ev_id] / self.required_energy[ev_id]
        if satisfaction_ratio >= 0.99:
            self.evs_processed.add(ev_id)
        
        return total_reward

    def _execute_move_vehicle(self, action, state):
        ev_id = action["ev_id"]
        new_spot = action["new_spot"]
        
        if new_spot not in state["available_spots"]:
            return -30
        
        current_spot = state["ev_details"][ev_id]["current_spot"]
        if current_spot is None:
            return -40
        
        self.occupied_spots[self.current_time_idx].discard(current_spot)
        self.occupied_spots[self.current_time_idx].add(new_spot)
        self.spot_assignments[ev_id][self.current_time_idx] = new_spot
        
        ev_detail = state["ev_details"][ev_id]
        if ev_detail["energy_progress"] >= 0.95:
            return 20
        else:
            return 5

    def _execute_release_vehicle(self, action, state):
        ev_id = action["ev_id"]
        
        current_spot = state["ev_details"][ev_id]["current_spot"]
        if current_spot is not None:
            self.occupied_spots[self.current_time_idx].discard(current_spot)
        
        self.evs_processed.add(ev_id)
        
        ev_detail = state["ev_details"][ev_id]
        satisfaction_bonus = ev_detail["energy_progress"] * 50
        early_release_penalty = -10 if ev_detail["energy_progress"] < 0.8 else 0
        
        return satisfaction_bonus + early_release_penalty

    def _execute_swap_vehicles(self, action, state):
        ev1 = action["ev1"]
        ev2 = action["ev2"]
        
        ev1_detail = state["ev_details"][ev1]
        ev2_detail = state["ev_details"][ev2]
        
        spot1 = ev1_detail["current_spot"]
        spot2 = ev2_detail["current_spot"]
        
        if spot1 is None or spot2 is None:
            return -60
        
        self.spot_assignments[ev1][self.current_time_idx] = spot2
        self.spot_assignments[ev2][self.current_time_idx] = spot1
        
        urgency_improvement = abs(ev1_detail["urgency"] - ev2_detail["urgency"]) * 25
        
        if ev1_detail["urgency"] > ev2_detail["urgency"]:
            return urgency_improvement
        else:
            return -urgency_improvement * 0.5















    def _execute_immediate_action(self, action, ev_id, state):
        """
        Executes an immediate action (only for the current time).
        New: Handles the simplified action structure.

        Args:
            action (dict): The action to execute.
            ev_id (int): The ID of the electric vehicle.
            state (dict): The current state of the environment.

        Returns:
            float: The reward obtained from executing the action.
        """
        spot = action["spot"]
        charger = action["charger"]
        power = action["power"]
        time_idx = action["time_idx"]
        energy_to_deliver = action["energy_to_deliver"]

        # Register assignment for this interval
        self.charging_schedule.append((ev_id, time_idx, charger, spot, power))
        self.occupied_spots[time_idx].add(spot)
        self.occupied_chargers[time_idx].add(charger)
        self.power_used[time_idx] += power

        # Calculate effective energy considering efficiencies
        efficiency = self.efficiency[ev_id] if self.has_efficiency_info else 0.9
        charger_eff = 0.95  # Charger efficiency
        effective_energy = energy_to_deliver * efficiency * charger_eff

        # Update delivered energy
        remaining_needed = max(0, self.required_energy[ev_id] - self.energy_delivered[ev_id])
        actual_energy = min(effective_energy, remaining_needed)
        self.energy_delivered[ev_id] += actual_energy

        # Calculate cost
        current_price = self.prices[time_idx]
        energy_cost = actual_energy * current_price

        # Calculate reward
        reward = self._calculate_assignment_reward(ev_id, actual_energy, energy_cost, power, state)

        # Check if the vehicle is fully charged
        satisfaction_ratio = self.energy_delivered[ev_id] / self.required_energy[ev_id]
        if satisfaction_ratio >= 0.99:  # 99% charged
            self.evs_processed.add(ev_id)
            #print(f"EV {ev_id} completely charged ({satisfaction_ratio:.1%})")

        # Advance to the next interval
        self.current_time_idx += 1

        return reward

    def _execute_legacy_action(self, action, ev_id, state):
        """
        Executes an action with a legacy structure (charging_profile).
        Compatibility: To maintain functionality with existing code.

        Args:
            action (dict): The action to execute, containing a 'charging_profile'.
            ev_id (int): The ID of the electric vehicle.
            state (dict): The current state of the environment.

        Returns:
            float: The reward obtained from executing the action.
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

        # Mark vehicle as processed if complete
        if energy_satisfaction_ratio >= 0.99:
            self.evs_processed.add(ev_id)

        # With legacy approach, we advance several intervals
        self.current_time_idx = max(charging_profile.keys()) + 1 if charging_profile else self.current_time_idx + 1

        return reward

    def _calculate_assignment_reward(self, ev_id, actual_energy, energy_cost, power, state):
        """
        Calculates the reward for a successful assignment with the new approach.

        Args:
            ev_id (int): The ID of the electric vehicle.
            actual_energy (float): The actual energy delivered.
            energy_cost (float): The cost incurred for the delivered energy.
            power (float): The power used for charging.
            state (dict): The current state of the environment.

        Returns:
            float: The calculated total reward.
        """
        # Base reward for energy delivered
        energy_satisfaction = min(1.0, self.energy_delivered[ev_id] / self.required_energy[ev_id])
        satisfaction_reward = self.ENERGY_SATISFACTION_WEIGHT * energy_satisfaction * 100

        # Bonus for completing the charge
        if energy_satisfaction >= 0.95:
            satisfaction_reward += self.REWARD_FOR_ASSIGNED_VEHICLE

        # Penalty for cost (normalized)
        price_penalty = -self.ENERGY_COST_WEIGHT * energy_cost

        # Efficiency bonus (use appropriate power)
        efficiency_bonus = 10 if actual_energy > 0 else 0

        # Priority factor
        priority_multiplier = 1.0
        if self.has_priority_info:
            priority_multiplier = 0.5 + 0.5 * (self.priority[ev_id] / self.max_priority)

        # Urgency factor (bonus for serving urgent vehicles)
        urgency_bonus = 0
        if "charging_urgency" in state:
            urgency_bonus = state["charging_urgency"] * 20

        # Competition factor (bonus for serving when there is high demand)
        competition_bonus = 0
        if "competition_pressure" in state:
            competition_bonus = state["competition_pressure"] * 15

        total_reward = ((satisfaction_reward + efficiency_bonus + urgency_bonus + competition_bonus) *
                        priority_multiplier + price_penalty)

        return total_reward

    def reset(self):
        """
        Resets the environment to its initial state.
        Corrected: Consistent initialization with the temporal approach.

        Returns:
            dict: The initial state of the environment.
        """
        self.current_time_idx = 0
        self.evs_processed = set()
        self.spot_assignments = {}
        self.charging_schedule = []
        self.occupied_spots = {t: set() for t in range(len(self.times))}
        self.occupied_chargers = {t: set() for t in range(len(self.times))}
        self.power_used = {t: 0 for t in range(len(self.times))}
        self.energy_delivered = {ev_id: 0 for ev_id in self.ev_ids}
        self.evs_present_at_time = self._build_ev_presence_map()
        self.evs_waiting_assignment = {t: set() for t in range(len(self.times))}
        self.evs_assigned = set()
        self.evs_waiting_by_time = self._build_waiting_schedule()
        
        for t in range(len(self.times)):
            for ev_id in self.evs_present_at_time[t]:
                if ev_id not in self.evs_assigned:
                    self.evs_waiting_assignment[t].add(ev_id)
        
        return self._get_state()

    def _get_possible_actions_legacy(self, state):
        """
        LEGACY VERSION for compatibility with existing code that expects charging_profile.
        This version maintains the original structure but with temporal corrections.

        Args:
            state (dict): The current state of the environment.

        Returns:
            list: A list of possible actions.
        """
        if state is None:
            return []

        if "representative_ev" not in state:
            return [{"skip": True}]

        ev_id = state["representative_ev"]
        time_indices = state["time_indices"]

        # Check if the vehicle still needs energy
        energy_needed = self.required_energy[ev_id] - self.energy_delivered[ev_id]
        if energy_needed <= 0.01:
            return [{"skip": True}]

        # Determine which slots are available during the ENTIRE future period
        consistent_spots = list(range(self.n_spots))
        for t in time_indices:
            consistent_spots = [s for s in consistent_spots if s not in self.occupied_spots[t]]

        actions = []

        # Skip option
        skip_action = {"skip": True}

        # Vehicle-specific constraints
        min_power = self.min_charge_rate[ev_id] if self.has_charge_rate_info else 3.5
        max_power = self.max_charge_rate[ev_id] if self.has_charge_rate_info else 50

        # Calculate ideal required power (energy/available time)
        available_time = len(time_indices) * self.dt

        if available_time <= 0:
            return [skip_action]  # No time, we can only skip

        ideal_power = energy_needed / available_time

        # OPTIMIZATION: Limit to a maximum of 3 spots to reduce complexity
        spots_to_try = consistent_spots[:3] if len(consistent_spots) > 3 else consistent_spots

        # For each available spot, generate possible actions
        for spot in spots_to_try:
            # OPTIMIZATION: Only 3 power levels instead of many
            power_levels = []

            # Determine compatible chargers
            if hasattr(self, 'ev_charger_compatible'):
                compatible_chargers = self.ev_charger_compatible.get(ev_id, self.charger_ids)
            else:
                compatible_chargers = self.charger_ids

            available_powers = sorted(set(self.max_charger_power_dict[c]
                                            for c in compatible_chargers))

            # Only use 3 levels: minimum, middle, maximum
            if len(available_powers) >= 3:
                power_levels = [available_powers[0], available_powers[len(available_powers)//2], available_powers[-1]]
            else:
                power_levels = available_powers

            # Add ideal power if within range
            if min_power <= ideal_power <= max_power:
                power_levels.append(ideal_power)

            # Remove duplicates and filter by range
            power_levels = sorted(set(p for p in power_levels if min_power <= p <= max_power))

            for required_power in power_levels:
                # Check if there is enough charger and transformer capacity
                feasible_charging = True
                charger_assignments = {}

                for t in time_indices:
                    # Look for a compatible and available charger with sufficient power
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

                    # Assign the charger with the most suitable power (least waste)
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

        # If no feasible charging actions, only add the skip option
        if not actions:
            actions.append(skip_action)
        else:
            # Add the skip option at the end
            actions.append(skip_action)

        return actions

    def get_schedule(self):
        """Returns the complete generated charging schedule."""
        return self.charging_schedule

    def get_energy_satisfaction_metrics(self):
        """Returns detailed metrics on energy satisfaction."""
        metrics = {}

        # Total required and delivered energy
        total_required = sum(self.required_energy.values())
        total_delivered = sum(self.energy_delivered.values())

        # Total percentage of satisfaction
        total_satisfaction_pct = (total_delivered / total_required) * 100 if total_required > 0 else 100

        metrics["total_required_energy"] = total_required
        metrics["total_delivered_energy"] = total_delivered
        metrics["total_satisfaction_pct"] = total_satisfaction_pct

        # Metrics per vehicle
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

        # Metrics grouped by priority if available
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
        """
        Updates reward weights dynamically.

        Args:
            weights_dict (Dict[str, float]): A dictionary containing the new weights for reward components.
        """
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