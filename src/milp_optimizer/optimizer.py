import pulp
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus
import numpy as np
import time

class EVChargingMILP:
    """
    Multi-objective MILP optimizer for electric vehicle charging scheduling.
    Considers advanced compatibility, efficiency, and customer satisfaction constraints.
    """
    def __init__(self, config):
        """
        Initializes the EVChargingMILP with a given configuration.

        Args:
            config (dict): A dictionary containing all necessary configuration parameters
                           for the charging optimization.
        """
        self.config = config
        self.process_config()
        
    def process_config(self):
        """Processes the configuration to extract necessary parameters."""
        data = self.config
        
        # Basic parameters
        self.times = data["times"]
        self.prices = data["prices"]
        self.arrivals = data["arrivals"]
        self.chargers = data["parking_config"]["chargers"]
        self.station_limit = data["parking_config"]["transformer_limit"]
        self.dt = data.get("dt", 0.25)  # Default to 15 minutes
        self.n_spots = data["parking_config"]["n_spots"]
        
        # Indices
        self.T = range(len(self.times))
        self.EVs = [arr["id"] for arr in self.arrivals]
        
        # Vehicle data mapping
        self.arrival_time = {arr["id"]: arr["arrival_time"] for arr in self.arrivals}
        self.departure_time = {arr["id"]: arr["departure_time"] for arr in self.arrivals}
        self.required_energy = {arr["id"]: arr["required_energy"] for arr in self.arrivals}
        
        # New advanced parameters
        self.min_charge_rate = {}
        self.max_charge_rate = {}
        self.ac_charge_rate = {}
        self.dc_charge_rate = {}
        self.priority = {}
        self.willingness_to_pay = {}
        self.efficiency = {}
        self.battery_capacity = {}
        self.brand = {}
        
        for arr in self.arrivals:
            ev_id = arr["id"]
            self.min_charge_rate[ev_id] = arr.get("min_charge_rate", 3.5)  # Default 3.5kW
            self.max_charge_rate[ev_id] = arr.get("max_charge_rate", 50)  # Default 50kW
            self.ac_charge_rate[ev_id] = arr.get("ac_charge_rate", 7)     # Default 7kW AC
            self.dc_charge_rate[ev_id] = arr.get("dc_charge_rate", 50)    # Default 50kW DC
            self.priority[ev_id] = arr.get("priority", 1)                 # Default priority 1
            self.willingness_to_pay[ev_id] = arr.get("willingness_to_pay", 1.0)  # Default 1.0
            self.efficiency[ev_id] = arr.get("efficiency", 0.9)          # Default 90% efficiency
            
            if "battery_capacity" in arr:
                self.battery_capacity[ev_id] = arr["battery_capacity"]
            if "brand" in arr:
                self.brand[ev_id] = arr["brand"]
        
        # Charger information
        self.charger_ids = [c["charger_id"] for c in self.chargers]
        self.max_charger_power = {c["charger_id"]: c["power"] for c in self.chargers}
        self.charger_type = {c["charger_id"]: c.get("type", "AC") for c in self.chargers}
        self.charger_efficiency = {c["charger_id"]: c.get("efficiency", 0.9) for c in self.chargers}
        
        # Charger-vehicle compatibility
        self.compatible_vehicles = {}
        for c in self.chargers:
            charger_id = c["charger_id"]
            if "compatible_vehicles" in c:
                self.compatible_vehicles[charger_id] = c["compatible_vehicles"]
            else:
                # By default, all chargers are compatible with all vehicles
                self.compatible_vehicles[charger_id] = [b for b in self.brand.values()]
        
        # Compatibility mapping
        self.ev_charger_compatible = {}
        for ev_id in self.EVs:
            self.ev_charger_compatible[ev_id] = []
            ev_brand = self.brand.get(ev_id, "")
            
            for charger_id in self.charger_ids:
                # Check if the charger is compatible with this brand
                compatible = True
                if ev_brand and self.compatible_vehicles.get(charger_id):
                    # Extract base brand without details
                    base_brand = ev_brand.split()[0] if " " in ev_brand else ev_brand
                    compatible = False
                    for comp_brand in self.compatible_vehicles[charger_id]:
                        if comp_brand in ev_brand or base_brand in comp_brand:
                            compatible = True
                            break
                
                if compatible:
                    # Check power limits
                    charger_power = self.max_charger_power[charger_id]
                    charger_type = self.charger_type[charger_id]
                    
                    # If AC charger, check AC charge rate
                    if charger_type == "AC" and charger_power <= self.ac_charge_rate.get(ev_id, 7):
                        self.ev_charger_compatible[ev_id].append(charger_id)
                    # If DC charger, check DC charge rate
                    elif charger_type == "DC" and charger_power <= self.dc_charge_rate.get(ev_id, 50):
                        self.ev_charger_compatible[ev_id].append(charger_id)
    
    def solve(self, penalty_unmet=1000.0, rl_schedule=None, time_limit=None,
              epsilon_satisfaction=None, return_infeasible=False):
        """
        Solves the charging scheduling problem with MILP using the epsilon-constraint method.
        The primary objective is to minimize cost. Customer satisfaction is converted into a constraint.
        
        Args:
            penalty_unmet (float): Penalty for unmet energy demand.
            rl_schedule (list): Initial solution (from RL) for warm start.
            time_limit (int): Time limit for solving MILP (seconds).
            epsilon_satisfaction (float): Minimum weighted satisfaction threshold that the model must meet (between 0 and 1).
                                          If None, the original weighted mode is used.
            return_infeasible (bool): If True, returns the best possible solution even if it is infeasible.
            
        Returns:
            model (LpProblem): Solved MILP model.
            schedule (dict): Dictionary with optimized solution.
            rejected_details (dict): Information about unsatisfied EVs.
            obj_values (dict): Final values of the objective functions (cost and satisfaction).
        """
        # Penalty coefficients for slacks
        M_slack = 1e4
        
        # Create MILP problem
        model = LpProblem("EV_Charging_Epsilon_Constraint", LpMinimize)
        
        # Main decision variables
        x = {}  # Power allocated to each EV, period, charger
        y = {}  # Binary variable for charger connection
        for i in self.EVs:
            for t in self.T:
                # Only consider periods when the EV is present
                if self.arrival_time[i] <= self.times[t] < self.departure_time[i]:
                    # Only consider compatible chargers
                    for c in self.ev_charger_compatible.get(i, self.charger_ids):
                        x[(i, t, c)] = LpVariable(f"x_{i}_{t}_{c}", lowBound=0)
                        y[(i, t, c)] = LpVariable(f"y_{i}_{t}_{c}", cat="Binary")
                        
                        # Initialization with RL solution if provided
                        if rl_schedule is not None:
                            for entry in rl_schedule:
                                if entry[0] == i and entry[1] == t and entry[2] == c:
                                    x[(i, t, c)].setInitialValue(entry[4])
                                    y[(i, t, c)].setInitialValue(1)
                                    break
        
        # Variable for energy not covered per EV
        u = {i: LpVariable(f"u_{i}", lowBound=0) for i in self.EVs}
        
        # Variable for weighted customer satisfaction (objective 2)
        satisfaction = {i: LpVariable(f"satisfaction_{i}", lowBound=0, upBound=1) for i in self.EVs}
        
        # Binary variable: EV parked in interval t
        z = {}
        for i in self.EVs:
            for t in self.T:
                if self.arrival_time[i] <= self.times[t] < self.departure_time[i]:
                    z[(i, t)] = LpVariable(f"z_{i}_{t}", cat="Binary")
                else:
                    z[(i, t)] = 0
        
        # Binary variables: slot assignment in each interval for each EV
        w = {}
        for i in self.EVs:
            for t in self.T:
                if self.arrival_time[i] <= self.times[t] < self.departure_time[i]:
                    for s in range(self.n_spots):
                        w[(i, t, s)] = LpVariable(f"w_{i}_{t}_{s}", cat="Binary")
        
        # Slack variables
        s_charger = {}      # For each charger's capacity
        s_transformer = {}  # For transformer limit
        s_parking = {}      # For parking capacity
        s_slot = {}         # For slot assignment
        s_min_charge = {}   # For minimum charge rate
        
        for t in self.T:
            for c in self.charger_ids:
                s_charger[(t, c)] = LpVariable(f"s_charger_{t}_{c}", lowBound=0)
            s_transformer[t] = LpVariable(f"s_transformer_{t}", lowBound=0)
            s_parking[t] = LpVariable(f"s_parking_{t}", lowBound=0)
        
        for i in self.EVs:
            for t in self.T:
                if self.arrival_time[i] <= self.times[t] < self.departure_time[i]:
                    s_slot[(i, t)] = LpVariable(f"s_slot_{i}_{t}", lowBound=0)
                    s_min_charge[(i, t)] = LpVariable(f"s_min_charge_{i}_{t}", lowBound=0)
        
        # MAIN OBJECTIVE: Minimize economic cost
        cost_obj = (
            lpSum(self.prices[t] * self.dt * x[(i, t, c)]
                  for i in self.EVs for t in self.T for c in self.charger_ids
                  if (i, t, c) in x)
            + penalty_unmet * lpSum(u[i] for i in self.EVs)
        )
        model += cost_obj + M_slack * (
            lpSum(s_charger[(t, c)] for t in self.T for c in self.charger_ids)
            + lpSum(s_transformer[t] for t in self.T)
            + lpSum(s_parking[t] for t in self.T)
            + lpSum(s_slot[(i, t)] for i in self.EVs for t in self.T 
                    if self.arrival_time[i] <= self.times[t] < self.departure_time[i])
            + lpSum(s_min_charge[(i, t)] for i in self.EVs for t in self.T 
                    if self.arrival_time[i] <= self.times[t] < self.departure_time[i])
        ), "Total_Objective_Cost"
        
        # NEW: Customer satisfaction constraint (epsilon-constraint)
        if epsilon_satisfaction is not None:
            # Validate epsilon_satisfaction
            assert 0 <= epsilon_satisfaction <= 1, "epsilon_satisfaction must be between 0 and 1"
            
            # Sum of weighted satisfaction of all EVs
            total_weighted_satisfaction = lpSum(self.priority[i] * self.willingness_to_pay[i] * satisfaction[i] 
                                                 for i in self.EVs)
            
            # Calculate the maximum possible weighted satisfaction to normalize epsilon
            # This assumes that maximum weighted satisfaction is achieved when all EVs have satisfaction 1
            max_possible_weighted_satisfaction = lpSum(self.priority[i] * self.willingness_to_pay[i] 
                                                         for i in self.EVs)
            
            # The sum of weighted satisfaction must be greater than or equal to epsilon * max_possible_weighted_satisfaction
            # Only add the constraint if there are EVs and if the maximum possible satisfaction is > 0
            if pulp.value(max_possible_weighted_satisfaction) is not None and pulp.value(max_possible_weighted_satisfaction) > 1e-6: # Avoid division by zero or insignificant thresholds
                model += (total_weighted_satisfaction >= epsilon_satisfaction * max_possible_weighted_satisfaction,
                          "Epsilon_Satisfaction_Constraint")
            else:
                print("Warning: No EVs or maximum possible satisfaction is zero. Epsilon-constraint not applied for satisfaction.")
        else:
            print("Epsilon-constraint for satisfaction not activated. Using pure cost optimization mode.")

        # Constraint 1: Delivered charge + unmet energy = required energy
        for i in self.EVs:
            model += (
                lpSum(x[(i, t, c)] * self.dt 
                      for t in self.T for c in self.charger_ids 
                      if (i, t, c) in x)
                + u[i]
                == self.required_energy[i]
            ), f"Energy_Balance_EV_{i}"
        
        # Constraint 2: Capacity of each charger (with slack)
        for t in self.T:
            for c in self.charger_ids:
                model += (
                    lpSum(x[(i, t, c)] for i in self.EVs if (i, t, c) in x) 
                    <= self.max_charger_power[c] + s_charger[(t, c)]
                ), f"ChargerCap_t{t}_c{c}"
        
        # Constraint 3: Transformer limit (with slack)
        for t in self.T:
            model += (
                lpSum(x[(i, t, c)] for i in self.EVs for c in self.charger_ids if (i, t, c) in x) 
                <= self.station_limit + s_transformer[t]
            ), f"StationCap_t{t}"
        
        # Constraint 4: Do not charge outside the availability window
        for i in self.EVs:
            for t in self.T:
                if not (self.arrival_time[i] <= self.times[t] < self.departure_time[i]):
                    for c in self.charger_ids:
                        if (i, t, c) in x:
                            model += x[(i, t, c)] == 0, f"NoChargeOutside_{i}_{t}_{c}"
        
        # Constraint 5: Unique charger assignment in each interval
        for i in self.EVs:
            for t in self.T:
                if self.arrival_time[i] <= self.times[t] < self.departure_time[i]:
                    model += (
                        lpSum(y[(i, t, c)] for c in self.charger_ids if (i, t, c) in y) <= 1
                    ), f"UniqueCharger_EV_{i}_t{t}"
        
        # Constraint 6: Linking between x and y
        for i in self.EVs:
            for t in self.T:
                for c in self.charger_ids:
                    if (i, t, c) in x:
                        model += (
                            x[(i, t, c)] <= y[(i, t, c)] * self.max_charger_power[c]
                        ), f"Link_x_y_EV_{i}_t{t}_c{c}"
                        
                        model += (
                            x[(i, t, c)] >= y[(i, t, c)] * self.min_charge_rate[i] - s_min_charge[(i, t)]
                        ), f"MinCharge_EV_{i}_t{t}_c{c}"
        
        for i in self.EVs:
            for t in self.T:
                if isinstance(z[(i, t)], LpVariable):
                    compatible_chargers = [c for c in self.charger_ids if (i, t, c) in y]
                    if compatible_chargers:
                        model += (
                            z[(i, t)] >= lpSum(y[(i, t, c)] for c in compatible_chargers) / len(compatible_chargers)
                        ), f"Link_z_y_EV_{i}_t{t}"
        
        for t in self.T:
            active_evs = [i for i in self.EVs if self.arrival_time[i] <= self.times[t] < self.departure_time[i]]
            if active_evs:
                model += (
                    lpSum(z[(i, t)] for i in active_evs if isinstance(z[(i, t)], LpVariable)) 
                    <= self.n_spots + s_parking[t]
                ), f"ParkingCap_t{t}"
        
        for i in self.EVs:
            for t in self.T:
                if self.arrival_time[i] <= self.times[t] < self.departure_time[i]:
                    model += (
                        lpSum(w[(i, t, s)] for s in range(self.n_spots) if (i, t, s) in w) + s_slot[(i, t)] == 1
                    ), f"AssignSlot_EV_{i}_t{t}"
        
        for t in self.T:
            for s in range(self.n_spots):
                evs_in_interval = [i for i in self.EVs if self.arrival_time[i] <= self.times[t] < self.departure_time[i]]
                if evs_in_interval:
                    slots_for_time_spot = [(i, t, s) for i in evs_in_interval if (i, t, s) in w]
                    if slots_for_time_spot:
                        model += (
                            lpSum(w[(i, t, s)] for i in evs_in_interval if (i, t, s) in w) <= 1
                        ), f"SlotUnique_t{t}_s{s}"
        
        for i in self.EVs:
            model += (
                satisfaction[i] * self.required_energy[i] 
                == lpSum(x[(i, t, c)] * self.dt for t in self.T for c in self.charger_ids if (i, t, c) in x)
            ), f"Satisfaction_EV_{i}"
        
        for i in self.EVs:
            model += satisfaction[i] <= 1, f"Max_Satisfaction_{i}"
        

        for i in self.EVs:
            for t in self.T:
                for c in self.charger_ids:
                    if (i, t, c) in x:

                        combined_efficiency = self.efficiency[i] * self.charger_efficiency[c]

        
        # Configure time limit and solver
        if time_limit:
            print(f"Configuring MILP with time limit: {time_limit} seconds and a gap of 0.01 (1%)")
            solver = pulp.PULP_CBC_CMD(msg=True, options=[
                f"sec {time_limit}", 
                "timeMode elapsed", 
                "ratioGap 0.01"
            ])
        else:
            solver = pulp.PULP_CBC_CMD(msg=True, options=["ratioGap 0.01"])
        
        # Solve
        start_time_model = time.time()
        try:
            status = model.solve(solver)
            solve_time = time.time() - start_time_model
            
            print("Status:", LpStatus[model.status])
            print("Objective value:", model.objective.value())
            print("MILP solving time:", solve_time, "seconds")
            
            if LpStatus[model.status] not in ("Optimal", "Not Solved"):  # Accept Optimal or if it couldn't be solved (e.g., timeout)
                if not return_infeasible:
                    print("The model could not be solved optimally.")
                    if rl_schedule is not None:
                        print("Returning original RL solution")
                        return None, self._convert_rl_schedule_to_dict(rl_schedule), {}, {}
                    else:
                        return None, {}, {}, {}
                    
        except Exception as e:
            print(f"Error solving MILP: {e}")
            print("Returning original RL solution")
            if rl_schedule is not None:
                return None, self._convert_rl_schedule_to_dict(rl_schedule), {}, {}
            else:
                return None, {}, {}, {}
        
        # Extract solution
        schedule = self._extract_solution(model, x, w)
        
        # Calculate separate objective values
        obj_values = self._calculate_objective_values(model, schedule, satisfaction)
        
        # Generate report of partially or unsatisfied EVs
        rejected_details = self._generate_rejection_report(model, u, satisfaction)
        
        return model, schedule, rejected_details, obj_values
        
    def _extract_solution(self, model, x, w):
        """
        Extracts the solution from the MILP model.

        Args:
            model (LpProblem): The solved PuLP MILP model.
            x (dict): Dictionary of LpVariable for power allocation.
            w (dict): Dictionary of LpVariable for slot assignment.

        Returns:
            dict: A dictionary representing the optimized charging schedule.
        """
        schedule = {}
        for i in self.EVs:
            schedule[i] = []
            
            for t in self.T:
                if not (self.arrival_time[i] <= self.times[t] < self.departure_time[i]):
                    continue
                    
                for c in self.charger_ids:
                    if (i, t, c) not in x:
                        continue
                        
                    var_val = x[(i, t, c)].varValue
                    if var_val is not None and var_val > 1e-4:
                        # Find the assigned slot
                        assigned_slot = None
                        for s in range(self.n_spots):
                            if (i, t, s) in w and w[(i, t, s)].varValue is not None and w[(i, t, s)].varValue > 0.5:
                                assigned_slot = s
                                break
                        
                        if assigned_slot is not None:
                            schedule[i].append((
                                self.times[t],      # t_start
                                self.times[t] + self.dt, # t_end
                                c,                  # charger_id
                                assigned_slot,      # slot
                                var_val             # power
                            ))
        
        return schedule
    
    def _calculate_objective_values(self, model, schedule, satisfaction):
        """
        Calculates the values of the individual objectives.

        Args:
            model (LpProblem): The solved PuLP MILP model.
            schedule (dict): The optimized charging schedule.
            satisfaction (dict): Dictionary of LpVariable for customer satisfaction.

        Returns:
            dict: A dictionary containing the calculated objective values.
        """
        # Energy cost
        energy_cost = sum(
            self.prices[self._get_time_index(entry[0])] * self.dt * entry[4]
            for ev_id, entries in schedule.items()
            for entry in entries
        )
        
        # Weighted satisfaction by priority and willingness to pay
        total_weighted_satisfaction = sum(
            self.priority[i] * self.willingness_to_pay[i] * satisfaction[i].varValue
            for i in self.EVs
            if hasattr(satisfaction[i], 'varValue') and satisfaction[i].varValue is not None
        )
        
        # Total energy delivered
        total_energy_delivered = sum(
            entry[4] * self.dt
            for ev_id, entries in schedule.items()
            for entry in entries
        )
        
        # Total energy required
        total_energy_required = sum(self.required_energy.values())
        
        # Percentage of energy satisfied
        energy_satisfaction_pct = (total_energy_delivered / total_energy_required) * 100 if total_energy_required > 0 else 100
        
        return {
            "energy_cost": energy_cost,
            "weighted_satisfaction": total_weighted_satisfaction,
            "total_energy_delivered": total_energy_delivered,
            "total_energy_required": total_energy_required,
            "energy_satisfaction_pct": energy_satisfaction_pct
        }
    
    def _generate_rejection_report(self, model, u, satisfaction):
        """
        Generates a detailed report of partially or entirely unsatisfied EVs.

        Args:
            model (LpProblem): The solved PuLP MILP model.
            u (dict): Dictionary of LpVariable for unmet energy.
            satisfaction (dict): Dictionary of LpVariable for customer satisfaction.

        Returns:
            dict: A dictionary containing details for rejected or partially satisfied EVs.
        """
        rejected_details = {}
        
        for i in self.EVs:
            if hasattr(u[i], 'varValue') and u[i].varValue is not None and u[i].varValue > 1e-4:
                unmet_energy = u[i].varValue
                actual_satisfaction = 1.0 - (unmet_energy / self.required_energy[i])
                
                rejected_details[i] = {
                    "required_energy": self.required_energy[i],
                    "delivered_energy": self.required_energy[i] - unmet_energy,
                    "unmet_energy": unmet_energy,
                    "satisfaction": actual_satisfaction,
                    "priority": self.priority[i],
                    "willingness_to_pay": self.willingness_to_pay[i],
                    "weighted_satisfaction": actual_satisfaction * self.priority[i] * self.willingness_to_pay[i]
                }
        
        return rejected_details
    
    def _get_time_index(self, time_value):
        """
        Gets the time index corresponding to a time value.

        Args:
            time_value (float): The time value to find the index for.

        Returns:
            int: The index of the time value in self.times. Defaults to 0 if not found.
        """
        for idx, t in enumerate(self.times):
            if abs(t - time_value) < 1e-5:
                return idx
        return 0 
    
    def _convert_rl_schedule_to_dict(self, rl_schedule):
        """
        Converts a list of RL assignments to a dictionary for visualization.

        Args:
            rl_schedule (list): A list of tuples, where each tuple represents an RL assignment
                                 (ev_id, t_idx, charger_id, slot, power).

        Returns:
            defaultdict: A dictionary where keys are EV IDs and values are lists of
                         (start_time, end_time, charger_id, slot, power) tuples.
        """
        from collections import defaultdict
        schedule_dict = defaultdict(list)
        
        for (ev_id, t_idx, charger_id, slot, power) in rl_schedule:
            if t_idx < len(self.times):
                t_start = self.times[t_idx]
                t_end = t_start + self.dt
                schedule_dict[ev_id].append((t_start, t_end, charger_id, slot, power))
        
        return schedule_dict