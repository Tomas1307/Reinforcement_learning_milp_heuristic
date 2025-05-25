import pulp
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus
import numpy as np
import time

class EVChargingMILP:
    """
    Optimizador MILP multiobjetivo para programación de carga de vehículos eléctricos.
    Considera restricciones avanzadas de compatibilidad, eficiencia y satisfacción del cliente.
    """
    def __init__(self, config):
        self.config = config
        self.process_config()
        
    def process_config(self):
        """Procesa la configuración para extraer parámetros necesarios."""
        data = self.config
        
        # Parámetros básicos
        self.times = data["times"]
        self.prices = data["prices"]
        self.arrivals = data["arrivals"]
        self.chargers = data["parking_config"]["chargers"]
        self.station_limit = data["parking_config"]["transformer_limit"]
        self.dt = data.get("dt", 0.25)  # Default a 15 minutos
        self.n_spots = data["parking_config"]["n_spots"]
        
        # Índices
        self.T = range(len(self.times))
        self.EVs = [arr["id"] for arr in self.arrivals]
        
        # Mapeo de datos de vehículos
        self.arrival_time = {arr["id"]: arr["arrival_time"] for arr in self.arrivals}
        self.departure_time = {arr["id"]: arr["departure_time"] for arr in self.arrivals}
        self.required_energy = {arr["id"]: arr["required_energy"] for arr in self.arrivals}
        
        # Nuevos parámetros avanzados
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
            self.max_charge_rate[ev_id] = arr.get("max_charge_rate", 50)   # Default 50kW
            self.ac_charge_rate[ev_id] = arr.get("ac_charge_rate", 7)      # Default 7kW AC
            self.dc_charge_rate[ev_id] = arr.get("dc_charge_rate", 50)     # Default 50kW DC
            self.priority[ev_id] = arr.get("priority", 1)                  # Default prioridad 1
            self.willingness_to_pay[ev_id] = arr.get("willingness_to_pay", 1.0)  # Default 1.0
            self.efficiency[ev_id] = arr.get("efficiency", 0.9)            # Default 90% eficiencia
            
            if "battery_capacity" in arr:
                self.battery_capacity[ev_id] = arr["battery_capacity"]
            if "brand" in arr:
                self.brand[ev_id] = arr["brand"]
        
        # Información sobre cargadores
        self.charger_ids = [c["charger_id"] for c in self.chargers]
        self.max_charger_power = {c["charger_id"]: c["power"] for c in self.chargers}
        self.charger_type = {c["charger_id"]: c.get("type", "AC") for c in self.chargers}
        self.charger_efficiency = {c["charger_id"]: c.get("efficiency", 0.9) for c in self.chargers}
        
        # Compatibilidad cargador-vehículo
        self.compatible_vehicles = {}
        for c in self.chargers:
            charger_id = c["charger_id"]
            if "compatible_vehicles" in c:
                self.compatible_vehicles[charger_id] = c["compatible_vehicles"]
            else:
                # Por defecto, todos los cargadores son compatibles con todos los vehículos
                self.compatible_vehicles[charger_id] = [b for b in self.brand.values()]
        
        # Mapeo de compatibilidad
        self.ev_charger_compatible = {}
        for ev_id in self.EVs:
            self.ev_charger_compatible[ev_id] = []
            ev_brand = self.brand.get(ev_id, "")
            
            for charger_id in self.charger_ids:
                # Verificar si el cargador es compatible con esta marca
                compatible = True
                if ev_brand and self.compatible_vehicles.get(charger_id):
                    # Extraer marca base sin los detalles
                    base_brand = ev_brand.split()[0] if " " in ev_brand else ev_brand
                    compatible = False
                    for comp_brand in self.compatible_vehicles[charger_id]:
                        if comp_brand in ev_brand or base_brand in comp_brand:
                            compatible = True
                            break
                
                if compatible:
                    # Verificar limites de potencia
                    charger_power = self.max_charger_power[charger_id]
                    charger_type = self.charger_type[charger_id]
                    
                    # Si es cargador AC, verificar AC charge rate
                    if charger_type == "AC" and charger_power <= self.ac_charge_rate.get(ev_id, 7):
                        self.ev_charger_compatible[ev_id].append(charger_id)
                    # Si es cargador DC, verificar DC charge rate
                    elif charger_type == "DC" and charger_power <= self.dc_charge_rate.get(ev_id, 50):
                        self.ev_charger_compatible[ev_id].append(charger_id)
    
    #
    def solve(self, penalty_unmet=1000.0, rl_schedule=None, time_limit=None,
              epsilon_satisfaction=None, return_infeasible=False):
        """
        Resuelve el problema de scheduling de carga con MILP utilizando el método epsilon-constraint.
        El objetivo principal es minimizar el costo. La satisfacción del cliente se convierte en una restricción.
        
        Args:
            penalty_unmet (float): Penalización por energía no entregada.
            rl_schedule (list): Solución inicial (por RL) para warm start.
            time_limit (int): Límite de tiempo para resolver MILP (segundos).
            epsilon_satisfaction (float): Umbral mínimo de satisfacción ponderada que debe cumplir el modelo (entre 0 y 1).
                                          Si es None, se usa el modo ponderado original.
            return_infeasible (bool): Si es True, devuelve la mejor solución posible incluso si es infactible.
            
        Returns:
            model (LpProblem): Modelo MILP resuelto.
            schedule (dict): Diccionario con solución optimizada.
            rejected_details (dict): Información sobre EVs no satisfechos.
            obj_values (dict): Valores finales de las funciones objetivo (costo y satisfacción).
        """
        # Coeficientes de penalización para las holguras
        M_slack = 1e4
        
        # Crear problema MILP
        model = LpProblem("EV_Charging_Epsilon_Constraint", LpMinimize)
        
        # Variables de decisión principales
        x = {}  # Potencia asignada a cada EV, periodo, cargador
        y = {}  # Variable binaria de conexión a cargador
        for i in self.EVs:
            for t in self.T:
                # Solo considerar períodos donde el EV está presente
                if self.arrival_time[i] <= self.times[t] < self.departure_time[i]:
                    # Solo considerar cargadores compatibles
                    for c in self.ev_charger_compatible.get(i, self.charger_ids):
                        x[(i, t, c)] = LpVariable(f"x_{i}_{t}_{c}", lowBound=0)
                        y[(i, t, c)] = LpVariable(f"y_{i}_{t}_{c}", cat="Binary")
                        
                        # Inicialización con solución RL si se proporciona
                        if rl_schedule is not None:
                            for entry in rl_schedule:
                                if entry[0] == i and entry[1] == t and entry[2] == c:
                                    x[(i, t, c)].setInitialValue(entry[4])
                                    y[(i, t, c)].setInitialValue(1)
                                    break
        
        # Variable para energía no cubierta por EV
        u = {i: LpVariable(f"u_{i}", lowBound=0) for i in self.EVs}
        
        # Variable para satisfacción ponderada del cliente (objetivo 2)
        satisfaction = {i: LpVariable(f"satisfaction_{i}", lowBound=0, upBound=1) for i in self.EVs}
        
        # Variable binaria: EV estacionado en intervalo t
        z = {}
        for i in self.EVs:
            for t in self.T:
                if self.arrival_time[i] <= self.times[t] < self.departure_time[i]:
                    z[(i, t)] = LpVariable(f"z_{i}_{t}", cat="Binary")
                else:
                    z[(i, t)] = 0
        
        # Variables binarias: asignación de slot en cada intervalo para cada EV
        w = {}
        for i in self.EVs:
            for t in self.T:
                if self.arrival_time[i] <= self.times[t] < self.departure_time[i]:
                    for s in range(self.n_spots):
                        w[(i, t, s)] = LpVariable(f"w_{i}_{t}_{s}", cat="Binary")
        
        # Variables de holgura
        s_charger = {}      # Para capacidad de cada cargador
        s_transformer = {}  # Para límite del transformador
        s_parking = {}      # Para capacidad del parqueadero
        s_slot = {}         # Para asignación de slot
        s_min_charge = {}   # Para tasa mínima de carga
        
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
        
        # OBJETIVO PRINCIPAL: Minimizar el costo económico
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
        
        # NUEVA: Restricción para la satisfacción del cliente (epsilon-constraint)
        if epsilon_satisfaction is not None:
            # Validar epsilon_satisfaction
            assert 0 <= epsilon_satisfaction <= 1, "epsilon_satisfaction debe estar entre 0 y 1"
            
            # Suma de la satisfacción ponderada de todos los EVs
            total_weighted_satisfaction = lpSum(self.priority[i] * self.willingness_to_pay[i] * satisfaction[i] 
                                                for i in self.EVs)
            
            # Calcular la máxima satisfacción ponderada posible para normalizar el epsilon
            # Esto asume que la máxima satisfacción ponderada se obtiene cuando todos los EVs tienen satisfacción 1
            max_possible_weighted_satisfaction = lpSum(self.priority[i] * self.willingness_to_pay[i] 
                                                       for i in self.EVs)
            
            # La suma de la satisfacción ponderada debe ser mayor o igual a epsilon * max_possible_weighted_satisfaction
            # Solo añadir la restricción si hay EVs y si la máxima satisfacción posible es > 0
            if pulp.value(max_possible_weighted_satisfaction) is not None and pulp.value(max_possible_weighted_satisfaction) > 1e-6: # Evitar división por cero o umbrales insignificantes
                model += (total_weighted_satisfaction >= epsilon_satisfaction * max_possible_weighted_satisfaction,
                          "Epsilon_Satisfaction_Constraint")
            else:
                print("Advertencia: No hay EVs o la máxima satisfacción posible es cero. Epsilon-constraint no aplicado para satisfacción.")
        else:
            print("Epsilon-constraint para satisfacción no activado. Usando modo de optimización de costo puro.")


        # Restricción 1: Carga entregada + energía no cubierta = energía requerida
        for i in self.EVs:
            model += (
                lpSum(x[(i, t, c)] * self.dt 
                      for t in self.T for c in self.charger_ids 
                      if (i, t, c) in x)
                + u[i]
                == self.required_energy[i]
            ), f"Energy_Balance_EV_{i}"
        
        # Restricción 2: Capacidad de cada cargador (con slack)
        for t in self.T:
            for c in self.charger_ids:
                model += (
                    lpSum(x[(i, t, c)] for i in self.EVs if (i, t, c) in x) 
                    <= self.max_charger_power[c] + s_charger[(t, c)]
                ), f"ChargerCap_t{t}_c{c}"
        
        # Restricción 3: Límite del transformador (con slack)
        for t in self.T:
            model += (
                lpSum(x[(i, t, c)] for i in self.EVs for c in self.charger_ids if (i, t, c) in x) 
                <= self.station_limit + s_transformer[t]
            ), f"StationCap_t{t}"
        
        # Restricción 4: No cargar fuera de la ventana de disponibilidad
        for i in self.EVs:
            for t in self.T:
                if not (self.arrival_time[i] <= self.times[t] < self.departure_time[i]):
                    for c in self.charger_ids:
                        if (i, t, c) in x:
                            model += x[(i, t, c)] == 0, f"NoChargeOutside_{i}_{t}_{c}"
        
        # Restricción 5: Asignación única de cargador en cada intervalo
        for i in self.EVs:
            for t in self.T:
                if self.arrival_time[i] <= self.times[t] < self.departure_time[i]:
                    model += (
                        lpSum(y[(i, t, c)] for c in self.charger_ids if (i, t, c) in y) <= 1
                    ), f"UniqueCharger_EV_{i}_t{t}"
        
        # Restricción 6: Vinculación entre x e y
        for i in self.EVs:
            for t in self.T:
                for c in self.charger_ids:
                    if (i, t, c) in x:
                        model += (
                            x[(i, t, c)] <= y[(i, t, c)] * self.max_charger_power[c]
                        ), f"Link_x_y_EV_{i}_t{t}_c{c}"
                        
                        # NUEVA: Restricción de carga mínima cuando está conectado
                        model += (
                            x[(i, t, c)] >= y[(i, t, c)] * self.min_charge_rate[i] - s_min_charge[(i, t)]
                        ), f"MinCharge_EV_{i}_t{t}_c{c}"
        
        # Restricción 7: Vinculación entre z e y
        for i in self.EVs:
            for t in self.T:
                if isinstance(z[(i, t)], LpVariable):
                    compatible_chargers = [c for c in self.charger_ids if (i, t, c) in y]
                    if compatible_chargers:
                        model += (
                            z[(i, t)] >= lpSum(y[(i, t, c)] for c in compatible_chargers) / len(compatible_chargers)
                        ), f"Link_z_y_EV_{i}_t{t}"
        
        # Restricción 8: Capacidad del parqueadero (con slack)
        for t in self.T:
            active_evs = [i for i in self.EVs if self.arrival_time[i] <= self.times[t] < self.departure_time[i]]
            if active_evs:
                model += (
                    lpSum(z[(i, t)] for i in active_evs if isinstance(z[(i, t)], LpVariable)) 
                    <= self.n_spots + s_parking[t]
                ), f"ParkingCap_t{t}"
        
        # Restricción 9: Asignación de slot de parqueo (con slack)
        for i in self.EVs:
            for t in self.T:
                if self.arrival_time[i] <= self.times[t] < self.departure_time[i]:
                    model += (
                        lpSum(w[(i, t, s)] for s in range(self.n_spots) if (i, t, s) in w) + s_slot[(i, t)] == 1
                    ), f"AssignSlot_EV_{i}_t{t}"
        
        # Restricción 10: Unicidad de slot en cada intervalo
        for t in self.T:
            for s in range(self.n_spots):
                evs_in_interval = [i for i in self.EVs if self.arrival_time[i] <= self.times[t] < self.departure_time[i]]
                if evs_in_interval:
                    slots_for_time_spot = [(i, t, s) for i in evs_in_interval if (i, t, s) in w]
                    if slots_for_time_spot:
                        model += (
                            lpSum(w[(i, t, s)] for i in evs_in_interval if (i, t, s) in w) <= 1
                        ), f"SlotUnique_t{t}_s{s}"
        
        # Restricción 11: Definición de satisfacción del cliente
        for i in self.EVs:
            # Satisfacción = Energía entregada / Energía requerida
            model += (
                satisfaction[i] * self.required_energy[i] 
                == lpSum(x[(i, t, c)] * self.dt for t in self.T for c in self.charger_ids if (i, t, c) in x)
            ), f"Satisfaction_EV_{i}"
        
        # Restricciones de control para variables auxiliares
        for i in self.EVs:
            model += satisfaction[i] <= 1, f"Max_Satisfaction_{i}"
        
        # Restricción 12: Incorporar eficiencia de carga
        # Esta restricción ajusta la potencia real que recibe la batería considerando eficiencias
        for i in self.EVs:
            for t in self.T:
                for c in self.charger_ids:
                    if (i, t, c) in x:
                        # La energía que llega a la batería es menor debido a las pérdidas
                        combined_efficiency = self.efficiency[i] * self.charger_efficiency[c]
                        # Nota: Si ya estás multiplicando por la eficiencia en el cálculo de energía_entregada,
                        # asegúrate de no duplicar este efecto. La forma actual de la restricción 11 ya usa
                        # la energía entregada 'x' directamente sin la eficiencia combinada para satisfacción,
                        # lo cual es usualmente la potencia de salida del cargador. Si quieres que la satisfacción
                        # se base en la energía *realmente recibida por la batería*, tendrías que modificar la R11.
                        # Por ahora, se mantiene como estaba para la compatibilidad con el cálculo de satisfacción.
        
        # Configurar límite de tiempo y optimizador
        if time_limit:
            print(f"Configurando MILP con tiempo límite: {time_limit} segundos y gap de 0.01 (1%)")
            solver = pulp.PULP_CBC_CMD(msg=True, options=[
                f"sec {time_limit}", 
                "timeMode elapsed", 
                "ratioGap 0.01"
            ])
        else:
            solver = pulp.PULP_CBC_CMD(msg=True, options=["ratioGap 0.01"])
        
        # Resolver
        start_time_model = time.time()
        try:
            status = model.solve(solver)
            solve_time = time.time() - start_time_model
            
            print("Status:", LpStatus[model.status])
            print("Objective value:", model.objective.value())
            print("Tiempo de resolución MILP:", solve_time, "segundos")
            
            # Si el modelo es infactible y no queremos resultados infactibles, retornar error
            if LpStatus[model.status] not in ("Optimal", "Not Solved"):  # Accept Optimal or if it couldn't be solved (e.g., timeout)
                if not return_infeasible:
                    print("El modelo no se pudo resolver de manera óptima.")
                    if rl_schedule is not None:
                        print("Devolviendo solución RL original")
                        return None, self._convert_rl_schedule_to_dict(rl_schedule), {}, {}
                    else:
                        return None, {}, {}, {}
                    
        except Exception as e:
            print(f"Error al resolver MILP: {e}")
            print("Devolviendo solución RL original")
            if rl_schedule is not None:
                return None, self._convert_rl_schedule_to_dict(rl_schedule), {}, {}
            else:
                return None, {}, {}, {}
        
        # Extraer solución
        schedule = self._extract_solution(model, x, w)
        
        # Calcular valores de objetivos separados
        obj_values = self._calculate_objective_values(model, schedule, satisfaction)
        
        # Generar reporte de EVs parcialmente o no atendidos
        rejected_details = self._generate_rejection_report(model, u, satisfaction)
        
        return model, schedule, rejected_details, obj_values
        
    def _extract_solution(self, model, x, w):
        """Extrae la solución del modelo MILP."""
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
                        # Buscar qué slot se asignó
                        assigned_slot = None
                        for s in range(self.n_spots):
                            if (i, t, s) in w and w[(i, t, s)].varValue is not None and w[(i, t, s)].varValue > 0.5:
                                assigned_slot = s
                                break
                        
                        if assigned_slot is not None:
                            schedule[i].append((
                                self.times[t],           # t_start
                                self.times[t] + self.dt, # t_end
                                c,                      # charger_id
                                assigned_slot,          # slot
                                var_val                 # power
                            ))
        
        return schedule
    
    def _calculate_objective_values(self, model, schedule, satisfaction):
        """Calcula los valores de los objetivos individuales."""
        # Costo de energía
        energy_cost = sum(
            self.prices[self._get_time_index(entry[0])] * self.dt * entry[4]
            for ev_id, entries in schedule.items()
            for entry in entries
        )
        
        # Satisfacción ponderada por prioridad y willingness to pay
        total_weighted_satisfaction = sum(
            self.priority[i] * self.willingness_to_pay[i] * satisfaction[i].varValue
            for i in self.EVs
            if hasattr(satisfaction[i], 'varValue') and satisfaction[i].varValue is not None
        )
        
        # Energía total entregada
        total_energy_delivered = sum(
            entry[4] * self.dt
            for ev_id, entries in schedule.items()
            for entry in entries
        )
        
        # Energía total requerida
        total_energy_required = sum(self.required_energy.values())
        
        # Porcentaje de energía satisfecha
        energy_satisfaction_pct = (total_energy_delivered / total_energy_required) * 100 if total_energy_required > 0 else 100
        
        return {
            "energy_cost": energy_cost,
            "weighted_satisfaction": total_weighted_satisfaction,
            "total_energy_delivered": total_energy_delivered,
            "total_energy_required": total_energy_required,
            "energy_satisfaction_pct": energy_satisfaction_pct
        }
    
    def _generate_rejection_report(self, model, u, satisfaction):
        """Genera un reporte detallado de EVs parcialmente o no atendidos."""
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
        """Obtiene el índice de tiempo correspondiente a un valor de tiempo."""
        for idx, t in enumerate(self.times):
            if abs(t - time_value) < 1e-5:
                return idx
        return 0  # Default si no se encuentra
    
    def _convert_rl_schedule_to_dict(self, rl_schedule):
        """Convierte una lista de asignaciones RL a un diccionario para visualización."""
        from collections import defaultdict
        schedule_dict = defaultdict(list)
        
        for (ev_id, t_idx, charger_id, slot, power) in rl_schedule:
            if t_idx < len(self.times):
                t_start = self.times[t_idx]
                t_end = t_start + self.dt
                schedule_dict[ev_id].append((t_start, t_end, charger_id, slot, power))
        
        return schedule_dict