import json 
import numpy as np 
import random 
import matplotlib.pyplot as plt 
import os 
import glob 
import time 
from collections import defaultdict
from modules.clusters.k_means_heuristic import modify_heuristic_with_best_params


def load_data(json_path): 
    """ Carga el archivo JSON y retorna los datos del sistema. """ 
    with open(json_path, 'r') as f: data = json.load(f)


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

class HeuristicaConstructivaEVs: 
    """ Implementación del algoritmo HeurísticaConstructivaEVs sin optimización local. """ 

    def __init__(self, config): 
        """ Inicializa la heurística con la configuración del sistema.
        Args:
            config: Diccionario con la configuración del sistema.
        """
        self.alpha_cost = config.get("alpha_cost", 0.6)
        self.alpha_satisfaction = config.get("alpha_satisfaction", 0.4)
        self.penalizacion_base = config.get("penalizacion_base", 1000.0)
        self.times = config["times"]
        self.prices = config["prices"]
        self.arrivals = config["arrivals"]
        self.chargers = config["chargers"]
        self.station_limit = config["station_limit"]
        self.dt = config["dt"]
        self.n_spots = config["n_spots"]
        
        # Mapeos para facilitar el acceso
        self.ev_ids = [arr["id"] for arr in self.arrivals]
        self.arrival_time = {arr["id"]: arr["arrival_time"] for arr in self.arrivals}
        self.departure_time = {arr["id"]: arr["departure_time"] for arr in self.arrivals}
        self.required_energy = {arr["id"]: arr["required_energy"] for arr in self.arrivals}
        self.priority = {arr["id"]: arr.get("priority", 1) for arr in self.arrivals}
        self.willingness_to_pay = {arr["id"]: arr.get("willingness_to_pay", 1.0) for arr in self.arrivals}
        self.efficiency = {arr["id"]: arr.get("efficiency", 0.9) for arr in self.arrivals}

        self.battery_capacity = {arr["id"]: arr.get("battery_capacity", 40) for arr in self.arrivals}
        self.min_charge_rate = {arr["id"]: arr.get("min_charge_rate", 3.5) for arr in self.arrivals}
        self.max_charge_rate = {arr["id"]: arr.get("max_charge_rate", 50) for arr in self.arrivals}
        self.ac_charge_rate = {arr["id"]: arr.get("ac_charge_rate", 7) for arr in self.arrivals}
        self.dc_charge_rate = {arr["id"]: arr.get("dc_charge_rate", 50) for arr in self.arrivals}
        self.brand = {arr["id"]: arr.get("brand", "Generic") for arr in self.arrivals}

        self.charger_type = {c["charger_id"]: c.get("type", "AC") for c in self.chargers}
        self.charger_efficiency = {c["charger_id"]: c.get("efficiency", 1.0) for c in self.chargers}

        self.charger_ids = [c["charger_id"] for c in self.chargers]
        self.max_charger_power = {c["charger_id"]: c["power"] for c in self.chargers}
        
        self.ev_charger_compatible = {}
        for ev_id in self.ev_ids:
            ev_brand = self.brand[ev_id]
            ac_limit = self.ac_charge_rate[ev_id]
            dc_limit = self.dc_charge_rate[ev_id]
            self.ev_charger_compatible[ev_id] = []

            for c in self.chargers:
                cid = c["charger_id"]
                ctype = c.get("type", "AC")
                power = c["power"]
                compatible_brands = c.get("compatible_vehicles", [])

                brand_match = not compatible_brands or any(b in ev_brand for b in compatible_brands)
                power_ok = (ctype == "AC" and power <= ac_limit) or (ctype == "DC" and power <= dc_limit)

                if brand_match and power_ok:
                    self.ev_charger_compatible[ev_id].append(cid)

        # Variables para almacenar el estado del sistema
        self.schedule = []  # [(ev_id, t_idx, charger_id, slot, power)]
        self.occupied_spots = {t: set() for t in range(len(self.times))}
        self.occupied_chargers = {t: set() for t in range(len(self.times))}
        self.power_used = {t: 0 for t in range(len(self.times))}
        self.energy_delivered = {ev_id: 0 for ev_id in self.ev_ids}
        self.energy_remaining = {ev_id: self.required_energy[ev_id] for ev_id in self.ev_ids}
        self.current_ev_assignment = {t: {} for t in range(len(self.times))}  # {t: {ev_id: (slot, charger, power)}}
        self.rejected_vehicles = set()
        
        # Construir compatibilidad EV–cargador por tipo y marca


    def run(self, track_progress=False):
        """
        Ejecuta la heurística constructiva con las mejoras de continuidad.
        
        Args:
            track_progress: Si es True, registra la evolución del costo en cada iteración.
            
        Returns:
            Lista de tuplas (ev_id, t_idx, charger_id, slot, power) con el schedule, 
            y, si track_progress es True, también la lista de costos por iteración.
        """
        print("Ejecutando HeurísticaConstructivaEVs mejorada...")
        # Preprocesar restricciones de capacidad (NUEVO)
        self._preprocess_vehicle_capacity_constraint()
        rechazados = len(getattr(self, 'rejected_vehicles', set()))
        print(f"Vehículos rechazados por restricciones de capacidad: {rechazados}")
        # Fase 1: Construcción inicial de la solución con continuidad
        self._fase_construccion_inicial()
        
        # Fase 2: Mejora mediante exploración del espacio de soluciones
        schedule_actual = self.schedule.copy()
        mejor_schedule = schedule_actual.copy()
        mejor_costo = self._evaluar_costo(mejor_schedule)
        intentos_sin_mejora = 0
        temperatura = 1.0
        factor_enfriamiento = 0.95
        umbral_reinicio = 20
        
        max_iteraciones = 1000
        
        # Para seguimiento de progreso
        costos_por_iteracion = []
        if track_progress:
            costos_por_iteracion.append(mejor_costo)
        
        print(f"Iniciando fase de mejora con {max_iteraciones} iteraciones...")
        for i in range(max_iteraciones):
            # Seleccionar estrategia de perturbación
            p = random.random()
            if p <= 0.4:
                estrategia = "Perturbacion_Intervalos"
            elif p <= 0.7:
                estrategia = "Perturbacion_Vehiculos"
            elif p <= 0.9:
                estrategia = "Perturbacion_Hibrida"
            else:
                estrategia = "Perturbacion_Sesiones"
            
            # Aplicar perturbación
            nuevo_schedule = self._perturba_solucion_mejorado(schedule_actual, estrategia)
            
            # Reconstruir solución
            nuevo_schedule = self._reconstruye_solucion(nuevo_schedule)
            
            # Evaluar nueva solución
            nuevo_costo = self._evaluar_costo(nuevo_schedule)
            
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
                prob_aceptacion = np.exp(-delta / (temperatura * mejor_costo / 1000))  # Escalar por el tamaño del problema
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
                temperatura = 0.5  # Reiniciar con temperatura moderada
                print(f"Iteración {i}: Reiniciando búsqueda - {umbral_reinicio} iteraciones sin mejora. T = {temperatura:.4f}")
        
        print(f"Mejora completada. Costo final: {mejor_costo:.2f}")
        self.schedule = mejor_schedule
        
        if track_progress:
            return self.schedule, costos_por_iteracion
        else:
            return self.schedule

    def _preprocess_vehicle_capacity_constraint(self):
        """
        Preprocesa los vehículos para identificar y rechazar aquellos que no podrán ser acomodados
        debido a limitaciones físicas de capacidad del parqueadero.
        
        Este método modifica el conjunto de vehículos a considerar en la solución.
        """
        print("Verificando restricciones de capacidad del parqueadero...")
            
        # Diccionario para contar cuántos vehículos están presentes en cada intervalo
        vehicles_per_interval = {t: [] for t in range(len(self.times))}
        
        # Contar vehículos presentes en cada intervalo
        for ev_id in self.ev_ids:
            for t in range(len(self.times)):
                if self.arrival_time[ev_id] <= self.times[t] < self.departure_time[ev_id]:
                    vehicles_per_interval[t].append(ev_id)
        
        # Identificar intervalos con sobrecarga de capacidad
        overloaded_intervals = {
            t: vehicles for t, vehicles in vehicles_per_interval.items() 
            if len(vehicles) > self.n_spots
        }
        
        if not overloaded_intervals:
            print("No hay restricciones de capacidad violadas. Todos los vehículos pueden ser acomodados.")
            return
        
        print(f"Detectadas {len(overloaded_intervals)} intervalos con restricciones de capacidad violadas.")
        print(f"Se aplicará un algoritmo de priorización para rechazar vehículos.")
        
        # Calcular métricas de prioridad para cada vehículo
        ev_priorities = {}
        for ev_id in self.ev_ids:
            # Tiempo de estancia
            stay_duration = self.departure_time[ev_id] - self.arrival_time[ev_id]
            
            # Energía requerida normalizada por tiempo
            energy_per_time = self.required_energy[ev_id] / max(0.25, stay_duration)
            
            # Prioridad: vehículos con mayor necesidad energética por unidad de tiempo y menor tiempo de estancia
            priority_score = energy_per_time / max(0.5, stay_duration)
            ev_priorities[ev_id] = priority_score
        
        # Conjunto para rastrear vehículos rechazados globalmente
        rejected_vehicles = set()
        
        # Para cada intervalo con sobrecarga, rechazar los vehículos con menor prioridad
        for t, vehicles in sorted(overloaded_intervals.items()):
            # Filtrar vehículos que ya han sido rechazados
            active_vehicles = [v for v in vehicles if v not in rejected_vehicles]
            
            # Si aún hay sobrecarga después de considerar rechazos previos
            if len(active_vehicles) > self.n_spots:
                # Ordenar por prioridad (menor a mayor)
                sorted_vehicles = sorted(active_vehicles, key=lambda v: ev_priorities.get(v, 0))
                
                # Rechazar los vehículos con menor prioridad
                vehicles_to_reject = sorted_vehicles[:len(active_vehicles) - self.n_spots]
                rejected_vehicles.update(vehicles_to_reject)
        
        # Registrar vehículos rechazados
        self.rejected_vehicles = rejected_vehicles
        
        # Ajustar energía entregada y restante para los vehículos rechazados
        for ev_id in rejected_vehicles:
            self.energy_delivered[ev_id] = 0
            self.energy_remaining[ev_id] = 0  # Opcional: indicar que no se intentará entregar energía
        
        print(f"Se han rechazado {len(rejected_vehicles)} vehículos debido a restricciones de capacidad.")
        print(f"IDs de vehículos rechazados: {sorted(list(rejected_vehicles))}")
        
    def _intentar_swap_mejorado(self, t, ultima_asignacion, umbral=5.0):
        """
        Versión mejorada de _intentar_swap que también considera la continuidad de carga.
        
        Args:
            t: Intervalo de tiempo actual
            ultima_asignacion: Diccionario con la última asignación por EV
            umbral: Energía faltante (kWh) que define que un EV está casi completo
        """
        # Identificar EVs que necesitan carga urgente pero no tienen cargador
        evs_sin_cargador = []
        for ev in self.ev_ids:
            # Solo considerar EVs presentes sin cargador asignado
            if (self.arrival_time[ev] <= self.times[t] < self.departure_time[ev] and
                ev in self.current_ev_assignment[t] and
                self.current_ev_assignment[t][ev][1] is None and
                self.energy_remaining[ev] > umbral):  # Necesita más que el umbral
                
                # Priorizar EVs que estaban cargando en intervalos anteriores
                prioridad = self._calcular_urgencia(ev, t)
                if ultima_asignacion[ev] is not None:
                    last_t, _, last_charger = ultima_asignacion[ev]
                    if last_t == t-1 and last_charger is not None:
                        prioridad *= 2  # Duplicar urgencia si venía cargando
                
                evs_sin_cargador.append((ev, prioridad))
        
        # Ordenar por prioridad (urgencia) descendente
        evs_sin_cargador.sort(key=lambda x: x[1], reverse=True)
        
        # Para cada EV que necesita carga urgente
        for ev, _ in evs_sin_cargador:
            # Buscar candidatos con cargador que casi han terminado
            ev_candidatos = []
            for ev2, asignacion in self.current_ev_assignment[t].items():
                if asignacion[1] is not None and ev2 != ev:
                    if self.energy_remaining[ev2] < umbral or self._calcular_urgencia(ev2, t) < self._calcular_urgencia(ev, t) / 2:
                        ev_candidatos.append((ev2, self._calcular_urgencia(ev2, t), asignacion))
            
            # Ordenar candidatos por urgencia ascendente (menor primero)
            ev_candidatos.sort(key=lambda x: x[1])
            
            # Intentar el swap con el candidato de menor urgencia
            if ev_candidatos:
                ev2, _, asignacion = ev_candidatos[0]
                
                # Realizar el swap
                self._quitar_asignacion(ev2, t)
                
                # Asignar el cargador a ev
                nuevo_power = min(
                    self.max_charger_power[asignacion[1]],
                    self.energy_remaining[ev] / self.dt,
                    self.station_limit - self.power_used[t]
                )
                if nuevo_power > 0:
                    self._asignar_carga(ev, t, asignacion[1], asignacion[0], nuevo_power)
                    ultima_asignacion[ev] = (t, asignacion[0], asignacion[1])
                
                # Intentar reubicar a ev2 en un slot sin cargador
                slots_disponibles = [
                    s for s in range(self.n_spots)
                    if s not in self.occupied_spots[t]
                ]
                
                if slots_disponibles:
                    slot = slots_disponibles[0]
                    self._asignar_spot(ev2, t, None, slot, 0)
                    ultima_asignacion[ev2] = (t, slot, None)


    def _fase_construccion_inicial(self):
        """
        Implementa una fase de construcción inicial mejorada que prioriza la continuidad 
        en la carga de los vehículos a lo largo de múltiples intervalos de tiempo.
        """
        print("Fase 1: Construcción inicial por intervalos con continuidad prioritaria...")
        
        # Inicializar seguimiento de sesiones de carga
        ultima_asignacion = {ev_id: None for ev_id in self.ev_ids}  # {ev_id: (last_t, slot, charger)}
        
        # Para cada intervalo de tiempo en orden cronológico
        for t in range(len(self.times)):
            # a. Identificar vehículos presentes
            ev_presentes = [
            ev_id for ev_id in self.ev_ids
            if ev_id not in self.rejected_vehicles and  # NUEVA CONDICIÓN
            self.arrival_time[ev_id] <= self.times[t] < self.departure_time[ev_id]
            ]
            
            if not ev_presentes:
                continue
            
            # b. Ordenar EV_presentes por urgencia Y priorizar continuidad
            def criterio_prioridad(ev):
                urgencia_base = self._calcular_urgencia(ev, t)
                # Bonificación por continuidad (tiempo consecutivo)
                bonificacion_continuidad = 0
                if ultima_asignacion[ev] is not None:
                    last_t, _, _ = ultima_asignacion[ev]
                    if last_t == t - 1:  # Si estaba cargando en el intervalo anterior
                        bonificacion_continuidad = urgencia_base * 0.5  # Aumenta prioridad en 50%
                
                return urgencia_base + bonificacion_continuidad
            
            ev_ordenados = sorted(
                ev_presentes,
                key=criterio_prioridad,
                reverse=True
            )
            
            # c. Primera pasada: extender sesiones de carga activas
            ev_ya_asignados = set()
            
            for ev in ev_ordenados:
                # Verificar si el EV tenía una sesión de carga en t-1
                if ultima_asignacion[ev] is not None:
                    last_t, prev_slot, prev_charger = ultima_asignacion[ev]
                    
                    # Solo extender si la asignación era en el intervalo inmediatamente anterior
                    if last_t == t-1 and prev_charger is not None and self.energy_remaining[ev] > 0:
                        # Verificar si el slot y cargador siguen disponibles
                        if prev_slot not in self.occupied_spots[t] and prev_charger not in self.occupied_chargers[t]:
                            power = min(
                                self.max_charger_power[prev_charger],
                                self.energy_remaining[ev] / self.dt,
                                self.station_limit - self.power_used[t]
                            )
                            
                            if power > 0:
                                self._asignar_carga(ev, t, prev_charger, prev_slot, power)
                                ultima_asignacion[ev] = (t, prev_slot, prev_charger)
                                ev_ya_asignados.add(ev)
            
            # d. Identificar recursos disponibles para nuevas asignaciones
            slots_con_cargador = []
            for s in range(self.n_spots):
                if s not in self.occupied_spots[t]:
                    for c in self.ev_charger_compatible.get(ev, []):
                        if c not in self.occupied_chargers[t]:
                            slots_con_cargador.append((s, c))
            
            slots_sin_cargador = [
                s for s in range(self.n_spots)
                if s not in self.occupied_spots[t] and s not in [sc[0] for sc in slots_con_cargador]
            ]
            
            # e. Segunda pasada: nuevas asignaciones para vehículos sin continuidad
            for ev in ev_ordenados:
                if ev in ev_ya_asignados or ev in self.current_ev_assignment[t]:
                    continue
                    
                # Si EV necesita energía y hay slots con cargador disponibles
                if self.energy_remaining[ev] > 0 and slots_con_cargador:
                    # Se intenta asignar a slot con cargador
                    slot, charger = slots_con_cargador.pop(0)
                    power = min(
                        self.max_charger_power[charger],
                        self.energy_remaining[ev] / self.dt,
                        self.station_limit - self.power_used[t]
                    )
                    if power > 0:
                        self._asignar_carga(ev, t, charger, slot, power)
                        ultima_asignacion[ev] = (t, slot, charger)
                    else:
                        # No hay suficiente potencia, se asigna a un slot sin cargador
                        if slots_sin_cargador:
                            slot = slots_sin_cargador.pop(0)
                            self._asignar_spot(ev, t, None, slot, 0)
                            ultima_asignacion[ev] = (t, slot, None)
                
                elif slots_sin_cargador:
                    # Asignar a slot sin cargador (no necesita energía o no hay slots con cargador)
                    slot = slots_sin_cargador.pop(0)
                    self._asignar_spot(ev, t, None, slot, 0)
                    ultima_asignacion[ev] = (t, slot, None)
                
                elif ev in self.current_ev_assignment[t]:
                    # Ya tiene asignación (procesado anteriormente)
                    continue
                
                else:
                    # No hay slots disponibles, buscar candidato para desplazar
                    ev_candidatos = [
                        e for e in ev_ordenados 
                        if e in self.current_ev_assignment[t] and
                        e != ev and
                        self._calcular_urgencia(e, t) < self._calcular_urgencia(ev, t) and
                        (ultima_asignacion[e] is None or ultima_asignacion[e][0] != t-1)  # No desplazar continuidades
                    ]
                    
                    if ev_candidatos:
                        # Elegir el candidato con menor urgencia
                        candidato = min(ev_candidatos, key=lambda e: self._calcular_urgencia(e, t))
                        
                        # Obtener asignación del candidato
                        c_slot, c_charger, c_power = self.current_ev_assignment[t][candidato]
                        
                        # Quitar asignación del candidato
                        self._quitar_asignacion(candidato, t)
                        
                        # Asignar spot al EV actual
                        if c_charger is not None and self.energy_remaining[ev] > 0:
                            power = min(
                                self.max_charger_power[c_charger],
                                self.energy_remaining[ev] / self.dt,
                                self.station_limit - self.power_used[t]
                            )
                            self._asignar_carga(ev, t, c_charger, c_slot, power)
                            ultima_asignacion[ev] = (t, c_slot, c_charger)
                        else:
                            self._asignar_spot(ev, t, None, c_slot, 0)
                            ultima_asignacion[ev] = (t, c_slot, None)
                        
                        # Intentar reubicar al candidato
                        if slots_sin_cargador:
                            slot = slots_sin_cargador.pop(0)
                            self._asignar_spot(candidato, t, None, slot, 0)
                            ultima_asignacion[candidato] = (t, slot, None)
            
            # f. Tercera pasada: intentar swaps para mejorar asignación
            self._intentar_swap_mejorado(t, ultima_asignacion)
        
        # g. Fase de consolidación: rellenar "huecos" en las sesiones de carga
        self._consolidar_sesiones_carga()

    def _consolidar_sesiones_carga(self):
        """
        Fase de consolidación que intenta rellenar "huecos" en las sesiones de carga.
        Busca intervalos donde un EV podría cargar pero no lo hace, estando entre dos intervalos de carga.
        """
        print("Fase 2: Optimizando continuidad de sesiones de carga...")

        # Agrupar el schedule por vehículo y ordenar por tiempo
        sesiones_por_ev = {}
        for ev_id, t_idx, charger_id, slot, power in self.schedule:
            if ev_id not in sesiones_por_ev:
                sesiones_por_ev[ev_id] = []
            sesiones_por_ev[ev_id].append((t_idx, charger_id, slot, power))

        # Para cada EV, ordenar sus asignaciones por tiempo
        for ev_id in sesiones_por_ev:
            sesiones_por_ev[ev_id].sort(key=lambda x: x[0])

        # Identificar huecos en las sesiones de carga
        cambios_realizados = 0

        for ev_id, sesiones in sesiones_por_ev.items():
            # Solo para EVs que aún necesitan energía
            if self.energy_remaining[ev_id] <= 0:
                continue

            # Identificar huecos: intervalos donde el EV está en el parking pero sin cargador
            for i in range(1, len(sesiones)):
                t_prev, charger_prev, slot_prev, _ = sesiones[i - 1]
                t_curr, charger_curr, slot_curr, _ = sesiones[i]

                # Si hay un hueco (más de un intervalo de diferencia)
                if t_curr - t_prev > 1:
                    # Intentar rellenar los intervalos en el hueco
                    for t in range(t_prev + 1, t_curr):
                        # Verificar si el EV está presente en este intervalo
                        if self.arrival_time[ev_id] <= self.times[t] < self.departure_time[ev_id]:
                            # Si no tiene asignación en este intervalo
                            if ev_id not in self.current_ev_assignment[t]:
                                # Buscar un cargador disponible
                                for c in self.charger_ids:
                                    if c not in self.occupied_chargers[t]:
                                        # Buscar un slot disponible
                                        for s in range(self.n_spots):
                                            if s not in self.occupied_spots[t]:
                                                # Calcular potencia posible
                                                power = min(
                                                    self.max_charger_power[c],
                                                    self.energy_remaining[ev_id] / self.dt,
                                                    self.station_limit - self.power_used[t]
                                                )

                                                if power > 0:
                                                    self._asignar_carga(ev_id, t, c, s, power)
                                                    cambios_realizados += 1
                                                    break  # Encontrado un slot

                                        break  # Encontrado un cargador

        if cambios_realizados > 0:
            print(f"Optimización completada: {cambios_realizados} asignaciones adicionales realizadas")
        else:
            print("Optimización completada: no se requirieron asignaciones adicionales")

    def _calcular_urgencia(self, ev_id, t_idx):
        """
        Calcula la urgencia de carga para un EV con prioridad mejorada.
        
        Args:
            ev_id: ID del vehículo
            t_idx: Índice de tiempo actual
            
        Returns:
            float: Valor de urgencia
        """
        # Tiempo restante hasta la salida (en horas)
        tiempo_hasta_salida = max(0.25, (self.departure_time[ev_id] - self.times[t_idx]))
        
        # Energía restante por entregar
        energia_restante = self.energy_remaining[ev_id]
        
        # Factor de completitud - para priorizar vehículos que están más cerca de completar su carga
        nivel_completitud = self.energy_delivered[ev_id] / (self.required_energy[ev_id] + 0.001)
        factor_completitud = 1 + nivel_completitud  # Bonificación máxima del 100%
        
        # Factor de ventana de tiempo - penalizar vehículos con ventanas muy cortas
        tiempo_total = max(0.25, (self.departure_time[ev_id] - self.arrival_time[ev_id]))
        factor_ventana = min(1.5, 1 + 2 * (tiempo_hasta_salida / tiempo_total))  # Bonificación mayor al inicio de la ventana
        
        # Urgencia base: energía restante / tiempo restante
        urgencia_base = energia_restante / tiempo_hasta_salida if tiempo_hasta_salida > 0 else float('inf')
        
        # Urgencia ajustada con los factores adicionales
        return urgencia_base * factor_completitud * factor_ventana


    def _asignar_carga(self, ev_id, t_idx, charger_id, slot, power):
        """
        Asigna carga a un EV en un intervalo de tiempo.
        
        Args:
            ev_id: ID del vehículo
            t_idx: Índice de tiempo
            charger_id: ID del cargador
            slot: Número de plaza
            power: Potencia asignada
        """
        self.schedule.append((ev_id, t_idx, charger_id, slot, power))
        self.occupied_spots[t_idx].add(slot)
        self.occupied_chargers[t_idx].add(charger_id)
        self.power_used[t_idx] += power
        eff = self.efficiency[ev_id] * self.charger_efficiency.get(charger_id, 1.0)
        energy = power * self.dt * eff

        self.energy_delivered[ev_id] += energy
        self.energy_remaining[ev_id] -= energy
        self.current_ev_assignment[t_idx][ev_id] = (slot, charger_id, power)

    def _asignar_spot(self, ev_id, t_idx, charger_id, slot, power):
        """
        Asigna un spot (sin carga) a un EV.
        
        Args:
            ev_id: ID del vehículo
            t_idx: Índice de tiempo
            charger_id: ID del cargador (None si no hay cargador)
            slot: Número de plaza
            power: Potencia asignada (0 para spots sin cargador)
        """
        self.schedule.append((ev_id, t_idx, charger_id, slot, power))
        self.occupied_spots[t_idx].add(slot)
        if charger_id is not None:
            self.occupied_chargers[t_idx].add(charger_id)
        self.current_ev_assignment[t_idx][ev_id] = (slot, charger_id, power)

    def _quitar_asignacion(self, ev_id, t_idx):
        """
        Quita la asignación de un EV en un intervalo de tiempo.
        
        Args:
            ev_id: ID del vehículo
            t_idx: Índice de tiempo
        """
        if ev_id in self.current_ev_assignment[t_idx]:
            slot, charger_id, power = self.current_ev_assignment[t_idx][ev_id]
            
            # Buscar la asignación en el schedule
            for i, (e, t, c, s, p) in enumerate(self.schedule):
                if e == ev_id and t == t_idx:
                    self.schedule.pop(i)
                    break
            
            # Liberar recursos
            if slot in self.occupied_spots[t_idx]:
                self.occupied_spots[t_idx].remove(slot)
            
            if charger_id is not None and charger_id in self.occupied_chargers[t_idx]:
                self.occupied_chargers[t_idx].remove(charger_id)
                self.power_used[t_idx] -= power
                
                # Restaurar energía
                eff = self.efficiency[ev_id] * self.charger_efficiency.get(charger_id, 1.0)
                energy = power * self.dt * eff

                self.energy_delivered[ev_id] -= energy
                self.energy_remaining[ev_id] += energy
            
            # Eliminar de la asignación actual
            del self.current_ev_assignment[t_idx][ev_id]

    def _perturba_solucion_mejorado(self, schedule, estrategia):
        """
        Aplica una perturbación a la solución según la estrategia, con mejoras para mantener continuidad.
        
        Args:
            schedule: Lista de tuplas (ev_id, t_idx, charger_id, slot, power)
            estrategia: Tipo de perturbación a aplicar
            
        Returns:
            Lista de tuplas con el schedule perturbado
        """
        nuevo_schedule = schedule.copy()
        
        if estrategia == "Perturbacion_Intervalos":
            # Seleccionar aleatoriamente k intervalos consecutivos (1 ≤ k ≤ 4)
            k = random.randint(1, 4)
            
            # Seleccionar un punto de inicio aleatorio
            if len(self.times) > k:
                start_idx = random.randint(0, len(self.times) - k)
                intervalos = list(range(start_idx, start_idx + k))
            else:
                intervalos = list(range(len(self.times)))
            
            # Liberar un porcentaje de las asignaciones en esos intervalos (no todas)
            asignaciones_por_intervalo = {}
            for intervalo in intervalos:
                asignaciones_por_intervalo[intervalo] = []
            
            for (e, t, c, s, p) in nuevo_schedule:
                if t in intervalos:
                    asignaciones_por_intervalo[t].append((e, t, c, s, p))
            
            # Eliminar entre 30% y 70% de las asignaciones en cada intervalo
            for intervalo in intervalos:
                if asignaciones_por_intervalo[intervalo]:
                    n = len(asignaciones_por_intervalo[intervalo])
                    n_eliminar = random.randint(int(0.3 * n), int(0.7 * n)) if n > 2 else 1
                    a_eliminar = random.sample(asignaciones_por_intervalo[intervalo], n_eliminar)
                    
                    nuevo_schedule = [
                        (e, t, c, s, p) for (e, t, c, s, p) in nuevo_schedule
                        if not (t == intervalo and (e, t, c, s, p) in a_eliminar)
                    ]
            
        elif estrategia == "Perturbacion_Vehiculos":
            # Seleccionar aleatoriamente m vehículos (10-30% del total)
            m = max(1, int(random.uniform(0.1, 0.3) * len(self.ev_ids)))
            vehiculos = random.sample(self.ev_ids, m)
            
            # Eliminar todas las asignaciones de estos vehículos
            nuevo_schedule = [
                (e, t, c, s, p) for (e, t, c, s, p) in nuevo_schedule
                if e not in vehiculos
            ]
            
        elif estrategia == "Perturbacion_Hibrida":
            # Seleccionar aleatoriamente una ventana temporal
            if len(self.times) <= 3:
                window_start = 0
                window_size = len(self.times)
            else:
                window_size = random.randint(2, min(8, len(self.times) // 2))
                window_start = random.randint(0, len(self.times) - window_size)
                    
            ventana = list(range(window_start, window_start + window_size))
            
            # Identificar k vehículos presentes en esa ventana
            evs_en_ventana = set()
            for (e, t, c, s, p) in nuevo_schedule:
                if t in ventana:
                    evs_en_ventana.add(e)
            
            k = max(1, len(evs_en_ventana) // 3)
            if evs_en_ventana:
                vehiculos = random.sample(list(evs_en_ventana), min(k, len(evs_en_ventana)))
                    
                # Liberar sus asignaciones solo en esa ventana
                nuevo_schedule = [
                    (e, t, c, s, p) for (e, t, c, s, p) in nuevo_schedule
                    if not (e in vehiculos and t in ventana)
                ]
                
        elif estrategia == "Perturbacion_Sesiones":
            # Nueva estrategia: perturbación centrada en sesiones de carga
            
            # 1. Identificar sesiones de carga (secuencias continuas)
            sesiones_por_ev = {}
            
            for (e, t, c, s, p) in nuevo_schedule:
                if c is not None and p > 0:  # Solo considerar intervalos con carga real
                    if e not in sesiones_por_ev:
                        sesiones_por_ev[e] = []
                    sesiones_por_ev[e].append((t, c, s, p))
            
            # Ordenar por tiempo y agrupar en sesiones continuas
            sesiones_continuas = []
            
            for ev, intervalos in sesiones_por_ev.items():
                intervalos.sort(key=lambda x: x[0])
                sesion_actual = []
                
                for i, (t, c, s, p) in enumerate(intervalos):
                    if i == 0 or t == intervalos[i-1][0] + 1:
                        # Continúa la sesión actual
                        sesion_actual.append((ev, t, c, s, p))
                    else:
                        # Nueva sesión
                        if sesion_actual:
                            sesiones_continuas.append(sesion_actual)
                        sesion_actual = [(ev, t, c, s, p)]
                
                if sesion_actual:
                    sesiones_continuas.append(sesion_actual)
            
            # Seleccionar aleatoriamente algunas sesiones para eliminar
            if sesiones_continuas:
                n_sesiones = min(5, len(sesiones_continuas))
                sesiones_a_eliminar = random.sample(sesiones_continuas, n_sesiones)
                
                # Convertir las sesiones seleccionadas a formato de tuplas
                tuplas_a_eliminar = []
                for sesion in sesiones_a_eliminar:
                    for (ev, t, c, s, p) in sesion:
                        tuplas_a_eliminar.append((ev, t, c, s, p))
                
                # Filtrar el schedule
                nuevo_schedule = [
                    (e, t, c, s, p) for (e, t, c, s, p) in nuevo_schedule
                    if (e, t, c, s, p) not in tuplas_a_eliminar
                ]
            
        return nuevo_schedule

    def _reconstruye_solucion(self, schedule_parcial):
        """
        Reconstruye una solución parcial tras una perturbación.
        
        Args:
            schedule_parcial: Lista de tuplas con el schedule perturbado
            
        Returns:
            Lista de tuplas con el schedule reconstruido
        """
        # Guardar el schedule actual
        schedule_original = self.schedule.copy()
        
        # Resetear el estado del sistema
        self.schedule = []
        self.occupied_spots = {t: set() for t in range(len(self.times))}
        self.occupied_chargers = {t: set() for t in range(len(self.times))}
        self.power_used = {t: 0 for t in range(len(self.times))}
        self.energy_delivered = {ev_id: 0 for ev_id in self.ev_ids}
        self.energy_remaining = {ev_id: self.required_energy[ev_id] for ev_id in self.ev_ids}
        self.current_ev_assignment = {t: {} for t in range(len(self.times))}
        
        # Primero, aplicar el schedule parcial
        for (ev_id, t_idx, charger_id, slot, power) in schedule_parcial:
            if slot not in self.occupied_spots[t_idx] and (charger_id is None or charger_id not in self.occupied_chargers[t_idx]):
                if charger_id is not None:
                    self._asignar_carga(ev_id, t_idx, charger_id, slot, power)
                else:
                    self._asignar_spot(ev_id, t_idx, charger_id, slot, power)
        
        # Luego, ejecutar la fase constructiva para llenar los huecos
        self._fase_construccion_inicial()
        
        # Obtener el nuevo schedule
        nuevo_schedule = self.schedule.copy()
        
        # Restaurar el estado original
        self.schedule = schedule_original
        
        return nuevo_schedule

    def _evaluar_costo(self, schedule):
        """
        Evalúa el costo total de un schedule con penalización mejorada.
        
        Args:
            schedule: Lista de tuplas (ev_id, t_idx, charger_id, slot, power)
            
        Returns:
            float: Costo total del schedule
        """
        # Contador de energía entregada por EV
        energia_entregada = defaultdict(float)
        
        # Calcular costos por intervalo y energía entregada
        costo_operacion = 0.0
        
        for (ev_id, t_idx, charger_id, slot, power) in schedule:
            if charger_id is not None and power > 0:
                eff = self.efficiency[ev_id] * self.charger_efficiency.get(charger_id, 1.0)
                energia = power * self.dt * eff
                costo_operacion += energia * self.prices[t_idx]
                energia_entregada[ev_id] += energia
        
        # Penalización mejorada por energía no entregada
        penalizacion_base = self.penalizacion_base
        costo_penalizacion = 0.0
        
        for ev_id in self.ev_ids:
            energia_requerida = self.required_energy[ev_id]
            energia_no_entregada = max(0, energia_requerida - energia_entregada[ev_id])
            
            if energia_no_entregada > 0:
                # Penalización progresiva: mayor cuanto mayor sea el porcentaje de energía no entregada
                # Esto incentiva entregar al menos algo de energía a todos los vehículos
                porcentaje_no_entregado = energia_no_entregada / energia_requerida
                factor_penalizacion = penalizacion_base * porcentaje_no_entregado * self.priority[ev_id] * self.willingness_to_pay[ev_id]

                
                costo_penalizacion += energia_no_entregada * factor_penalizacion
        
        return self.alpha_cost * costo_operacion + self.alpha_satisfaction * costo_penalizacion


    def get_resultados(self):
        """
        Retorna los resultados del schedule.
        
        Returns:
            dict: Diccionario con los resultados
        """
        # Ordenar schedule por EV y tiempo
        schedule_ordenado = sorted(self.schedule, key=lambda x: (x[0], x[1]))
        
        # Convertir a formato de intervalos para cada EV
        resultado = defaultdict(list)
        for (ev_id, t_idx, charger_id, slot, power) in schedule_ordenado:
            # Solo incluir intervalos con carga real
            if charger_id is not None and power > 0:
                t_start = self.times[t_idx]
                t_end = t_start + self.dt
                resultado[ev_id].append([t_start, t_end, charger_id, slot, power])
        
        # Calcular estadísticas
        total_energia_requerida = sum(self.required_energy.values())
        total_energia_entregada = sum(self.energy_delivered.values())
        costo_total = self._evaluar_costo(self.schedule)
        
        evs_completos = sum(1 for ev_id in self.ev_ids 
                            if abs(self.energy_delivered[ev_id] - self.required_energy[ev_id]) < 1e-6)
        evs_served = sum(1 for ev_id in self.ev_ids 
                    if self.energy_delivered[ev_id] > 0)
        
        # Preparar estadísticas base
        estadisticas = {
            "costo_total": costo_total,
            "energia_requerida_total": total_energia_requerida,
            "energia_entregada_total": total_energia_entregada,
            "porcentaje_satisfaccion": (total_energia_entregada / total_energia_requerida * 100) if total_energia_requerida > 0 else 100,
            "evs_totales": len(self.ev_ids),
            "evs_satisfechos_completamente": evs_completos,
            "evs_con_alguna_carga": evs_served,
            "porcentaje_evs_satisfechos": (evs_completos / len(self.ev_ids) * 100) if self.ev_ids else 100,
            "porcentaje_evs_con_alguna_carga": (evs_served / len(self.ev_ids) * 100) if self.ev_ids else 100,
            "evs_rechazados_por_capacidad": len(getattr(self, 'rejected_vehicles', set()))
        }
        
        # Crear detalles de vehículos rechazados por capacidad
        rejected_ids = sorted(list(getattr(self, 'rejected_vehicles', set())))
        
        # Crear detalles de vehículos con energía parcial o no entregada
        rejected_details = {}
        for ev_id in self.ev_ids:
            if ev_id in getattr(self, 'rejected_vehicles', set()):
                # Para vehículos rechazados por capacidad
                rejected_details[str(ev_id)] = {
                    "required_energy": self.required_energy[ev_id],
                    "delivered_energy": 0,
                    "unmet_energy": self.required_energy[ev_id],
                    "penalty_cost": self.required_energy[ev_id] * 1000,  # Penalización estándar
                    "rejected_by_capacity": True
                }
            elif self.energy_delivered[ev_id] < self.required_energy[ev_id]:
                # Para vehículos con entrega parcial de energía
                unmet = self.required_energy[ev_id] - self.energy_delivered[ev_id]
                rejected_details[str(ev_id)] = {
                    "required_energy": self.required_energy[ev_id],
                    "delivered_energy": self.energy_delivered[ev_id],
                    "unmet_energy": unmet,
                    "penalty_cost": unmet * self.penalizacion_base * self.priority[ev_id] * self.willingness_to_pay[ev_id],
                    "rejected_by_capacity": False
                }
        
        # Estructura de información extra
        extra_info = {
            "rejected_details": rejected_details,
            "rejected_by_capacity": {
                "count": len(getattr(self, 'rejected_vehicles', set())),
                "ids": rejected_ids
            }
        }
        
        return {
            "schedule": dict(resultado),
            "estadisticas": estadisticas,
            "extra_info": extra_info
        }


def save_schedule_to_json(resultado, file_path="resultados_heuristica.json"): 
    """ Guarda los resultados en un archivo JSON. """ 
    with open(file_path, "w") as f: json.dump(resultado, f, indent=4) 
    print(f"Resultados guardados en {file_path}")

def plot_charging_schedule(config, schedule): 
    """ Grafica los perfiles de carga de los EVs.

    Args:
        config: Configuración del sistema
        schedule: Diccionario con el schedule de carga {ev_id: [(t_start, t_end, charger, slot, power),...]}
    """
    times = config["times"]
    prices = config["prices"]
    dt = config["dt"]

    # Crear una figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

    # Crear una paleta de colores para los EVs
    num_evs = len(schedule)
    cmap = plt.cm.get_cmap('tab20', min(20, num_evs))

    # Para cada EV, graficar su perfil de carga
    for idx, (ev_id, intervals) in enumerate(schedule.items()):
        # Ordenar los intervalos por tiempo
        intervals_sorted = sorted(intervals, key=lambda x: x[0])
        
        # Extraer tiempos y potencias
        time_points = [interval[0] for interval in intervals_sorted]
        power_values = [interval[4] for interval in intervals_sorted]
        
        # Graficar el perfil de carga
        color_idx = idx % 20
        ax1.step(time_points, power_values, where='post', 
                label=f'EV {ev_id}', color=cmap(color_idx))

    # Graficar los precios de energía
    ax2.step(times, prices, where='post', color='red', label='Precio de energía')
    ax2.set_xlabel('Tiempo (horas)')
    ax2.set_ylabel('Precio ($/kWh)')
    ax2.grid(True)

    ax1.set_title(f'Perfiles de carga de vehículos eléctricos ({num_evs} EVs)')
    ax1.set_ylabel('Potencia (kW)')
    ax1.grid(True)

    if num_evs <= 20:
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    else:
        ax1.text(0.99, 0.99, f'Total: {num_evs} EVs', 
                transform=ax1.transAxes, ha='right', va='top',
                bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig
def plot_parking_schedule(config, schedule): 
    """ Grafica la asignación de plazas de parqueo.
    Args:
        config: Configuración del sistema
        schedule: Diccionario con el schedule de carga {ev_id: [(t_start, t_end, charger, slot, power),...]}
    """
    times = config["times"]
    n_spots = config["n_spots"]
    dt = config["dt"]

    total_time = times[-1] + dt if times else dt

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("tab20")

    ev_ids = list(schedule.keys())
    ev_colors = {ev: cmap(i % 20) for i, ev in enumerate(ev_ids)}

    for ev_id, intervals in schedule.items():
        for interval in intervals:
            t_start, t_end, _, slot, _ = interval
            ax.broken_barh([(t_start, t_end - t_start)], (slot - 0.4, 0.8),
                        facecolors=ev_colors[ev_id])
            ax.text(t_start + (t_end - t_start)/2, slot, f"EV {ev_id}", color='white',
                ha="center", va="center", fontsize=8)

    ax.set_xlabel("Tiempo (horas)")
    ax.set_ylabel("Plazas de parqueo")
    ax.set_title("Asignación de EVs a plazas de parqueo en el tiempo")
    ax.set_xlim(0, total_time)
    ax.set_ylim(-0.5, n_spots - 0.5)
    ax.set_yticks(range(n_spots))
    ax.set_yticklabels([str(i) for i in range(n_spots)])
    plt.grid(True)
    plt.tight_layout()
    return fig

def plot_cost_evolution(costos_por_iteracion, title=None): 
    """ Grafica la evolución del costo durante la optimización.
    Args:
        costos_por_iteracion: Lista de costos en cada iteración
        title: Título opcional para la gráfica
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(costos_por_iteracion)), costos_por_iteracion, 'b-', linewidth=2)
    ax.set_xlabel('Iteración')
    ax.set_ylabel('Costo')
    ax.set_title(title or 'Evolución del costo durante la optimización')
    ax.grid(True)
    plt.tight_layout()
    return fig
def main(json_path, output_path=None, plot_results=True): 
    """ Ejecuta el algoritmo HeurísticaConstructivaEVs sin optimización local.
    Args:
        json_path: Ruta al archivo JSON con los datos del sistema
        output_path: Ruta donde guardar los resultados (opcional)
        plot_results: Si es True, genera y muestra gráficas
    """
    print(f"Cargando datos desde {json_path}...")
    config = load_data(json_path)

    print(f"Datos cargados: {len(config['arrivals'])} vehículos, {config['n_spots']} plazas, {len(config['chargers'])} cargadores")

    heuristica = HeuristicaConstructivaEVs(config)
    instance_name = os.path.splitext(os.path.basename(json_path))[0]

    # Medir el tiempo de ejecución de la heurística
    start_heuristic = time.time()
    heuristica, selected_params, assigned_cluster, features = modify_heuristic_with_best_params(heuristica, config, instance_name)
    print("Hiperparámetros verificados en main:")
    print(selected_params)

    result = heuristica.run(track_progress=plot_results)
    time_heuristic = time.time() - start_heuristic

    if plot_results:
        schedule, costos_por_iteracion = result
    else:
        schedule = result

    resultado = heuristica.get_resultados()

    stats = resultado["estadisticas"]
    print("\nEstadísticas de la solución:")
    print(f"- Costo total: ${stats['costo_total']:.2f}")
    print(f"- Energía requerida total: {stats['energia_requerida_total']:.2f} kWh")
    print(f"- Energía entregada total: {stats['energia_entregada_total']:.2f} kWh")
    print(f"- Porcentaje de satisfacción: {stats['porcentaje_satisfaccion']:.2f}%")
    print(f"- EVs satisfechos completamente: {stats['evs_satisfechos_completamente']}/{stats['evs_totales']} ({stats['porcentaje_evs_satisfechos']:.2f}%)")
    print(f"- Porcentaje de EVs con alguna carga: {stats['porcentaje_evs_con_alguna_carga']:.2f}%")
    print(f"- EVs rechazados por capacidad: {stats.get('evs_rechazados_por_capacidad', 0)}")

    print(f"- Tiempo de ejecución: {time_heuristic:.2f} segundos")
    # Si no se especifica output_path, guardar en el directorio "heuristic_results"
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        output_dir = "../../results/heuristic_results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, f"{base_name}_resultado.json")
    else:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # Calcular rejected_details para cada EV no satisfecho
    rejected_details = {}
    for ev_id in heuristica.ev_ids:
        required = heuristica.required_energy[ev_id]
        delivered = heuristica.energy_delivered[ev_id]
        if delivered < required:
            unmet = required - delivered
            rejected_details[str(ev_id)] = {
                "required_energy": required,
                "delivered_energy": delivered,
                "unmet_energy": unmet,
                "penalty_cost": unmet * 1000
            }

    # Agregar extra_info al resultado
    resultado["extra_info"] = {
        "rejected_details": rejected_details,
        "time": time_heuristic
    }

    save_schedule_to_json(resultado, output_path)

    # Generar gráficos si se está procesando una única instancia
    if plot_results:
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        
        # Graficar perfiles de carga
        fig_charge = plot_charging_schedule(config, resultado["schedule"])
        fig_charge.savefig(os.path.join(output_dir, f"{base_name}_carga.png"))
        plt.close(fig_charge)
        
        # Graficar asignación de parqueo
        fig_parking = plot_parking_schedule(config, resultado["schedule"])
        fig_parking.savefig(os.path.join(output_dir, f"{base_name}_parqueo.png"))
        plt.close(fig_parking)
        
        # Graficar evolución del costo (si se registró el progreso)
        if plot_results and 'costos_por_iteracion' in locals():
            fig_cost = plot_cost_evolution(costos_por_iteracion, f"Evolución del costo - {base_name}")
            fig_cost.savefig(os.path.join(output_dir, f"{base_name}_evolucion_costo.png"))
            plt.close(fig_cost)

    return resultado


def procesar_todos_los_datos(data_dir="./data", output_dir="./resultados", compare_results=True): 
    """ Procesa todos los archivos JSON en un directorio y genera resultados para cada uno.
    Args:
        data_dir: Directorio con los archivos JSON de entrada
        output_dir: Directorio donde guardar los resultados
        compare_results: Si es True, genera una tabla comparativa de los resultados
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    json_files = glob.glob(os.path.join(data_dir, "*.json"))

    if not json_files:
        print(f"No se encontraron archivos JSON en {data_dir}")
        return

    print(f"Se encontraron {len(json_files)} archivos JSON para procesar")

    resultados_comparativos = []

    for i, json_file in enumerate(json_files):
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        output_path = os.path.join(output_dir, f"{base_name}_resultado.json")
        print(f"\n[{i+1}/{len(json_files)}] Procesando {base_name}...")
        
        try:
            start_time = time.time()
            resultado = main(json_file, output_path, plot_results=True)
            elapsed_time = time.time() - start_time
            
            stats = resultado["estadisticas"]
            resultados_comparativos.append({
                "archivo": base_name,
                "costo_total": stats["costo_total"],
                "energia_requerida": stats["energia_requerida_total"],
                "energia_entregada": stats["energia_entregada_total"],
                "porcentaje_satisfaccion": stats["porcentaje_satisfaccion"],
                "evs_totales": stats["evs_totales"],
                "evs_satisfechos": stats["evs_satisfechos_completamente"],
                "porcentaje_evs_satisfechos": stats["porcentaje_evs_satisfechos"],
                "tiempo_ejecucion": elapsed_time
            })
            
            print(f"Completado en {elapsed_time:.2f} segundos")
        except Exception as e:
            print(f"Error procesando {json_file}: {str(e)}")

    if compare_results and resultados_comparativos:
        print("\n=== TABLA COMPARATIVA DE RESULTADOS ===")
        print(f"{'Archivo':<20} {'Costo Total':>12} {'Satisfacción':>12} {'EVs Completados':>15} {'Tiempo (s)':>10}")
        print("-" * 80)
        
        for r in resultados_comparativos:
            print(f"{r['archivo']:<20} {r['costo_total']:>12.2f} {r['porcentaje_satisfaccion']:>11.2f}% {r['evs_satisfechos']:>6}/{r['evs_totales']} {r['tiempo_ejecucion']:>10.2f}")
        
        csv_path = os.path.join(output_dir, "resultados_comparativos.csv")
        with open(csv_path, "w") as f:
            f.write("archivo,costo_total,energia_requerida,energia_entregada,porcentaje_satisfaccion,evs_totales,evs_satisfechos,porcentaje_evs_satisfechos,tiempo_ejecucion\n")
            for r in resultados_comparativos:
                f.write(f"{r['archivo']},{r['costo_total']},{r['energia_requerida']},{r['energia_entregada']},{r['porcentaje_satisfaccion']},{r['evs_totales']},{r['evs_satisfechos']},{r['porcentaje_evs_satisfechos']},{r['tiempo_ejecucion']}\n")
        
        print(f"\nTabla comparativa guardada en {csv_path}")

    return resultados_comparativos

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso:")
        print("1. Para procesar un solo archivo:")
        print("   python heuristica_constructiva_evs.py <archivo_json> [archivo_salida]")
        print("2. Para procesar todos los archivos en un directorio:")
        print("   python heuristica_constructiva_evs.py --all [directorio_datos] [directorio_salida]")
        sys.exit(1)

    if sys.argv[1] == "--all":
        data_dir = sys.argv[2] if len(sys.argv) > 2 else "./data"
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "./resultados"
        procesar_todos_los_datos(data_dir, output_dir)
    else:
        json_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        main(json_path, output_path, plot_results=True)