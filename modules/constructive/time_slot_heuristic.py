from modules.constructive.heuristic import HeuristicaConstructivaEVs

class HeuristicaPorVentanaTiempo(HeuristicaConstructivaEVs):
    """
    Implementación de una heurística constructiva que prioriza la carga
    de vehículos con ventanas de tiempo más cortas.
    """
    
    def __init__(self, config):
        """
        Inicializa la heurística con la configuración del sistema.
        """
        super().__init__(config)
        
        # Calcular duración de ventana para cada EV
        self.duracion_ventana = {
            ev_id: self.departure_time[ev_id] - self.arrival_time[ev_id] 
            for ev_id in self.ev_ids
        }
        
    def _fase_construccion_inicial(self):
        """
        Implementa una fase de construcción inicial que prioriza vehículos
        con ventanas de tiempo más cortas.
        """
        print("Fase 1: Construcción inicial priorizando ventanas de tiempo cortas...")
        
        # Ordenar los EVs por duración de ventana (de menor a mayor)
        evs_ordenados_por_ventana = sorted(
            self.ev_ids,
            key=lambda ev: self.duracion_ventana[ev]
        )
        
        # Para cada EV, asignar carga en sus intervalos disponibles
        for ev in evs_ordenados_por_ventana:
            # Intervalos de tiempo disponibles para este EV
            intervalos_disponibles = [
                t for t in range(len(self.times))
                if self.arrival_time[ev] <= self.times[t] < self.departure_time[ev]
            ]
            
            # Ordenar intervalos por precio (menor a mayor)
            intervalos_ordenados = sorted(
                intervalos_disponibles,
                key=lambda t: self.prices[t]
            )
            
            # Energía que necesitamos entregar
            energia_requerida = self.energy_remaining[ev]
            
            # Intentar asignar carga en cada intervalo
            for t in intervalos_ordenados:
                if energia_requerida <= 0:
                    break
                
                # Buscar un slot con cargador disponible
                for s in range(self.n_spots):
                    if s in self.occupied_spots[t]:
                        continue
                    
                    for c in self.charger_ids:
                        if c in self.occupied_chargers[t]:
                            continue
                        
                        # Calcular potencia
                        power = min(
                            self.max_charger_power[c],
                            energia_requerida / self.dt,
                            self.station_limit - self.power_used[t]
                        )
                        
                        if power > 0:
                            self._asignar_carga(ev, t, c, s, power)
                            energia_requerida -= power * self.dt
                            break
                    
                    if s not in self.occupied_spots[t]:  # Si no se asignó carga
                        # Asignar a slot sin cargador
                        self._asignar_spot(ev, t, None, s, 0)
                    
                    break  # Pasar al siguiente intervalo
        
        # Intentar asignar el resto de vehículos a spots
        for ev in self.ev_ids:
            if self.energy_delivered[ev] == 0:  # No ha recibido carga
                # Identificar intervalos disponibles
                for t in range(len(self.times)):
                    if self.arrival_time[ev] <= self.times[t] < self.departure_time[ev]:
                        # Si no tiene asignación en este intervalo
                        if ev not in self.current_ev_assignment[t]:
                            # Buscar un spot libre
                            for s in range(self.n_spots):
                                if s not in self.occupied_spots[t]:
                                    self._asignar_spot(ev, t, None, s, 0)
                                    break
        
        # Optimizar las asignaciones
        self._optimizar_asignaciones()
    
    def _optimizar_asignaciones(self):
        """
        Intenta mejorar las asignaciones reasignando cargadores.
        """
        print("Fase 2: Optimizando asignaciones...")
        
        cambios = 0
        
        # Para cada intervalo de tiempo
        for t in range(len(self.times)):
            # Identificar EVs que están en spots sin cargador pero necesitan energía
            evs_sin_cargador = []
            for ev, asignacion in self.current_ev_assignment[t].items():
                slot, charger, power = asignacion
                if charger is None and self.energy_remaining[ev] > 0:
                    evs_sin_cargador.append(ev)
            
            # Ordenar por duración de ventana (menor primero)
            evs_sin_cargador.sort(key=lambda ev: self.duracion_ventana[ev])
            
            for ev in evs_sin_cargador:
                # Buscar un cargador disponible
                for c in self.charger_ids:
                    if c not in self.occupied_chargers[t]:
                        # Obtener el slot actual
                        slot = self.current_ev_assignment[t][ev][0]
                        
                        # Quitar asignación actual
                        self._quitar_asignacion(ev, t)
                        
                        # Asignar carga
                        power = min(
                            self.max_charger_power[c],
                            self.energy_remaining[ev] / self.dt,
                            self.station_limit - self.power_used[t]
                        )
                        
                        if power > 0:
                            self._asignar_carga(ev, t, c, slot, power)
                            cambios += 1
                        else:
                            # Restaurar asignación sin cargador
                            self._asignar_spot(ev, t, None, slot, 0)
                        
                        break
        
        print(f"Optimización completada: {cambios} reasignaciones realizadas")