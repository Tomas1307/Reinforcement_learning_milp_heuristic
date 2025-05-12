from modules.constructive.heuristic import HeuristicaConstructivaEVs


class HeuristicaPorPrecio(HeuristicaConstructivaEVs):
    """
    Implementación de una heurística constructiva que prioriza la carga
    en intervalos con precios más bajos de energía.
    """
    
    def __init__(self, config):
        """
        Inicializa la heurística con la configuración del sistema.
        """
        super().__init__(config)
        
        # Ordenar los intervalos de tiempo por precio (de menor a mayor)
        self.intervalos_ordenados_por_precio = sorted(
            range(len(self.times)), 
            key=lambda t_idx: self.prices[t_idx]
        )
        
    def _fase_construccion_inicial(self):
        """
        Implementa una fase de construcción inicial que prioriza cargar
        en intervalos de tiempo con precios más bajos.
        """
        print("Fase 1: Construcción inicial priorizando intervalos de bajo costo...")
        
        # Inicializar seguimiento de sesiones de carga
        ultima_asignacion = {ev_id: None for ev_id in self.ev_ids}
        
        # Procesar primero los intervalos con precios más bajos
        for t in self.intervalos_ordenados_por_precio:
            # Identificar vehículos presentes en este intervalo
            ev_presentes = [
                ev_id for ev_id in self.ev_ids
                if self.arrival_time[ev_id] <= self.times[t] < self.departure_time[ev_id]
                and self.energy_remaining[ev_id] > 0  # Solo considerar EVs que necesitan carga
            ]
            
            if not ev_presentes:
                continue
            
            # Ordenar EVs por energía requerida (de mayor a menor)
            ev_ordenados = sorted(
                ev_presentes,
                key=lambda ev: self.energy_remaining[ev],
                reverse=True
            )
            
            # Identificar recursos disponibles para asignaciones
            slots_con_cargador = []
            for s in range(self.n_spots):
                if s not in self.occupied_spots[t]:
                    for c in self.charger_ids:
                        if c not in self.occupied_chargers[t]:
                            slots_con_cargador.append((s, c))
            
            slots_sin_cargador = [
                s for s in range(self.n_spots)
                if s not in self.occupied_spots[t] and s not in [sc[0] for sc in slots_con_cargador]
            ]
            
            # Asignar EVs a slots con cargador (priorizar los que necesitan más energía)
            for ev in ev_ordenados:
                if ev in self.current_ev_assignment[t]:
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
                    # Asignar a slot sin cargador
                    slot = slots_sin_cargador.pop(0)
                    self._asignar_spot(ev, t, None, slot, 0)
                    ultima_asignacion[ev] = (t, slot, None)
        
        # Fase de consolidación: intentar llenar espacios vacíos
        self._consolidar_sesiones_carga()
    
    def _calcular_urgencia(self, ev_id, t_idx):
        """
        Sobreescribe el cálculo de urgencia para priorizar por precio.
        Menor precio = mayor prioridad.
        
        Args:
            ev_id: ID del vehículo
            t_idx: Índice de tiempo actual
            
        Returns:
            float: Valor de urgencia (inverso al precio)
        """
        # Inversamente proporcional al precio
        precio_normalizado = self.prices[t_idx] / max(self.prices)
        
        # Energía restante por entregar (para desempate)
        energia_restante = self.energy_remaining[ev_id]
        
        # También considerar el tiempo restante de estancia
        tiempo_hasta_salida = max(0.25, (self.departure_time[ev_id] - self.times[t_idx]))
        
        # A menor precio, mayor urgencia
        return (1.0 / precio_normalizado) * (energia_restante / tiempo_hasta_salida)
    

