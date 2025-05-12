# Pseudocódigo: Heurística Constructiva para Programación de Carga de Vehículos Eléctricos

## Algoritmo: HeurísticaConstructivaEVs

### Entrada:
- Conjunto de vehículos EVs con tiempos de llegada/salida y requisitos energéticos
- Conjunto de cargadores con capacidades máximas
- Precios de energía por intervalo
- Número de plazas de parqueo (con y sin cargador)
- Límite del transformador
- Delta de tiempo (dt)

### Salida:
- Schedule de carga optimizado para cada vehículo

## Procedimiento Principal

### 1. Inicialización
```
Inicializar:
    Schedule = []
    Estado_cargadores[t] = disponible para todo t
    Estado_slots[t] = disponible para todo t
    Potencia_usada[t] = 0 para todo t
    Energía_entregada[ev] = 0 para todo ev
    Energía_restante[ev] = required_energy[ev] para todo ev
    Asignaciones_actuales[t] = {} para todo t  // Mapeo de {ev: (slot, cargador, potencia)}
    Última_asignación[ev] = null para todo ev  // Seguimiento de continuidad: (t, slot, cargador)
```

### 2. Fase de Construcción Inicial
```
Para cada intervalo de tiempo t en orden cronológico:
    // a. Identificar vehículos presentes
    EV_presentes = {ev | arrival_time[ev] ≤ t < departure_time[ev]}
    
    // b. Ordenar vehículos con criterio de prioridad que considera:
    //    - Urgencia base (energía restante / tiempo restante)
    //    - Nivel de completitud de carga
    //    - Fase en la ventana de tiempo
    //    - Bonificación por continuidad de carga
    EV_ordenados = ordenar EV_presentes por criterio_prioridad(ev) descendente
    
    // c. Primera pasada: Extender sesiones de carga activas
    EV_ya_asignados = {}
    
    Para cada ev en EV_ordenados:
        Si ev tenía carga en t-1:
            Intentar mantenerlo en el mismo cargador/slot
            Si se logra, marcar como ya asignado
    
    // d. Identificar recursos disponibles
    slots_con_cargador = []  // Pares (slot, cargador) disponibles
    slots_sin_cargador = []  // Slots sin cargador disponibles
    
    // e. Segunda pasada: Nuevas asignaciones para vehículos sin continuidad
    Para cada ev en EV_ordenados que no esté ya asignado:
        Si ev necesita energía y hay slots con cargador:
            Asignar a slot con cargador con potencia óptima
        Sino si hay slots sin cargador:
            Asignar a slot sin cargador
        Sino:
            Buscar candidato para intercambio (con menor prioridad)
            Si se encuentra candidato:
                Realizar intercambio y reubicar candidato si es posible
    
    // f. Intentar intercambios adicionales para mejorar asignación
    Intentar_swap_mejorado(t, Última_asignación)

// g. Fase de consolidación: Rellenar "huecos" en sesiones de carga
Consolidar_sesiones_carga()
```

### 3. Fase de Mejora - Exploración del Espacio de Soluciones
```
schedule_actual = solución de Fase 1
mejor_schedule = schedule_actual
mejor_costo = Evaluar_costo(schedule_actual)
intentos_sin_mejora = 0
temperatura = 1.0
factor_enfriamiento = 0.95
umbral_reinicio = 20

Para i = 1 hasta max_iteraciones_mejora:
    // a. Seleccionar estrategia de perturbación
    Elegir aleatoriamente entre:
    - "Perturbación_Intervalos" (40%)
    - "Perturbación_Vehículos" (30%)
    - "Perturbación_Híbrida" (20%)
    - "Perturbación_Sesiones" (10%)
    
    // b. Aplicar perturbación
    nuevo_schedule = Perturba_solución(schedule_actual, estrategia)
    
    // c. Reconstruir solución
    nuevo_schedule = Reconstruye_solución(nuevo_schedule)
    
    // d. Evaluar nueva solución
    nuevo_costo = Evaluar_costo(nuevo_schedule)
    
    // e. Criterio de aceptación (Simulated Annealing)
    Si nuevo_costo < mejor_costo:
        mejor_schedule = nuevo_schedule
        mejor_costo = nuevo_costo
        schedule_actual = nuevo_schedule
        intentos_sin_mejora = 0
    Sino:
        // Aceptación probabilística
        delta = nuevo_costo - mejor_costo
        prob_aceptación = exp(-delta / (temperatura * mejor_costo / 1000))
        Si aleatorio(0,1) < prob_aceptación:
            schedule_actual = nuevo_schedule
        intentos_sin_mejora += 1
    
    // f. Ajustar temperatura
    temperatura *= factor_enfriamiento
    
    // g. Reinicio si estancado
    Si intentos_sin_mejora >= umbral_reinicio:
        schedule_actual = mejor_schedule
        intentos_sin_mejora = 0
        temperatura = 0.5  // Reiniciar con temperatura moderada

Retornar mejor_schedule
```

## Subprocedimientos Clave

### calcular_urgencia(ev_id, t_idx)
```
// Calcula la prioridad/urgencia de carga para un EV

// Tiempo restante hasta la salida (en horas)
tiempo_hasta_salida = max(0.25, (departure_time[ev_id] - times[t_idx]))

// Energía restante por entregar
energia_restante = Energía_restante[ev_id]

// Factor de completitud - priorizar vehículos cerca de completar su carga
nivel_completitud = Energía_entregada[ev_id] / (required_energy[ev_id] + 0.001)
factor_completitud = 1 + nivel_completitud  // Bonificación máxima del 100%

// Factor de ventana de tiempo - penalizar vehículos con ventanas muy cortas
tiempo_total = max(0.25, (departure_time[ev_id] - arrival_time[ev_id]))
factor_ventana = min(1.5, 1 + 2 * (tiempo_hasta_salida / tiempo_total))

// Urgencia base: energía restante / tiempo restante
urgencia_base = energia_restante / tiempo_hasta_salida si tiempo_hasta_salida > 0 sino infinito

// Urgencia ajustada con los factores adicionales
Retornar urgencia_base * factor_completitud * factor_ventana
```

### Asignar_carga(ev_id, t_idx, charger_id, slot, power)
```
Schedule.append((ev_id, t_idx, charger_id, slot, power))
Estado_slots[t_idx].add(slot)
Estado_cargadores[t_idx].add(charger_id)
Potencia_usada[t_idx] += power
energy = power * dt
Energía_entregada[ev_id] += energy
Energía_restante[ev_id] -= energy
Asignaciones_actuales[t_idx][ev_id] = (slot, charger_id, power)
```

### Asignar_spot(ev_id, t_idx, charger_id, slot, power)
```
// Asigna un spot sin carga
Schedule.append((ev_id, t_idx, charger_id, slot, power))
Estado_slots[t_idx].add(slot)
Si charger_id no es null:
    Estado_cargadores[t_idx].add(charger_id)
Asignaciones_actuales[t_idx][ev_id] = (slot, charger_id, power)
```

### Quitar_asignación(ev_id, t_idx)
```
Si ev_id en Asignaciones_actuales[t_idx]:
    slot, charger_id, power = Asignaciones_actuales[t_idx][ev_id]
    
    // Buscar la asignación en el Schedule
    Para i, (e, t, c, s, p) en enumerate(Schedule):
        Si e == ev_id y t == t_idx:
            Schedule.pop(i)
            break
    
    // Liberar recursos
    Si slot en Estado_slots[t_idx]:
        Estado_slots[t_idx].remove(slot)
    
    Si charger_id no es null y charger_id en Estado_cargadores[t_idx]:
        Estado_cargadores[t_idx].remove(charger_id)
        Potencia_usada[t_idx] -= power
        
        // Restaurar energía
        energy = power * dt
        Energía_entregada[ev_id] -= energy
        Energía_restante[ev_id] += energy
    
    // Eliminar de la asignación actual
    Eliminar Asignaciones_actuales[t_idx][ev_id]
```

### Intentar_swap_mejorado(t, última_asignación, umbral=5.0)
```
// Identificar EVs que necesitan carga urgente pero no tienen cargador
evs_sin_cargador = []
Para cada ev en ev_ids:
    // Solo considerar EVs presentes sin cargador asignado con necesidad de energía
    Si (está presente en t sin cargador y necesita más que el umbral de energía):
        Calcular prioridad con bonus por continuidad interrumpida
        evs_sin_cargador.append((ev, prioridad))

// Ordenar por prioridad (urgencia) descendente
evs_sin_cargador = ordenar por prioridad descendente

// Para cada EV que necesita carga urgente
Para cada (ev, _) en evs_sin_cargador:
    // Buscar candidatos con cargador que casi han terminado o baja urgencia
    ev_candidatos = []
    Para cada (ev2, asignación) en Asignaciones_actuales[t]:
        Si tiene cargador y urgencia baja o casi ha terminado:
            ev_candidatos.append((ev2, urgencia, asignación))
    
    // Ordenar candidatos por urgencia ascendente
    ev_candidatos = ordenar por urgencia ascendente
    
    // Intentar el swap con el candidato de menor urgencia
    Si hay candidatos:
        Quitar asignación del candidato
        Asignar su cargador al EV sin carga
        Intentar reubicar al candidato en un slot sin cargador
```

### Consolidar_sesiones_carga()
```
// Agrupar el schedule por vehículo y ordenar por tiempo
sesiones_por_ev = {}
// Agrupar asignaciones por vehículo

// Para cada EV, identificar huecos entre sesiones de carga
Para cada ev_id con energía restante > 0:
    // Buscar huecos (más de un intervalo consecutivo sin asignación de carga)
    Para cada hueco identificado:
        Intentar asignar carga en los intervalos del hueco
        utilizando recursos disponibles (cargadores y slots)
```

### Perturba_solución(schedule, estrategia)
```
nuevo_schedule = copiar schedule
    
Si estrategia == "Perturbación_Intervalos":
    // Seleccionar k intervalos consecutivos y eliminar 30-70% de asignaciones
    
Sino si estrategia == "Perturbación_Vehículos":
    // Seleccionar 10-30% de vehículos y eliminar todas sus asignaciones
    
Sino si estrategia == "Perturbación_Híbrida":
    // Seleccionar ventana temporal y algunos vehículos en esa ventana
    // Eliminar asignaciones de esos vehículos solo en la ventana
    
Sino si estrategia == "Perturbación_Sesiones":
    // Identificar sesiones continuas de carga
    // Eliminar aleatoriamente algunas sesiones completas

Retornar nuevo_schedule
```

### Reconstruye_solución(schedule_parcial)
```
// Resetear el estado del sistema
// Aplicar primero el schedule parcial dado
// Ejecutar fase constructiva para completar el schedule
// Retornar el nuevo schedule completo
```

### Evaluar_costo(schedule)
```
// Calcular costo de operación: suma(energía * precio del intervalo)
    
// Calcular penalización por energía no entregada:
// - Penalización mayor cuanto mayor sea el porcentaje no entregado
// - Factor de penalización proporcional al porcentaje faltante
    
Retornar costo_operacion + costo_penalizacion
```
