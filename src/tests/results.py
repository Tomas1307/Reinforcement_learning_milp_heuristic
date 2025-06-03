import json
import os
from typing import Dict, Any, Optional

def load_json_safe(filepath: str) -> Optional[Dict[str, Any]]:
    """Carga un JSON de forma segura, retorna None si no existe o hay error."""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error cargando {filepath}: {e}")
    return None

def extract_system_info(system_config: Dict[str, Any]) -> Dict[str, int]:
    """Extrae información básica del sistema desde la configuración."""
    arrivals = system_config.get("arrivals", [])
    parking_config = system_config.get("parking_config", {})
    chargers = parking_config.get("chargers", [])
    
    return {
        "vehicles": len(arrivals),
        "slots": parking_config.get("n_spots", 0),
        "chargers": len(chargers)
    }

def extract_solution_metrics(solution_data: Dict[str, Any], method: str) -> Dict[str, Any]:
    """Extrae métricas de una solución según el método."""
    
    if method == "MILP":
        # Para soluciones MILP solo
        obj_values = solution_data.get("objective_values", {})
        
        return {
            "cost": obj_values.get("energy_cost", "N/A"),
            "satisfaction_pct": obj_values.get("energy_satisfaction_pct", "N/A"),
            "time": solution_data.get("solve_time", "N/A")  # Si no existe este campo en MILP solo
        }
    
    elif method == "MILP+RL":
        # Para soluciones MILP del método híbrido
        metrics = solution_data.get("metrics", {})
        obj_values = metrics.get("obj_values", {})
        
        return {
            "cost": obj_values.get("energy_cost", "N/A"),
            "satisfaction_pct": obj_values.get("energy_satisfaction_pct", "N/A"),
            "time": metrics.get("solve_time", "N/A")
        }
    
    elif method == "RL":
        # Para soluciones RL del Scatter Search
        metrics = solution_data.get("metrics", {})
        energy_metrics = metrics.get("energy_metrics", {})
        
        return {
            "cost": metrics.get("total_cost", "N/A"),
            "satisfaction_pct": energy_metrics.get("total_satisfaction_pct", "N/A"),
            "time": metrics.get("execution_time", "N/A")
        }
    
    return {"cost": "N/A", "satisfaction_pct": "N/A", "time": "N/A"}

def generate_results_table():
    """Genera la tabla de resultados comparativa."""
    
    # Paths base
    system_configs_path = "src/configs/system_data"
    milp_rl_path = "results/MILP_RL"
    milp_only_path = "results/MILP_only"
    
    # Resultados por instancia
    results = {}
    
    print("Procesando instancias...")
    
    for i in range(1, 18):  # Sistemas 1 a 17
        print(f"Procesando sistema {i}...")
        
        # 1. Cargar configuración del sistema
        system_config_file = os.path.join(system_configs_path, f"test_system_{i}.json")
        system_config = load_json_safe(system_config_file)
        
        if system_config is None:
            print(f"   No se encontró configuración para sistema {i}")
            continue
            
        system_info = extract_system_info(system_config)
        
        # Inicializar resultado para esta instancia
        results[i] = {
            "vehicles": system_info["vehicles"],
            "slots": system_info["slots"], 
            "chargers": system_info["chargers"],
            "milp_rl": {"cost": "Infactible", "satisfaction_pct": "Infactible", "time": "Infactible"},
            "milp_only": {"cost": "Infactible", "satisfaction_pct": "Infactible", "time": "Infactible"},
            "rl_only": {"cost": "Infactible", "satisfaction_pct": "Infactible", "time": "Infactible"}
        }
        
        # 2. Procesar MILP+RL (archivo scatter_milp_solution dentro de la carpeta)
        milp_rl_dir = os.path.join(milp_rl_path, f"milp_optimizer_system_{i}")
        milp_rl_file = os.path.join(milp_rl_dir, f"scatter_milp_solution_{i}.json")
        milp_rl_data = load_json_safe(milp_rl_file)
        if milp_rl_data:
            results[i]["milp_rl"] = extract_solution_metrics(milp_rl_data, "MILP+RL")
            print(f"   MILP+RL encontrado")
        else:
            print(f"   MILP+RL no encontrado en {milp_rl_file}")
        
        # 3. Procesar RL (archivo scatter_rl_solution dentro de la carpeta MILP_RL)
        rl_file = os.path.join(milp_rl_dir, f"scatter_rl_solution_{i}.json")
        rl_data = load_json_safe(rl_file)
        if rl_data:
            results[i]["rl_only"] = extract_solution_metrics(rl_data, "RL")
            print(f"   RL encontrado")
        else:
            print(f"   RL no encontrado en {rl_file}")
        
        # 5. Procesar MILP solo
        milp_only_dir = os.path.join(milp_only_path, f"milp_optimizer_system_{i}")
        if os.path.exists(milp_only_dir):
            # Buscar archivo JSON en la carpeta solutions
            solutions_dir = os.path.join(milp_only_dir, "solutions")
            if os.path.exists(solutions_dir):
                # Buscar cualquier archivo JSON en solutions
                json_files = [f for f in os.listdir(solutions_dir) if f.endswith('.json')]
                if json_files:
                    milp_only_file = os.path.join(solutions_dir, json_files[0])
                    milp_only_data = load_json_safe(milp_only_file)
                    if milp_only_data:
                        results[i]["milp_only"] = extract_solution_metrics(milp_only_data, "MILP")
                        print(f"   MILP solo encontrado: {json_files[0]}")
                    else:
                        print(f"   Error cargando MILP solo")
                else:
                    print(f"   No hay archivos JSON en solutions para sistema {i}")
            else:
                print(f"   No existe carpeta solutions para sistema {i}")
        else:
            print(f"   No existe carpeta MILP para sistema {i}")
    
    # Generar tabla
    print("\n" + "="*120)
    print("GENERANDO TABLA DE RESULTADOS")
    print("="*120)
    
    # Crear tabla en formato texto
    table_lines = []
    
    # Encabezado
    header = f"{'Instancia':<10} {'Veh':<5} {'Slots':<7} {'Carg':<6} {'MILP+RL':<25} {'MILP':<25} {'RL':<25}"
    table_lines.append(header)
    table_lines.append("-" * len(header))
    
    # Subencabezado para métricas
    subheader = f"{'':>10} {'':>5} {'':>7} {'':>6} {'Costo|Sat%|Tiempo':<25} {'Costo|Sat%|Tiempo':<25} {'Costo|Sat%|Tiempo':<25}"
    table_lines.append(subheader)
    table_lines.append("-" * len(header))
    
    # Datos
    for i in sorted(results.keys()):
        data = results[i]
        
        # Formatear métricas
        def format_metrics(metrics):
            cost = f"{metrics['cost']:.2f}" if isinstance(metrics['cost'], (int, float)) else str(metrics['cost'])
            sat = f"{metrics['satisfaction_pct']:.1f}" if isinstance(metrics['satisfaction_pct'], (int, float)) else str(metrics['satisfaction_pct'])
            time = f"{metrics['time']:.2f}" if isinstance(metrics['time'], (int, float)) else str(metrics['time'])
            return f"{cost}|{sat}|{time}"
        
        milp_rl_str = format_metrics(data["milp_rl"])
        milp_only_str = format_metrics(data["milp_only"])
        rl_only_str = format_metrics(data["rl_only"])
        
        line = f"{i:<10} {data['vehicles']:<5} {data['slots']:<7} {data['chargers']:<6} {milp_rl_str:<25} {milp_only_str:<25} {rl_only_str:<25}"
        table_lines.append(line)
    
    # Escribir tabla a archivo
    output_file = "results_comparison_table.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("TABLA COMPARATIVA DE RESULTADOS\n")
        f.write("Optimización de Carga de Vehículos Eléctricos\n")
        f.write("="*120 + "\n\n")
        
        f.write("LEYENDA:\n")
        f.write("- Instancia: ID del sistema de prueba\n")
        f.write("- Veh: Número de vehículos\n")
        f.write("- Slots: Número de espacios de estacionamiento\n")
        f.write("- Carg: Número de cargadores\n")
        f.write("- Costo: Costo energético de la solución\n")
        f.write("- Sat%: Porcentaje de satisfacción energética\n")
        f.write("- Tiempo: Tiempo de resolución (segundos)\n")
        f.write("- 'Infactible': No se encontró solución o archivo\n\n")
        
        for line in table_lines:
            f.write(line + "\n")
            print(line)
    
    print(f"\n Tabla guardada en: {output_file}")
    print(f"Procesadas {len(results)} instancias")
    
    # Estadísticas resumen
    milp_rl_count = sum(1 for r in results.values() if r["milp_rl"]["cost"] != "Infactible")
    milp_only_count = sum(1 for r in results.values() if r["milp_only"]["cost"] != "Infactible")
    rl_only_count = sum(1 for r in results.values() if r["rl_only"]["cost"] != "Infactible")
    
    print(f"\nRESUMEN:")
    print(f"- MILP+RL: {milp_rl_count}/17 instancias resueltas")
    print(f"- MILP solo: {milp_only_count}/17 instancias resueltas") 
    print(f"- RL solo: {rl_only_count}/17 instancias resueltas")

if __name__ == "__main__":
    generate_results_table()