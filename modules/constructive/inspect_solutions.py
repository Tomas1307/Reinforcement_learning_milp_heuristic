"""
Script para inspeccionar soluciones generadas por diferentes métodos
para identificar el problema de diversidad.
"""

import json
import sys
import os
import pickle


from modules.constructive.heuristic import load_data, HeuristicaConstructivaEVs
from modules.constructive.price_heuristic import HeuristicaPorPrecio
from modules.constructive.time_slot_heuristic import HeuristicaPorVentanaTiempo
from modules.metaheuristic.scatter_search import generar_solucion_aleatoria

def inspect_solution(solution, name):
    """
    Inspecciona una solución y muestra información detallada sobre su estructura.
    """
    print(f"\n===== Inspeccionando solución {name} =====")
    
    if solution is None:
        print("¡SOLUCIÓN ES NONE!")
        return
    
    print(f"Tipo: {type(solution)}")
    
    if hasattr(solution, '__len__'):
        print(f"Longitud: {len(solution)}")
    else:
        print("No tiene longitud definida")
    
    if not solution:
        print("¡SOLUCIÓN ESTÁ VACÍA!")
        return
    
    # Si es una lista o tupla, inspeccionar el primer elemento
    if isinstance(solution, (list, tuple)):
        print(f"Primer elemento: {solution[0]}")
        print(f"Tipo del primer elemento: {type(solution[0])}")
        
        # Si el primer elemento es una tupla, inspeccionarlo más detalladamente
        if isinstance(solution[0], tuple):
            print(f"Longitud del primer elemento: {len(solution[0])}")
            for i, val in enumerate(solution[0]):
                print(f"  Componente {i}: {val}, Tipo: {type(val)}")
    
    # Contar elementos None
    if isinstance(solution, list):
        none_count = sum(1 for item in solution if item is None)
        print(f"Número de elementos None: {none_count}")
        
        # Verificar específicamente problemas con charger_id (índice 2) siendo None
        if solution and isinstance(solution[0], tuple) and len(solution[0]) >= 3:
            charger_none_count = sum(1 for item in solution if item[2] is None)
            print(f"Número de elementos con charger_id=None: {charger_none_count}")
    
    # Identificar si hay elementos que no son tuplas de 5 elementos
    if isinstance(solution, list):
        irregular_items = [
            (i, item) for i, item in enumerate(solution)
            if not isinstance(item, tuple) or len(item) != 5
        ]
        
        if irregular_items:
            print("¡ENCONTRADOS ELEMENTOS IRREGULARES!")
            for i, item in irregular_items[:5]:  # Mostrar solo los primeros 5
                print(f"  Índice {i}: {item}, Tipo: {type(item)}")
            if len(irregular_items) > 5:
                print(f"  ... y {len(irregular_items) - 5} más")

def main():
    """
    Genera soluciones utilizando diferentes métodos y las inspecciona.
    """
    # Definir la ruta al archivo de configuración
    config_path = "./results/cluster_results/representative_files/test_system_2.json"
    
    # Cargar datos
    print(f"Cargando datos desde {config_path}")
    try:
        data = load_data(config_path)
        print("Datos cargados correctamente")
    except Exception as e:
        print(f"Error al cargar datos: {str(e)}")
        return
    
    # Generar soluciones con diferentes métodos
    solutions = {}
    
    # 1. Heurística constructiva
    try:
        print("\nGenerando solución con heurística constructiva...")
        heur = HeuristicaConstructivaEVs(data)
        solutions['constructiva'] = heur.run(track_progress=False)
    except Exception as e:
        print(f"Error al generar solución constructiva: {str(e)}")
    
    # 2. Heurística por precio
    try:
        print("\nGenerando solución con heurística por precio...")
        heur = HeuristicaPorPrecio(data)
        solutions['precio'] = heur.run(track_progress=False)
    except Exception as e:
        print(f"Error al generar solución por precio: {str(e)}")
    
    # 3. Heurística por ventana de tiempo
    try:
        print("\nGenerando solución con heurística por ventana de tiempo...")
        heur = HeuristicaPorVentanaTiempo(data)
        solutions['ventana'] = heur.run(track_progress=False)
    except Exception as e:
        print(f"Error al generar solución por ventana: {str(e)}")
    
    # 4. Solución aleatoria
    try:
        print("\nGenerando solución aleatoria...")
        solutions['aleatoria'] = generar_solucion_aleatoria(data)
    except Exception as e:
        print(f"Error al generar solución aleatoria: {str(e)}")
    
    # Inspeccionar cada solución
    for name, solution in solutions.items():
        inspect_solution(solution, name)
    
    # Intentar calcular diversidad entre pares de soluciones
    print("\n===== Probando cálculo de diversidad =====")
    try:
        from modules.metaheuristic.scatter_search import measure_diversity
        
        for name1, sol1 in solutions.items():
            for name2, sol2 in solutions.items():
                if name1 != name2:
                    try:
                        print(f"Calculando diversidad entre {name1} y {name2}...")
                        div = measure_diversity(sol1, sol2)
                        print(f"  Diversidad: {div}")
                    except Exception as e:
                        print(f"  ERROR: {str(e)}")
    except Exception as e:
        print(f"Error al importar measure_diversity: {str(e)}")
    
    # Guardar soluciones para análisis posterior
    output_dir = "debug_output"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "solutions.pickle"), "wb") as f:
        pickle.dump(solutions, f)
    
    print(f"\nSoluciones guardadas en {output_dir}/solutions.pickle")
    
    # También guardar una versión JSON (sin pickle, para facilitar la inspección)
    solution_info = {}
    for name, solution in solutions.items():
        if solution:
            # Tomar solo los primeros 5 elementos para el JSON
            sample = [list(item) if isinstance(item, tuple) else item for item in solution[:5]]
            solution_info[name] = {
                "longitud": len(solution),
                "muestra": sample
            }
        else:
            solution_info[name] = {
                "longitud": 0,
                "muestra": None
            }
    
    with open(os.path.join(output_dir, "solutions_info.json"), "w") as f:
        json.dump(solution_info, f, indent=2)
    
    print(f"Información resumida guardada en {output_dir}/solutions_info.json")
    
if __name__ == "__main__":
    main()