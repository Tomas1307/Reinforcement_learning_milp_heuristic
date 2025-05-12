import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os
from itertools import combinations, product
from matplotlib import cm
import matplotlib.gridspec as gridspec

# Definir rutas de archivos
files = {
    "test_system_2": "hyperparameter/hyperparameter_results_constructive/instancia_test_system_2_grid_complete.csv",
    "test_system_9": "hyperparameter/hyperparameter_results_constructive/instancia_test_system_9_grid_complete.csv",
    "test_system_10": "hyperparameter/hyperparameter_results_constructive/instancia_test_system_10_grid_complete.csv"
}

# Directorio para guardar las visualizaciones
output_dir = "../../hyperparameter/hyperparameter_analysis_constructive"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_data():
    """Carga los datos de los archivos CSV"""
    datasets = {}
    for system_name, file_path in files.items():
        try:
            data = pd.read_csv(file_path)
            datasets[system_name] = data
            print(f"Cargado {system_name} con {len(data)} filas")
        except Exception as e:
            print(f"Error al cargar {system_name}: {e}")
    return datasets

def find_best_hyperparameters(datasets):
    """Encuentra los mejores hiperparámetros para cada instancia"""
    best_configs = {}
    summary_text = "RESUMEN DE MEJORES HIPERPARÁMETROS\n"
    summary_text += "=================================\n\n"
    
    for system_name, data in datasets.items():
        print(f"\nEstadísticas básicas para {system_name}:")
        print(f"Número de filas: {len(data)}")
        print(f"Rango de costo final: {data['final_cost'].min():.2f} - {data['final_cost'].max():.2f}")
        print(f"Rango de satisfacción de energía: {data['energy_satisfaction'].min():.2f}% - {data['energy_satisfaction'].max():.2f}%")
        print(f"Máximo número de EVs satisfechos: {data['evs_satisfied'].max()} de {data['evs_total'].max()}")
        
        # Encontrar la mejor configuración (menor costo)
        best_config_idx = data['final_cost'].idxmin()
        best_config = data.loc[best_config_idx]
        best_configs[system_name] = best_config
        
        summary_text += f"Configuración: {system_name}\n"
        summary_text += "-" * 50 + "\n"
        
        summary_text += f"Costo final: {best_config['final_cost']:.2f}\n"
        summary_text += f"Satisfacción de energía: {best_config['energy_satisfaction']:.2f}%\n"
        summary_text += f"EVs satisfechos: {best_config['evs_satisfied']}/{best_config['evs_total']} ({best_config['evs_satisfaction_percentage']:.2f}%)\n"
        summary_text += f"Tiempo de ejecución: {best_config['execution_time']:.2f} segundos\n"
        summary_text += f"Mejora de costo: {best_config['cost_improvement']:.2f}%\n\n"
        
        summary_text += "Mejores hiperparámetros:\n"
        hyperparams = [
            'max_iteraciones', 'temperatura_inicial', 'factor_enfriamiento', 'umbral_reinicio',
            'prob_perturbacion_intervalos', 'prob_perturbacion_vehiculos', 
            'factor_completitud', 'factor_ventana'
        ]
        
        for param in hyperparams:
            summary_text += f"  {param}: {best_config[param]}\n"
        
        summary_text += "\n\n"
    
    # Guardar el resumen en un archivo
    with open(os.path.join(output_dir, "best_parameters_summary.txt"), "w") as f:
        f.write(summary_text)
    
    return best_configs

def visualize_parameter_importance(datasets):
    """Visualiza la importancia de cada parámetro en el fitness"""
    hyperparams = [
        'max_iteraciones', 'temperatura_inicial', 'factor_enfriamiento', 
        'prob_perturbacion_intervalos', 'prob_perturbacion_vehiculos', 
        'factor_completitud', 'factor_ventana'
    ]
    
    for system_name, data in datasets.items():
        print(f"\nGenerando visualizaciones para {system_name}...")
        
        # Crear directorio para las visualizaciones de este sistema
        system_dir = os.path.join(output_dir, system_name)
        if not os.path.exists(system_dir):
            os.makedirs(system_dir)
        
        # 1. Visualización individual de cada parámetro
        for param in hyperparams:
            if len(data[param].unique()) > 1:  # Solo visualizar si hay variación
                plt.figure(figsize=(10, 6))
                
                # Si hay pocos valores únicos, usar boxplot
                if len(data[param].unique()) <= 5:
                    # Agrupar por valor del parámetro
                    grouped_data = []
                    param_values = sorted(data[param].unique())
                    
                    for val in param_values:
                        grouped_data.append(data[data[param] == val]['final_cost'])
                    
                    plt.boxplot(grouped_data, labels=[str(val) for val in param_values])
                    plt.title(f'Efecto de {param} en el Costo Final - {system_name}')
                    plt.xlabel(param)
                    plt.ylabel('Costo Final')
                    plt.grid(True, linestyle='--', alpha=0.7)
                else:
                    # Si hay muchos valores, usar scatter plot
                    plt.scatter(data[param], data['final_cost'], alpha=0.6)
                    plt.title(f'Efecto de {param} en el Costo Final - {system_name}')
                    plt.xlabel(param)
                    plt.ylabel('Costo Final')
                    plt.grid(True, linestyle='--', alpha=0.7)
                
                # Añadir línea de tendencia
                try:
                    z = np.polyfit(data[param], data['final_cost'], 1)
                    p = np.poly1d(z)
                    plt.plot(sorted(data[param].unique()), p(sorted(data[param].unique())), "r--", alpha=0.8)
                except:
                    pass
                
                plt.tight_layout()
                plt.savefig(os.path.join(system_dir, f'{param}_vs_cost.png'))
                plt.close()
        
        # 2. Visualizaciones de pares de parámetros (superficie 3D)
        # Tomar solo combinaciones relevantes para no generar demasiadas gráficas
        key_params = ['temperatura_inicial', 'factor_enfriamiento', 'factor_completitud', 'factor_ventana']
        for param1, param2 in combinations(key_params, 2):
            if len(data[param1].unique()) > 1 and len(data[param2].unique()) > 1:
                try:
                    fig = plt.figure(figsize=(12, 10))
                    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
                    
                    # Gráfico 3D
                    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
                    
                    # Crear una malla para interpolar
                    x_unique = sorted(data[param1].unique())
                    y_unique = sorted(data[param2].unique())
                    
                    if len(x_unique) >= 3 and len(y_unique) >= 3:  # Necesitamos al menos 3 puntos para interpolar
                        x, y = np.meshgrid(x_unique, y_unique)
                        z_data = np.zeros((len(y_unique), len(x_unique)))
                        
                        # Calcular el valor medio del costo para cada combinación de parámetros
                        for i, x_val in enumerate(x_unique):
                            for j, y_val in enumerate(y_unique):
                                subset = data[(data[param1] == x_val) & (data[param2] == y_val)]
                                if not subset.empty:
                                    z_data[j, i] = subset['final_cost'].mean()
                                else:
                                    # Si no hay datos para esta combinación, usar NaN
                                    z_data[j, i] = np.nan
                        
                        # Rellenar los valores NaN con interpolación si es posible
                        mask = np.isnan(z_data)
                        if not np.all(mask):  # Si no todos son NaN
                            z_data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), z_data[~mask])
                        
                        # Graficar superficie
                        surf = ax1.plot_surface(x, y, z_data, cmap=cm.viridis, edgecolor='none', alpha=0.8)
                        ax1.set_xlabel(param1)
                        ax1.set_ylabel(param2)
                        ax1.set_zlabel('Costo Final')
                        ax1.set_title(f'Superficie de Costo para {param1} vs {param2}')
                        fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
                    
                    # Heat map
                    ax2 = fig.add_subplot(gs[0, 1])
                    pivot_table = data.pivot_table(values='final_cost', index=param2, columns=param1, aggfunc='mean')
                    sns.heatmap(pivot_table, cmap='viridis', annot=True, fmt=".0f", linewidths=.5, ax=ax2)
                    ax2.set_title(f'Heat Map de Costo para {param1} vs {param2}')
                    
                    # Scatter plot con color por costo
                    ax3 = fig.add_subplot(gs[1, 0])
                    scatter = ax3.scatter(data[param1], data[param2], c=data['final_cost'], cmap='viridis', alpha=0.7)
                    ax3.set_xlabel(param1)
                    ax3.set_ylabel(param2)
                    ax3.set_title(f'Scatter Plot de {param1} vs {param2} coloreado por Costo')
                    fig.colorbar(scatter, ax=ax3)
                    
                    # Contour plot
                    ax4 = fig.add_subplot(gs[1, 1])
                    if len(x_unique) >= 3 and len(y_unique) >= 3:
                        contour = ax4.contourf(x, y, z_data, cmap='viridis', levels=20)
                        ax4.set_xlabel(param1)
                        ax4.set_ylabel(param2)
                        ax4.set_title(f'Contour Plot de Costo para {param1} vs {param2}')
                        fig.colorbar(contour, ax=ax4)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(system_dir, f'{param1}_{param2}_3d.png'))
                    plt.close()
                except Exception as e:
                    print(f"Error al crear gráfica 3D para {param1} vs {param2}: {e}")
        
        # 3. Visualización de métricas adicionales
        plt.figure(figsize=(10, 6))
        plt.scatter(data['execution_time'], data['final_cost'], alpha=0.6, c=data['factor_completitud'], cmap='viridis')
        plt.colorbar(label='factor_completitud')
        plt.title(f'Tiempo de Ejecución vs Costo Final - {system_name}')
        plt.xlabel('Tiempo de Ejecución (s)')
        plt.ylabel('Costo Final')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(system_dir, 'time_vs_cost.png'))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.scatter(data['improvement_iterations'], data['cost_improvement'], alpha=0.6, c=data['temperatura_inicial'], cmap='viridis')
        plt.colorbar(label='temperatura_inicial')
        plt.title(f'Iteraciones vs Mejora de Costo (%) - {system_name}')
        plt.xlabel('Iteraciones')
        plt.ylabel('Mejora de Costo (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(system_dir, 'iterations_vs_improvement.png'))
        plt.close()
        
        # 4. Correlación entre hiperparámetros y métricas
        plt.figure(figsize=(14, 12))
        correlation_columns = hyperparams + ['final_cost', 'energy_satisfaction', 'evs_satisfaction_percentage', 'cost_improvement', 'execution_time']
        correlation_data = data[correlation_columns].corr()
        sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=.5)
        plt.title(f'Matriz de Correlación - {system_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(system_dir, 'correlation_matrix.png'))
        plt.close()
        
        # 5. Crear un dashboard con los resultados más importantes
        plt.figure(figsize=(16, 20))
        gs = gridspec.GridSpec(4, 2)
        
        # Top 5 parámetros más influyentes según la correlación con el costo final
        corr_with_cost = abs(correlation_data['final_cost']).sort_values(ascending=False)
        top_params = corr_with_cost.index[:6].tolist()
        if 'final_cost' in top_params:
            top_params.remove('final_cost')  # Quitar el costo final de la lista
        top_params = top_params[:5]  # Tomar solo los 5 primeros
        
        # Crear gráficos para los parámetros más importantes
        for i, param in enumerate(top_params):
            row = i // 2
            col = i % 2
            
            ax = plt.subplot(gs[row, col])
            if len(data[param].unique()) <= 5:
                sns.boxplot(x=param, y='final_cost', data=data, ax=ax)
            else:
                sns.scatterplot(x=param, y='final_cost', data=data, ax=ax)
                
                # Añadir línea de tendencia
                try:
                    z = np.polyfit(data[param], data['final_cost'], 1)
                    p = np.poly1d(z)
                    ax.plot(sorted(data[param].unique()), p(sorted(data[param].unique())), "r--", alpha=0.8)
                except:
                    pass
            
            ax.set_title(f'Efecto de {param} en el Costo Final')
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Añadir matriz de correlación reducida
        ax_corr = plt.subplot(gs[2, :])
        sns.heatmap(correlation_data.loc[top_params, ['final_cost', 'energy_satisfaction', 'evs_satisfaction_percentage', 'cost_improvement']], 
                   annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=.5, ax=ax_corr)
        ax_corr.set_title('Correlación de Parámetros Clave con Métricas')
        
        # Añadir información de la mejor configuración
        ax_text = plt.subplot(gs[3, :])
        best_config_text = f"MEJOR CONFIGURACIÓN PARA {system_name.upper()}\n\n"
        best_config_text += f"Costo final: {best_configs[system_name]['final_cost']:.2f}\n"
        best_config_text += f"Satisfacción de energía: {best_configs[system_name]['energy_satisfaction']:.2f}%\n"
        best_config_text += f"EVs satisfechos: {best_configs[system_name]['evs_satisfied']}/{best_configs[system_name]['evs_total']} "
        best_config_text += f"({best_configs[system_name]['evs_satisfaction_percentage']:.2f}%)\n"
        best_config_text += f"Mejora de costo: {best_configs[system_name]['cost_improvement']:.2f}%\n"
        best_config_text += f"Tiempo de ejecución: {best_configs[system_name]['execution_time']:.2f} segundos\n\n"
        
        best_config_text += "Hiperparámetros óptimos:\n"
        for param in hyperparams:
            best_config_text += f"- {param}: {best_configs[system_name][param]}\n"
        
        ax_text.text(0.05, 0.95, best_config_text, transform=ax_text.transAxes, fontsize=12, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax_text.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(system_dir, 'dashboard.png'))
        plt.close()

        # 6. Crear visualización del espacio de fitness para las 4 mejores combinaciones de parámetros
        combinations_to_test = []
        for param1, param2 in combinations(key_params, 2):
            corr1 = abs(correlation_data.loc[param1, 'final_cost'])
            corr2 = abs(correlation_data.loc[param2, 'final_cost'])
            combinations_to_test.append((param1, param2, corr1 + corr2))
        
        combinations_to_test.sort(key=lambda x: x[2], reverse=True)
        top_combinations = combinations_to_test[:4]  # Las 4 mejores combinaciones
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        for i, (param1, param2, _) in enumerate(top_combinations):
            pivot_table = data.pivot_table(values='final_cost', index=param2, columns=param1, aggfunc='mean')
            im = sns.heatmap(pivot_table, cmap='viridis', annot=True, fmt=".0f", linewidths=.5, ax=axes[i])
            axes[i].set_title(f'Heat Map de Costo para {param1} vs {param2}')
            
            # Marcar el punto óptimo
            best_val1 = best_configs[system_name][param1]
            best_val2 = best_configs[system_name][param2]
            
            # Encontrar las etiquetas más cercanas en la tabla pivote
            closest_col = min(pivot_table.columns, key=lambda x: abs(x - best_val1))
            closest_idx = min(pivot_table.index, key=lambda x: abs(x - best_val2))
            
            # Marcar el punto óptimo
            axes[i].scatter(pivot_table.columns.get_loc(closest_col) + 0.5, 
                           pivot_table.index.get_loc(closest_idx) + 0.5, 
                           marker='X', color='red', s=100)
        
        plt.tight_layout()
        plt.savefig(os.path.join(system_dir, 'top_combinations_heatmap.png'))
        plt.close()
        
        print(f"Visualizaciones para {system_name} guardadas en {system_dir}")

def main():
    # Cargar los datos
    datasets = load_data()
    
    # Encontrar los mejores hiperparámetros
    global best_configs
    best_configs = find_best_hyperparameters(datasets)
    
    # Visualizar la importancia de los parámetros
    visualize_parameter_importance(datasets)
    
    print(f"\nAnálisis completo. Los resultados se han guardado en {output_dir}")

if __name__ == "__main__":
    main()