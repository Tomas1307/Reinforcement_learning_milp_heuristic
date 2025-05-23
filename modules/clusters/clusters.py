import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from collections import defaultdict
import seaborn as sns
from scipy.spatial.distance import cdist


def load_instance(json_path):
    """
    Carga una instancia desde un archivo JSON y extrae características para clustering.
    
    Args:
        json_path: Ruta al archivo JSON con los datos de la instancia
        
    Returns:
        dict: Diccionario con las características extraídas
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extraer características clave
    energy_prices = data.get("energy_prices", [])
    arrivals = data.get("arrivals", [])
    parking_config = data.get("parking_config", {})
    
    # Calcular estadísticas de precios
    if energy_prices:
        prices = [ep.get("price", 0) for ep in energy_prices]
        price_mean = np.mean(prices)
        price_std = np.std(prices)
        price_min = np.min(prices)
        price_max = np.max(prices)
        price_range = price_max - price_min
        price_volatility = price_std / price_mean if price_mean > 0 else 0
    else:
        price_mean = price_std = price_min = price_max = price_range = price_volatility = 0
    
    # Calcular estadísticas de llegadas
    n_evs = len(arrivals)
    
    if arrivals:
        # Tiempos de llegada y salida
        arrival_times = [ev.get("arrival_time", 0) for ev in arrivals]
        departure_times = [ev.get("departure_time", 0) for ev in arrivals]
        
        # Ventanas de tiempo
        time_windows = [
            max(0, dep - arr) 
            for arr, dep in zip(arrival_times, departure_times)
        ]
        
        mean_time_window = np.mean(time_windows)
        std_time_window = np.std(time_windows)
        min_time_window = np.min(time_windows)
        max_time_window = np.max(time_windows)
        
        # Energía requerida
        required_energies = [ev.get("required_energy", 0) for ev in arrivals]
        total_energy = sum(required_energies)
        mean_energy = np.mean(required_energies)
        std_energy = np.std(required_energies)
        max_energy = np.max(required_energies)
        
        # Distribución temporal
        if arrival_times:
            min_arrival = min(arrival_times)
            max_departure = max(departure_times)
            total_horizon = max_departure - min_arrival
            
            # Calcular concurrencia (cuántos EVs están presentes en cada momento)
            resolution = 0.25  # horas
            time_points = np.arange(min_arrival, max_departure + resolution, resolution)
            concurrency = [
                sum(1 for arr, dep in zip(arrival_times, departure_times) 
                    if arr <= t < dep)
                for t in time_points
            ]
            
            max_concurrency = max(concurrency)
            mean_concurrency = np.mean(concurrency)
            
            # Calcular congestión
            n_spots = parking_config.get("n_spots", 0)
            congestion_ratio = max_concurrency / n_spots if n_spots > 0 else float('inf')
        else:
            total_horizon = 0
            max_concurrency = mean_concurrency = 0
            congestion_ratio = 0
    else:
        mean_time_window = std_time_window = min_time_window = max_time_window = 0
        total_energy = mean_energy = std_energy = max_energy = 0
        total_horizon = 0
        max_concurrency = mean_concurrency = 0
        congestion_ratio = 0
    
    # Características de infraestructura
    n_spots = parking_config.get("n_spots", 0)
    n_chargers = len(parking_config.get("chargers", []))
    charger_powers = [c.get("power", 0) for c in parking_config.get("chargers", [])]
    total_charger_power = sum(charger_powers)
    mean_charger_power = np.mean(charger_powers) if charger_powers else 0
    transformer_limit = parking_config.get("transformer_limit", 0)
    
    # Calcular características derivadas
    chargers_per_spot = n_chargers / n_spots if n_spots > 0 else 0
    power_per_spot = total_charger_power / n_spots if n_spots > 0 else 0
    energy_demand_per_spot = total_energy / n_spots if n_spots > 0 else 0
    
    # Complejidad estimada
    complexity_factor = (n_evs * congestion_ratio * price_volatility) / (chargers_per_spot + 0.001)
    
    # Crear diccionario de características
    instance_features = {
        "instance_name": os.path.basename(json_path),
        # Características de demanda
        "n_evs": n_evs,
        "total_energy": total_energy,
        "mean_energy": mean_energy,
        "std_energy": std_energy,
        "max_energy": max_energy,
        "mean_time_window": mean_time_window,
        "std_time_window": std_time_window,
        "min_time_window": min_time_window,
        "max_time_window": max_time_window,
        # Características temporales
        "total_horizon": total_horizon,
        "max_concurrency": max_concurrency,
        "mean_concurrency": mean_concurrency,
        "congestion_ratio": congestion_ratio,
        # Características de precios
        "price_mean": price_mean,
        "price_std": price_std,
        "price_min": price_min,
        "price_max": price_max,
        "price_range": price_range,
        "price_volatility": price_volatility,
        # Características de infraestructura
        "n_spots": n_spots,
        "n_chargers": n_chargers,
        "chargers_per_spot": chargers_per_spot,
        "total_charger_power": total_charger_power,
        "mean_charger_power": mean_charger_power,
        "transformer_limit": transformer_limit,
        "power_per_spot": power_per_spot,
        "energy_demand_per_spot": energy_demand_per_spot,
        # Indicadores derivados
        "complexity_factor": complexity_factor
    }
    
    return instance_features


def extract_features_from_instances(data_dir="./data"):
    """
    Extrae características de todas las instancias JSON en un directorio.
    
    Args:
        data_dir: Directorio que contiene los archivos JSON de instancias
        
    Returns:
        DataFrame: Características de cada instancia
    """
    # Buscar archivos JSON
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    
    if not json_files:
        raise ValueError(f"No se encontraron archivos JSON en {data_dir}")
    
    print(f"Procesando {len(json_files)} archivos JSON...")
    
    # Extraer características de cada instancia
    features_list = []
    for json_path in json_files:
        try:
            features = load_instance(json_path)
            features_list.append(features)
        except Exception as e:
            print(f"Error procesando {json_path}: {str(e)}")
    
    # Crear DataFrame
    features_df = pd.DataFrame(features_list)
    
    print(f"Características extraídas para {len(features_df)} instancias")
    
    return features_df


def find_optimal_k(X, max_k=15):
    n_samples = X.shape[0]
    max_k = min(max_k, n_samples - 1)  # Limitar k a n_samples - 1
    inertias = []
    silhouette_scores = []
    k_values = range(2, max_k + 1)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        
        silhouette_avg = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)
    
    # Método del codo
    differences = np.diff(inertias)
    second_differences = np.diff(differences)
    elbow_index = np.argmax(np.abs(second_differences)) + 1  # +1 para ajustar por diff
    elbow_k = list(k_values)[elbow_index]
    
    # Método de la silueta
    silhouette_k = list(k_values)[np.argmax(silhouette_scores)]
    
    # Gráficas de diagnóstico (sin cambios)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(list(k_values), inertias, 'bo-')
    plt.plot(elbow_k, inertias[elbow_index], 'ro', markersize=10)
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Inercia')
    plt.title('Método del Codo')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(list(k_values), silhouette_scores, 'go-')
    plt.plot(silhouette_k, silhouette_scores[silhouette_k - 2], 'ro', markersize=10)
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Puntuación de silueta')
    plt.title('Método de la Silueta')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('optimal_k_methods.png')
    plt.close()
    
    print(f"Método del codo sugiere k = {elbow_k}")
    print(f"Método de la silueta sugiere k = {silhouette_k}")
    
    return elbow_k, silhouette_k


def cluster_instances_function(features_df, k=None, features_to_use=None):
    """
    Agrupa las instancias en k clusters usando K-means.
    
    Args:
        features_df: DataFrame con características de instancias
        k: Número de clusters (si es None, se determina automáticamente)
        features_to_use: Lista de características a usar (si es None, se usan todas excepto el nombre)
    
    Returns:
        DataFrame: Datos originales con etiquetas de clusters y distancias al centroide
    """
    # Seleccionar características para clustering
    if features_to_use is None:
        # Usar todas las características numéricas
        features_to_use = [col for col in features_df.columns 
                         if col != 'instance_name' and pd.api.types.is_numeric_dtype(features_df[col])]
    
    print(f"Usando {len(features_to_use)} características para clustering")
    
    # Extraer matriz de características
    X = features_df[features_to_use].values
    
    # Normalizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determinar k óptimo si no se proporciona
    if k is None:
        elbow_k, silhouette_k = find_optimal_k(X_scaled)
        # Podemos usar el promedio de ambos métodos o el valor del método de silueta
        k = silhouette_k
    
    # Aplicar K-means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Calcular distancias a los centroides
    centroids = kmeans.cluster_centers_
    distances = cdist(X_scaled, centroids, 'euclidean')
    
    # Obtener la distancia de cada punto a su centroide
    distances_to_centroid = np.min(distances, axis=1)
    
    # Añadir etiquetas y distancias al DataFrame original
    result_df = features_df.copy()
    result_df['clusters'] = cluster_labels
    result_df['distance_to_centroid'] = distances_to_centroid
    
    # Calcular estadísticas por clusters
    cluster_stats = result_df.groupby('clusters').agg({
        'n_evs': ['mean', 'min', 'max'],
        'congestion_ratio': ['mean', 'min', 'max'],
        'price_volatility': ['mean', 'min', 'max'],
        'complexity_factor': ['mean', 'min', 'max'],
        'instance_name': 'count'
    })
    
    print("\nEstadísticas por clusters:")
    print(cluster_stats)
    
    # Mostrar los centroides
    print("\nCentroides de los clusters:")
    centroids_df = pd.DataFrame(
        scaler.inverse_transform(centroids),
        columns=features_to_use
    )
    print(centroids_df)
    
    return result_df


def select_representative_instances(clustered_df, n_per_cluster=1):
    """
    Selecciona instancias representativas de cada clusters.
    
    Args:
        clustered_df: DataFrame con instancias agrupadas
        n_per_cluster: Número de instancias a seleccionar por clusters
    
    Returns:
        DataFrame: Instancias representativas seleccionadas
    """
    representatives = []
    
    # Para cada clusters
    for cluster in clustered_df['clusters'].unique():
        cluster_instances = clustered_df[clustered_df['clusters'] == cluster]
        
        # Ordenar por distancia al centroide (ascendente)
        sorted_instances = cluster_instances.sort_values('distance_to_centroid')
        
        # Seleccionar las n instancias más cercanas al centroide
        selected = sorted_instances.head(n_per_cluster)
        representatives.append(selected)
    
    # Combinar todas las instancias seleccionadas
    representative_df = pd.concat(representatives)
    
    print(f"\nSeleccionadas {len(representative_df)} instancias representativas:")
    for cluster in representative_df['clusters'].unique():
        instances = representative_df[representative_df['clusters'] == cluster]
        print(f"Cluster {cluster}: {len(instances)} instancias")
        for idx, row in instances.iterrows():
            print(f"  - {row['instance_name']} (distancia: {row['distance_to_centroid']:.4f})")
    
    return representative_df


def visualize_clusters(clustered_df, features_to_plot=None):
    """
    Visualiza los clusters usando PCA y gráficos de dispersión.
    
    Args:
        clustered_df: DataFrame con instancias agrupadas
        features_to_plot: Características específicas para visualizar en pares
    """
    # 1. Visualización PCA
    # Seleccionar características numéricas
    features = [col for col in clustered_df.columns 
               if col not in ['instance_name', 'clusters', 'distance_to_centroid']
               and pd.api.types.is_numeric_dtype(clustered_df[col])]
    
    # Aplicar PCA
    X = clustered_df[features].values
    X_scaled = StandardScaler().fit_transform(X)
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    
    # Crear DataFrame para visualización
    pca_df = pd.DataFrame({
        'PCA1': pca_result[:, 0],
        'PCA2': pca_result[:, 1],
        'clusters': clustered_df['clusters'],
        'instance': clustered_df['instance_name']
    })
    
    # Gráfico PCA
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    sns.scatterplot(
        x='PCA1', y='PCA2', 
        hue='clusters',
        data=pca_df,
        palette='viridis',
        s=100,
        alpha=0.7
    )
    
    # Añadir etiquetas de instancia
    for idx, row in pca_df.iterrows():
        plt.text(
            row['PCA1'] + 0.02, 
            row['PCA2'] + 0.02, 
            row['instance'],
            fontsize=8
        )
    
    plt.title('Visualización de Clusters con PCA')
    plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]:.2%} varianza)')
    plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]:.2%} varianza)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Visualización de características específicas
    if features_to_plot is None:
        # Usar algunas características importantes por defecto
        features_to_plot = ['n_evs', 'congestion_ratio', 'price_volatility', 'complexity_factor']
    
    # Asegurarse de que tenemos al menos 2 características
    if len(features_to_plot) < 2:
        features_to_plot = features[:2]
    
    # Seleccionar las primeras 2 características para el gráfico
    x_feature, y_feature = features_to_plot[:2]
    
    plt.subplot(2, 1, 2)
    sns.scatterplot(
        x=x_feature, y=y_feature, 
        hue='clusters',
        data=clustered_df,
        palette='viridis',
        s=100,
        alpha=0.7
    )
    
    # Añadir etiquetas de instancia
    for idx, row in clustered_df.iterrows():
        plt.text(
            row[x_feature] + 0.02 * row[x_feature], 
            row[y_feature] + 0.02 * row[y_feature], 
            row['instance_name'],
            fontsize=8
        )
    
    plt.title(f'Clusters por {x_feature} vs {y_feature}')
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('cluster_visualization.png')
    plt.close()
    
    # 3. Matriz de gráficos de dispersión
    if len(features_to_plot) > 2:
        plt.figure(figsize=(15, 15))
        scatter_matrix = pd.plotting.scatter_matrix(
            clustered_df[features_to_plot + ['clusters']],
            c=clustered_df['clusters'],
            figsize=(15, 15), 
            marker='o',
            hist_kwds={'bins': 20},
            alpha=0.8,
            cmap='viridis'
        )
        
        plt.suptitle('Matriz de Dispersión de Características por Cluster', size=16)
        plt.savefig('cluster_scatter_matrix.png')
        plt.close()
    
    print("Visualizaciones guardadas como 'cluster_visualization.png' y 'cluster_scatter_matrix.png'")


def analyze_feature_importance(clustered_df):
    """
    Analiza la importancia de las características en la formación de clusters.
    
    Args:
        clustered_df: DataFrame con instancias agrupadas
    """
    # Seleccionar características numéricas
    features = [col for col in clustered_df.columns 
               if col not in ['instance_name', 'clusters', 'distance_to_centroid']
               and pd.api.types.is_numeric_dtype(clustered_df[col])]
    
    # Calcular estadísticas por clusters
    cluster_stats = {}
    
    for feature in features:
        cluster_means = clustered_df.groupby('clusters')[feature].mean()
        feature_mean = clustered_df[feature].mean()
        feature_std = clustered_df[feature].std()
        
        # Calcular diferencia normalizada respecto a la media global
        normalized_diff = np.abs(cluster_means - feature_mean) / feature_std if feature_std > 0 else np.abs(cluster_means - feature_mean)
        
        # Guardar la máxima diferencia para esta característica
        cluster_stats[feature] = normalized_diff.max()
    
    # Ordenar características por importancia
    sorted_features = sorted(cluster_stats.items(), key=lambda x: x[1], reverse=True)
    
    print("\nImportancia de características para la formación de clusters:")
    for feature, importance in sorted_features:
        print(f"{feature}: {importance:.4f}")
    
    # Visualizar importancia de características
    plt.figure(figsize=(12, 8))
    feature_names = [f[0] for f in sorted_features]
    importance_values = [f[1] for f in sorted_features]
    
    # Invertir orden para visualización
    feature_names = feature_names[::-1]
    importance_values = importance_values[::-1]
    
    plt.barh(feature_names, importance_values, color='teal')
    plt.xlabel('Importancia Relativa')
    plt.title('Importancia de Características en la Formación de Clusters')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    print("Visualización de importancia guardada como 'feature_importance.png'")
    
    return dict(sorted_features)


def main():
    """
    Función principal que ejecuta el análisis completo.
    """
    # Configuración
    import argparse
    parser = argparse.ArgumentParser(description="Clustering de instancias de carga de EVs")
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directorio con archivos JSON de instancias')
    parser.add_argument('--k', type=int, default=None,
                        help='Número de clusters (si no se especifica, se determina automáticamente)')
    parser.add_argument('--n_per_cluster', type=int, default=1,
                        help='Número de instancias representativas por clusters')
    parser.add_argument('--output_dir', type=str, default='./results/cluster_results',
                        help='Directorio para guardar resultados')
    
    args = parser.parse_args()
    
    # Crear directorio de salida si no existe
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    print(f"Analizando instancias en {args.data_dir}")
    
    # Paso 1: Extraer características
    features_df = extract_features_from_instances(args.data_dir)
    
    # Guardar características extraídas
    features_path = os.path.join(args.output_dir, 'instance_features.txt')
    with open(features_path, 'w') as f:
        f.write("Características de instancias\n")
        f.write("===========================\n\n")
        
        for idx, row in features_df.iterrows():
            f.write(f"Instancia: {row['instance_name']}\n")
            for col in features_df.columns:
                if col != 'instance_name':
                    f.write(f"  {col}: {row[col]}\n")
            f.write("\n")
    
    print(f"Características guardadas en {features_path}")
    
    # Paso 2: Aplicar clustering
    # Seleccionar características más relevantes para el clustering
    key_features = [
        'n_evs', 'total_energy', 'mean_energy', 'std_energy',
        'mean_time_window', 'std_time_window', 'congestion_ratio',
        'price_volatility', 'chargers_per_spot', 'power_per_spot',
        'energy_demand_per_spot', 'complexity_factor'
    ]
    
    # Llamar a la función correctamente - usando la definición de arriba
    clustered_df = cluster_instances_function(features_df, k=args.k, features_to_use=key_features)
    
    # Guardar resultados del clustering
    clustered_path = os.path.join(args.output_dir, 'clustered_instances.txt')
    with open(clustered_path, 'w') as f:
        f.write("Resultados del clustering\n")
        f.write("========================\n\n")
        
        for cluster in sorted(clustered_df['clusters'].unique()):
            f.write(f"Cluster {cluster}:\n")
            cluster_data = clustered_df[clustered_df['clusters'] == cluster]
            
            for idx, row in cluster_data.iterrows():
                f.write(f"  {row['instance_name']} (distancia: {row['distance_to_centroid']:.4f})\n")
            f.write("\n")

    
    print(f"Resultados del clustering guardados en {clustered_path}")
    
    # Paso 3: Seleccionar instancias representativas
    representatives_df = select_representative_instances(clustered_df, n_per_cluster=args.n_per_cluster)
    
    # Guardar instancias representativas
    representatives_path = os.path.join(args.output_dir, 'representative_instances.txt')
    with open(representatives_path, 'w') as f:
        f.write("Instancias representativas\n")
        f.write("========================\n\n")
        
        for cluster in sorted(representatives_df['clusters'].unique()):
            f.write(f"Cluster {cluster}:\n")
            rep_cluster_data = representatives_df[representatives_df['clusters'] == cluster]
            
            for idx, row in rep_cluster_data.iterrows():
                f.write(f"  {row['instance_name']} (distancia: {row['distance_to_centroid']:.4f})\n")
                f.write(f"    n_evs: {row['n_evs']}\n")
                f.write(f"    congestion_ratio: {row['congestion_ratio']:.2f}\n")
                f.write(f"    complexity_factor: {row['complexity_factor']:.2f}\n")
            f.write("\n")


    
    print(f"Instancias representativas guardadas en {representatives_path}")
    
    # Copiar los archivos representativos a un directorio específico
    representatives_dir = os.path.join(args.output_dir, 'representative_files')
    if not os.path.exists(representatives_dir):
        os.makedirs(representatives_dir)
    
    for instance_name in representatives_df['instance_name']:
        source_path = os.path.join(args.data_dir, instance_name)
        if os.path.exists(source_path):
            import shutil
            dest_path = os.path.join(representatives_dir, instance_name)
            shutil.copy2(source_path, dest_path)
            print(f"Copiado {instance_name} a {representatives_dir}")
    
    # Paso 4: Visualizar clusters
    visualize_clusters(clustered_df)
    
    # Paso 5: Analizar importancia de características
    feature_importance = analyze_feature_importance(clustered_df)
    
    # Crear informe completo
    report_path = os.path.join(args.output_dir, 'clustering_report.md')
    with open(report_path, 'w') as f:
        f.write("# Informe de Clustering de Instancias\n\n")
        
        # Información general
        f.write(f"## Información General\n\n")
        f.write(f"- Total de instancias analizadas: {len(features_df)}\n")
        f.write(f"- Número de clusters: {len(clustered_df['clusters'].unique())}\n")
        f.write(f"- Instancias representativas por clusters: {args.n_per_cluster}\n\n")
        
        # Distribución de clusters
        f.write(f"## Distribución de Clusters\n\n")
        cluster_counts = clustered_df['clusters'].value_counts().sort_index()
        for cluster, count in cluster_counts.items():
            f.write(f"- Cluster {cluster}: {count} instancias\n")
        f.write("\n")
        
        # Características más importantes
        f.write(f"## Características Más Importantes\n\n")
        for i, (feature, importance) in enumerate(feature_importance.items()):
            if i < 10:  # Top 10 características
                f.write(f"{i+1}. {feature}: {importance:.4f}\n")
        f.write("\n")
        
        # Instancias representativas
        f.write(f"## Instancias Representativas\n\n")
        for cluster in representatives_df['clusters'].unique():
            f.write(f"### Cluster {cluster}\n\n")
            cluster_instances = representatives_df[representatives_df['clusters'] == cluster]
            for idx, row in cluster_instances.iterrows():
                f.write(f"- **{row['instance_name']}**\n")
                f.write(f"  - Distancia al centroide: {row['distance_to_centroid']:.4f}\n")
                f.write(f"  - Número de EVs: {row['n_evs']}\n")
                f.write(f"  - Congestión: {row['congestion_ratio']:.2f}\n")
                f.write(f"  - Complejidad: {row['complexity_factor']:.2f}\n\n")
    
    print(f"Informe completo guardado en {report_path}")


if __name__ == "__main__":
    main()