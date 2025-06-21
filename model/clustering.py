# model/clustering.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
import torch

def clustering_jerarquico(embeddings, n_clusters=3):
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    etiquetas = clustering.fit_predict(embeddings_scaled)
    score = silhouette_score(embeddings_scaled, etiquetas)
    print(f"Silhouette Score Clustering Jerárquico: {score:.4f}")
    return etiquetas, clustering

def agrupar_respuestas_por_participante(df, columnas_respuestas, obtener_embedding):
    """
    Convierte las respuestas en embeddings por participante.
    Devuelve lista de vectores promedio y metadata (edad, ocupación, etc.)
    """
    embeddings = []
    metadatos = []

    for idx, row in df.iterrows():
        respuestas = [str(row[col]) for col in columnas_respuestas]
        vectores = [obtener_embedding(r) for r in respuestas]
        if vectores:
            promedio = torch.stack(vectores).mean(dim=0).cpu().numpy()
            embeddings.append(promedio)
            metadatos.append(row[['Edad', 'Ocupación', 'Estado de residencia']].to_dict())

    return np.array(embeddings), pd.DataFrame(metadatos)


def realizar_clustering(embeddings, n_clusters=3):
    """
    Realiza clustering KMeans sobre los embeddings.
    """
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    etiquetas = kmeans.fit_predict(embeddings_scaled)

    return etiquetas, kmeans


def visualizar_clusters(embeddings, etiquetas, output_path):
    """
    Reduce los datos a 2D y guarda un gráfico con la distribución de clústeres.
    """
    pca = PCA(n_components=2)
    componentes = pca.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(componentes[:, 0], componentes[:, 1], c=etiquetas, cmap='viridis')
    plt.title("Clustering de Rasgos de Personalidad")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    cbar = plt.colorbar(scatter)
    cbar.set_label('Cluster')
    plt.tight_layout()

    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, "clustering_personalidad.png"))
    plt.close()


def exportar_resultados(metadatos, etiquetas, output_path):
    """
    Guarda los metadatos junto con los clústeres asignados.
    """
    if not isinstance(metadatos, pd.DataFrame):
        metadatos = pd.DataFrame(metadatos)

    resultado_df = metadatos.copy()
    resultado_df["Cluster"] = etiquetas
    resultado_df.to_csv(os.path.join(output_path, "resultado_clusters.csv"), index=False)

def graficar_dendrograma(embeddings, output_path="output/resultados/"):
    """
    Genera y guarda un dendrograma jerárquico para ayudar a decidir el número de clusters.
    """
    linkage_matrix = linkage(embeddings, method='ward')  # método ward = minimiza varianza intra-cluster

    plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix, truncate_mode='level', p=10)
    plt.title("Dendrograma jerárquico")
    plt.xlabel("Índices de muestra o clústeres")
    plt.ylabel("Distancia (Ward)")
    plt.tight_layout()
    
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, "dendrograma.png"))
    plt.close()