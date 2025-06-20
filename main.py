import pandas as pd
from utils.pln import analizar_respuesta, obtener_embedding
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Cargar respuestas
df = pd.read_csv('data/ejemplo_respuestas_60frases.csv')  # Cada fila: una persona, cada columna: una frase

embeddings = []
tiempos_verbales = []

for index, row in df.iterrows():
    respuestas = row[1:]  # Suponiendo que la columna 0 es un ID
    vectores = [obtener_embedding(r) for r in respuestas]
    tiempos = [analizar_respuesta(r) for r in respuestas]

    promedio_embedding = sum(vectores) / len(vectores)
    embeddings.append(promedio_embedding)
    tiempos_verbales.append(max(set(tiempos), key=tiempos.count))  # Tiempo más frecuente

# Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(embeddings)

# Visualización rápida
pca = PCA(n_components=2)
coords = pca.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='viridis')
plt.title('Clustering de Rasgos de Personalidad')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.colorbar(label='Cluster')
plt.show()
