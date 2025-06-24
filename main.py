# main.py
import pandas as pd
import numpy as np
import torch 
import joblib
import os
from utils.pln import (
    obtener_embedding,
    analizar_respuesta,
    agrupar_respuestas_por_tema, 
    synonym_replacement,
    contextual_augmentation
)
from model.clustering import (
    realizar_clustering,
    visualizar_clusters,
    exportar_resultados,
    clustering_jerarquico,
    graficar_dendrograma
)
from utils.inferencia import inferir_rasgos_por_area
from model.mlp import entrenar_mlp
from sklearn.preprocessing import LabelEncoder


# Configuración para usar GPU si está disponible
device = torch.device("cuda") 

# Vacía la caché de la GPU antes de cargar el modelo (opcional pero útil)
if device.type == "cuda":
    torch.cuda.empty_cache()

def aumentar_datos_textuales(df_temas, metodo=None, n_aug=1):
    """
    Aplica aumento de datos a las respuestas según método seleccionado.
    metodo: None, 'synonym_replacement', 'contextual_augmentation'
    Devuelve DataFrame con filas aumentadas agregadas.
    """
    if metodo is None:
        return df_temas  # Sin cambios

    nuevas_filas = []
    for _, row in df_temas.iterrows():
        nueva_fila = row.copy()
        for tema, respuestas in row.items():
            if tema in ["ID", "Edad", "Ocupacion", "Estado"]:
                continue
            respuestas_aumentadas = []
            for r in respuestas:
                if not isinstance(r, str) or not r.strip():
                    respuestas_aumentadas.append(r)
                    continue
                for _ in range(n_aug):
                    if metodo == "synonym_replacement":
                        r_aug = synonym_replacement(r)
                    elif metodo == "contextual_augmentation":
                        r_aug = contextual_augmentation(r)
                    else:
                        r_aug = r
                    respuestas_aumentadas.append(r_aug)
            nueva_fila[tema] = respuestas + respuestas_aumentadas
        nuevas_filas.append(nueva_fila)

    df_aumentado = pd.concat([df_temas, pd.DataFrame(nuevas_filas)], ignore_index=True)
    return df_aumentado


# === CONFIGURACIÓN GENERAL ===
RUTA_CSV = "data/respuestas.csv"
OUTPUT_PATH = "output/resultados/"

NUM_CLUSTERS = 5

AUMENTAR_DATOS = True
METODO_AUMENTO = "contextual_augmentation"  # o "contextual_augmentation" o None o "synonym_replacement"
N_AUG = 5  # cuántas veces aumentamos por respuesta

# === CARGAR DATOS ===
print("[INFO] Cargando datos...")
df = pd.read_csv(RUTA_CSV, encoding="utf-8", sep=",", quotechar='"', on_bad_lines='skip')
columnas_respuestas = df.columns[4:]  # Asume columnas 0-3: Marca temporal, Edad, Ocupación, Estado

# === AGRUPAR RESPUESTAS POR CATEGORÍA TEMÁTICA ===
print("[INFO] Agrupando respuestas por categoría temática...")
df_temas = agrupar_respuestas_por_tema(df, columnas_respuestas)

if AUMENTAR_DATOS and METODO_AUMENTO is not None:
    print(f"[INFO] Aplicando aumento de datos con método {METODO_AUMENTO}...")
    df_temas = aumentar_datos_textuales(df_temas, metodo=METODO_AUMENTO, n_aug=N_AUG)


# === EMBEDDINGS POR TEMA ===
print("[INFO] Calculando embeddings por participante (por tema)...")
embeddings = []
metadatos = []

for _, row in df_temas.iterrows():
    embedding_final = []
    for tema, respuestas in row.items():
        if tema in ["ID", "Edad", "Ocupacion", "Estado"]:
            continue
        vectores = [obtener_embedding(r) for r in respuestas if isinstance(r, str) and r.strip()]
        if vectores:
            promedio = sum(vectores) / len(vectores)
            embedding_final.extend(promedio.tolist())
        else:
            embedding_final.extend([0.0] * 768)
    embeddings.append(embedding_final)
    metadatos.append({
        "ID": row["ID"],
        "Edad": row["Edad"],
        "Ocupacion": row["Ocupacion"],
        "Estado": row["Estado"]
    })
    
embeddings_np = np.array(embeddings)

# === ANÁLISIS LINGÜÍSTICO (TIEMPO VERBAL Y LONGITUD) ===
print("[INFO] Analizando tiempo verbal y longitud por tema...")
linguistico = []

for _, row in df_temas.iterrows():
    resumen = {"ID": row["ID"], "Edad": row["Edad"]}
    for tema, respuestas in row.items():
        if tema in ["ID", "Edad", "Ocupacion", "Estado"]:
            continue

        tiempos = [analizar_respuesta(r)['tiempo_verbal'] for r in respuestas if isinstance(r, str)]
        longitudes = [analizar_respuesta(r)['longitud'] for r in respuestas if isinstance(r, str)]

        resumen[f"{tema}_pasado"] = tiempos.count("pasado")
        resumen[f"{tema}_presente"] = tiempos.count("presente")
        resumen[f"{tema}_futuro"] = tiempos.count("futuro")
        resumen[f"{tema}_longitud_promedio"] = sum(longitudes) / len(longitudes) if longitudes else 0
    linguistico.append(resumen)

    
df_linguistico = pd.DataFrame(linguistico)
df_linguistico.to_csv(os.path.join(OUTPUT_PATH, "resumen_linguistico.csv"), index=False)

# === CLUSTERING ===
print("[INFO] Realizando clustering KMeans...")
etiquetas_kmeans, modelo_kmeans = realizar_clustering(embeddings_np, n_clusters=NUM_CLUSTERS)

print("[INFO] Realizando clustering Jerárquico...")
etiquetas_jer, modelo_jer = clustering_jerarquico(embeddings_np, n_clusters=NUM_CLUSTERS)

# === GRAFICAR DENDROGRAMA PARA ANÁLISIS DE CLÚSTERES ===
print("[INFO] Graficando dendrograma jerárquico para sugerir número de clusters...")
graficar_dendrograma(embeddings_np, output_path=OUTPUT_PATH)

print("[INFO] Visualizando distribución de clusters...")
visualizar_clusters(embeddings_np, etiquetas_kmeans, OUTPUT_PATH)  # usar embeddings_np y etiquetas_kmeans

print("[INFO] Exportando metadatos + clusters...")
exportar_resultados(metadatos, etiquetas_kmeans, OUTPUT_PATH)

# === Entrenamiento MLP ===
print("[INFO] Entrenando modelo MLP para clasificación...")
etiquetas_codificadas = LabelEncoder().fit_transform(etiquetas_kmeans)  # usar etiquetas_kmeans
mlp_model = entrenar_mlp(embeddings_np, etiquetas_codificadas, output_path=OUTPUT_PATH)

print("[INFO] Calculando probabilidades de rasgos psicológicos por área temática...")
df_rasgos = inferir_rasgos_por_area(df_temas)

#print(df_rasgos.head())  # para revisar resultados

# Puedes combinar df_rasgos con df_temas para análisis o exportar a CSV
df_completo = df_temas.merge(df_rasgos, on='ID', how='left')
df_completo.to_csv("output/resultados/resultado_con_rasgos.csv", index=False)

# === GUARDAR MODELO ENTRENADO Y OBJETOS DE APOYO ===
print("[INFO] Guardando modelo entrenado...")

mlp_model = entrenar_mlp(embeddings_np, etiquetas_codificadas, output_path=OUTPUT_PATH)
label_encoder = LabelEncoder()
etiquetas_codificadas = label_encoder.fit_transform(etiquetas_kmeans)
joblib.dump(mlp_model, "output/modelo_personalidad.pkl")
joblib.dump(label_encoder, "output/label_encoder.pkl")

print("[INFO] Modelo y encoder guardados correctamente.")

print("Proceso completado. Revisa los resultados en:", OUTPUT_PATH)