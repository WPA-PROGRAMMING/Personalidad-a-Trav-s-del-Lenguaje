import spacy
from spacy.cli import download
from transformers import AutoTokenizer, AutoModel
import torch

# ──────────────────────────────────────────────
# Cargar modelo de spaCy con fallback automático
# ──────────────────────────────────────────────
try:
    nlp = spacy.load("es_core_news_md")
except OSError:
    print("[INFO] Modelo 'es_core_news_md' no encontrado. Descargando...")
    download("es_core_news_md")
    nlp = spacy.load("es_core_news_md")

# ──────────────────────────────────────────────
# Cargar modelo BETO para embeddings
# ──────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
model = AutoModel.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")


# ──────────────────────────────────────────────
# Función para detectar tiempo verbal predominante
# ──────────────────────────────────────────────
def analizar_respuesta(texto):
    doc = nlp(texto)
    tiempos = {'PRESENTE': 0, 'PASADO': 0, 'FUTURO': 0}
    for token in doc:
        if token.pos_ == 'VERB':
            tiempo = token.morph.get("Tense")
            if 'Past' in tiempo:
                tiempos['PASADO'] += 1
            elif 'Pres' in tiempo:
                tiempos['PRESENTE'] += 1
            elif 'Fut' in tiempo:
                tiempos['FUTURO'] += 1
    return max(tiempos, key=tiempos.get) if sum(tiempos.values()) > 0 else "INDEFINIDO"


# ──────────────────────────────────────────────
# Función para obtener embedding de una oración
# ──────────────────────────────────────────────
def obtener_embedding(texto):
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
