import spacy
from transformers import AutoTokenizer, AutoModel
import torch

# Cargar modelos
nlp = spacy.load("es_core_news_md")
tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
model = AutoModel.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

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

def obtener_embedding(texto):
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
