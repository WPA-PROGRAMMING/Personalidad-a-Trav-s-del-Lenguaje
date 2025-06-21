# utils/pln.py
import nltk
import re
import spacy
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from transformers import pipeline
import random
from nltk.corpus import wordnet

nltk.download('wordnet')
nltk.download('omw-1.4')

# Cargar SpaCy para análisis gramatical (tiempo verbal)
try:
    nlp = spacy.load("es_core_news_md")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "es_core_news_md"])
    nlp = spacy.load("es_core_news_md")

# Configuración para usar GPU si está disponible
device = torch.device("cpu") 

'''
device = torch.device("cpu") 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
'''

# Cargar modelo BERT español (U. de Chile)
tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
model = AutoModel.from_pretrained("dccuchile/bert-base-spanish-wwm-cased").to(device) # device GPU "cpu"
model.eval()

# Carga pipeline para análisis de sentimiento (puedes elegir otro modelo)
# Cambiar device = 0 para GPU o -1 para CPU
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device=-1 if torch.cuda.is_available() else -1)

# Inicializar fill-mask para aumento contextual
fill_mask = pipeline(
    "fill-mask",
    model="dccuchile/bert-base-spanish-wwm-cased",
    tokenizer="dccuchile/bert-base-spanish-wwm-cased",
    device=0 if torch.cuda.is_available() else -1
)

def analizar_sentimiento(texto):
    texto_limpio = limpiar_texto(texto)
    if texto_limpio == "":
        return None
    resultado = sentiment_analyzer(texto_limpio[:512])  # limita a 512 tokens
    return resultado[0]  # devuelve dict con label y score


def limpiar_texto(texto):
    """
    Limpia texto de signos de puntuación, números y espacios extras.
    """
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r"[^\w\sáéíóúüñ]", "", texto)  # permite letras y acentos
    texto = re.sub(r"\d+", "", texto)  # elimina números
    texto = re.sub(r"\s+", " ", texto).strip()  # elimina espacios extras
    return texto

# Luego modifica obtener_embedding para limpiar texto antes
def obtener_embedding(texto):
    texto_limpio = limpiar_texto(texto)
    if texto_limpio == "":
        return torch.zeros(model.config.hidden_size).to(device)
    inputs = tokenizer(texto_limpio, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state.mean(dim=1).squeeze()


def analizar_respuesta(texto):
    """
    Retorna:
    - tiempo_verbal: 'pasado', 'presente' o 'futuro'
    - longitud: número de palabras
    """

    if not isinstance(texto, str) or texto.strip() == "":
        return {'tiempo_verbal': 'desconocido', 'longitud': 0}

    # Tokeniza el texto con spaCy
    doc = nlp(texto)

    # Diccionario extendido de formas verbales
    verbos_tiempo = {
        'pasado': [
            "fui", "fuiste", "fue", "fuimos", "fueron",
            "estuve", "estuviste", "estuvo", "estuvimos", "estuvieron",
            "tuve", "tuviste", "tuvo", "tuvimos", "tuvieron",
            "hice", "hiciste", "hizo", "hicimos", "hicieron",
            "quise", "quería", "amaba", "amé", "pensé", "soñé", "veía", "veímos"
        ],
        'presente': [
            "soy", "eres", "es", "somos", "son",
            "estoy", "estás", "está", "estamos", "están",
            "tengo", "tienes", "tiene", "tenemos", "tienen",
            "hago", "haces", "hace", "hacemos", "hacen",
            "quiero", "amo", "digo", "pienso", "sueño", "veo"
        ],
        'futuro': [
            "seré", "serás", "será", "seremos", "serán",
            "estaré", "estarás", "estará", "estaremos", "estarán",
            "tendré", "tendrás", "tendrá", "tendremos", "tendrán",
            "haré", "harás", "hará", "haremos", "harán",
            "iré", "iremos", "vas", "voy", "van", "planeo", "planeamos", "intentaré"
        ]
    }

    tiempos = {'pasado': 0, 'presente': 0, 'futuro': 0}

    # Extraer solo las palabras del texto en minúscula
    palabras = [token.text.lower() for token in doc]

    # Reglas superficiales basadas en diccionario
    for palabra in palabras:
        for tiempo, formas in verbos_tiempo.items():
            if palabra in formas:
                tiempos[tiempo] += 1

    # Reglas de spaCy usando análisis morfológico
    for token in doc:
        if token.pos_ == 'VERB':
            morph = token.morph
            if 'Tense=Past' in morph:
                tiempos['pasado'] += 1
            elif 'Tense=Pres' in morph:
                tiempos['presente'] += 1
            elif 'Tense=Fut' in morph:
                tiempos['futuro'] += 1

    # Reglas para perífrasis como "voy a", "vamos a"
    texto_lower = texto.lower()
    if "voy a" in texto_lower or "vamos a" in texto_lower or "van a" in texto_lower:
        tiempos['futuro'] += 1
    if "he " in texto_lower or "ha " in texto_lower or "hemos " in texto_lower:
        tiempos['pasado'] += 1

    # Determinar el tiempo verbal dominante
    tiempo_verbal = max(tiempos, key=tiempos.get)
    if tiempos[tiempo_verbal] == 0:
        tiempo_verbal = 'desconocido'

    # Longitud del texto (palabras alfabéticas)
    longitud = len([t for t in doc if t.is_alpha])

    return {'tiempo_verbal': tiempo_verbal, 'longitud': longitud}

def agrupar_respuestas_por_tema(df, columnas_respuestas):
    """
    Agrupa respuestas en categorías temáticas: familiar, laboral, emocional, social.

    Retorna un nuevo DataFrame con columnas:
    - ID, Edad, Ocupacion, Estado
    - familiar: lista de respuestas
    - laboral: lista de respuestas
    - emocional: lista de respuestas
    - social: lista de respuestas
    """
    temas = {
        'familiar': [],
        'laboral': [],
        'emocional': [],
        'social': []
    }

    for col in columnas_respuestas:
        col_lower = col.lower()
        if any(p in col_lower for p in ["madre", "padre", "familia"]):
            temas['familiar'].append(col)
        elif any(p in col_lower for p in ["trabajo", "jefe", "órdenes", "maestros", "ambición", "ocupación"]):
            temas['laboral'].append(col)
        elif any(p in col_lower for p in ["temo", "miedo", "culpable", "peor", "equivocación", "feliz", "triste", "deseo"]):
            temas['emocional'].append(col)
        elif any(p in col_lower for p in ["amigos", "gente", "relaciones", "mujeres", "social", "superiores"]):
            temas['social'].append(col)

    data = []
    for _, row in df.iterrows():
        agrupado = {
            "ID": row.get("ID") or row.get("Marca temporal"),
            "Edad": row.get("Edad"),
            "Ocupacion": row.get("Ocupación"),
            "Estado": row.get("Estado de residencia") or row.get("Estado"),
            "familiar": [row[col] for col in temas['familiar'] if col in row],
            "laboral": [row[col] for col in temas['laboral'] if col in row],
            "emocional": [row[col] for col in temas['emocional'] if col in row],
            "social": [row[col] for col in temas['social'] if col in row]
        }
        data.append(agrupado)

    return pd.DataFrame(data)

# === Funciones de aumento de datos ===

def synonym_replacement(text, n=1):
    doc = nlp(text)
    words = [token.text for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]]
    if not words:
        return text

    new_text = text.split()
    random_words = random.sample(words, min(n, len(words)))

    for word in random_words:
        syns = wordnet.synsets(word, lang='spa')
        lemmas = set()
        for syn in syns:
            for l in syn.lemmas('spa'):
                if l.name() != word:
                    lemmas.add(l.name().replace('_', ' '))
        if lemmas:
            synonym = random.choice(list(lemmas))
            for i, w in enumerate(new_text):
                if w == word:
                    new_text[i] = synonym
                    break

    return " ".join(new_text)


def contextual_augmentation(text, n=1):
    doc = nlp(text)
    candidates = [token for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]]
    if not candidates:
        return text

    token_to_mask = random.choice(candidates)
    token_idx = token_to_mask.i

    words = text.split()
    if token_idx >= len(words):
        return text

    words[token_idx] = fill_mask.tokenizer.mask_token
    masked_text = " ".join(words)

    results = fill_mask(masked_text)
    for pred in results:
        if pred['token_str'].strip() != token_to_mask.text:
            words[token_idx] = pred['token_str'].strip()
            return " ".join(words)
    return text


def aumentar_datos_textuales(df_temas, metodo=None, n_aug=1):
    if metodo is None:
        return df_temas

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