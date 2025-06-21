from utils.pln import analizar_sentimiento, analizar_respuesta
import pandas as pd

def calcular_features(texto):
    sentimiento = analizar_sentimiento(texto)
    tiempo = analizar_respuesta(texto)

    # Si no se pudo analizar el sentimiento, asignamos valores neutros
    if sentimiento is None:
        return {'ansiedad': 0.5, 'resignacion': 0.5, 'proactividad': 0.5, 'evasion': 0.5}

    label = sentimiento['label'].lower()
    score = sentimiento['score']
    tiempo_verbal = tiempo['tiempo_verbal']

    # Convertimos a probabilidad simple (puedes mejorar esto con un modelo real)
    ansiedad = 0.0
    resignacion = 0.0
    proactividad = 0.0
    evasion = 0.0

    if "neg" in label:
        ansiedad = score
    elif "pos" in label:
        proactividad = score
    else:
        resignacion = 1 - score

    if tiempo_verbal == "futuro":
        proactividad += 0.2
    elif tiempo_verbal == "pasado":
        resignacion += 0.2
    elif tiempo_verbal == "desconocido":
        evasion += 0.2

    return {
        'ansiedad': min(ansiedad, 1.0),
        'resignacion': min(resignacion, 1.0),
        'proactividad': min(proactividad, 1.0),
        'evasion': min(evasion, 1.0)
    }

def inferir_rasgos_por_area(df_temas):
    registros = []

    for _, row in df_temas.iterrows():
        id_ = row["ID"]
        resultado = {"ID": id_}

        for area in ['familiar', 'laboral', 'emocional', 'social']:
            textos = row.get(area, [])
            textos_validos = [t for t in textos if isinstance(t, str) and t.strip()]

            #print(f"[DEBUG] Procesando ID {id_} - Área {area} - Respuestas válidas: {len(textos_validos)}")

            if not textos_validos:
                # Valores neutros si no hay datos
                probs_promedio = {'ansiedad': 0.5, 'resignacion': 0.5, 'proactividad': 0.5, 'evasion': 0.5}
            else:
                features_list = [calcular_features(t) for t in textos_validos]
                if not features_list:
                    probs_promedio = {'ansiedad': 0.5, 'resignacion': 0.5, 'proactividad': 0.5, 'evasion': 0.5}
                else:
                    # Promediar resultados
                    keys = features_list[0].keys()
                    probs_promedio = {
                        k: sum(f[k] for f in features_list) / len(features_list) for k in keys
                    }

            # Agregar al resultado
            for k, v in probs_promedio.items():
                resultado[f"{k}_{area}"] = round(v, 3)

        registros.append(resultado)

    return pd.DataFrame(registros)
