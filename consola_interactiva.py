# consola_interactiva.py
TEMAS = {
    'familiar': [
        "Mi madre siempre...",
        "Mi padre me decía que...",
        "En mi familia se espera que yo...",
    ],
    'laboral': [
        "Cuando estoy en el trabajo, yo...",
        "Mis superiores me hacen sentir...",
        "Aspiro a que mi ocupación me permita..."
    ],
    'emocional': [
        "Temo que...",
        "Me siento culpable cuando...",
        "Deseo profundamente que..."
    ],
    'social': [
        "Mis amigos suelen...",
        "Me relaciono con las personas...",
        "La gente que me rodea piensa que yo..."
    ]
}

def recolectar_respuestas():
    respuestas = {"ID": "usuario_consola"}
    for area, preguntas in TEMAS.items():
        print(f"\n--- Área: {area.upper()} ---")
        respuestas[area] = []
        for pregunta in preguntas:
            respuesta = input(f"{pregunta}\n> ")
            respuestas[area].append(respuesta)
    return respuestas

def analizar_respuestas_usuario(respuestas_dict):
    filas = []
    for area, textos in respuestas_dict.items():
        if area == "ID":
            continue
        area_features = []
        for texto in textos:
            if isinstance(texto, str) and texto.strip():
                features = calcular_features(texto)
                area_features.append(features)
        if area_features:
            # Promediar probabilidades de los 4 rasgos
            mean_probs = {
                k: round(sum(d[k] for d in area_features) / len(area_features), 3)
                for k in area_features[0]
            }
        else:
            mean_probs = {"ansiedad": 0, "resignacion": 0, "proactividad": 0, "evasion": 0}
        fila = {f"{k}_{area}": v for k, v in mean_probs.items()}
        filas.append(fila)

    perfil = {"ID": respuestas_dict["ID"]}
    for d in filas:
        perfil.update(d)

    return perfil

def imprimir_perfil(perfil):
    print("\n=== Perfil psicológico inferido ===")
    for clave, valor in perfil.items():
        if clave == "ID":
            continue
        print(f"{clave}: {valor}")

if __name__ == "__main__":
    respuestas = recolectar_respuestas()
    perfil = analizar_respuestas_usuario(respuestas)
    imprimir_perfil(perfil)
