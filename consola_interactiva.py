import streamlit as st
import torch
import joblib
import numpy as np
import pandas as pd
import ollama
import sys
import os

st.set_page_config(page_title="Análisis de Perfil Psicológico con IA", layout="wide")

# --- Ajuste de Path para Importaciones ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'model')))

try:
    from pln import obtener_embedding, agrupar_respuestas_por_tema
    from mlp import MLPModel
    from inferencia import inferir_rasgos_por_area
except ImportError as e:
    st.error(f"Error al importar módulos locales: {e}")
    st.info("Asegúrate de que 'pln.py', 'mlp.py', e 'inferencia.py' estén en las carpetas 'utils' y 'model' respectivamente, y que los paths sean correctos.")
    st.stop() # Detiene la ejecución de Streamlit si hay un error crítico de importación.


# --- 1. CONFIGURACIÓN GLOBAL Y CACHE DE MODELOS ---
@st.cache_resource
def load_models():
    torch.cuda.empty_cache()
    """Carga el LabelEncoder y el modelo MLP."""
    st.info("Cargando modelos y objetos necesarios... Esto puede tomar un momento.")
    label_encoder = None
    mlp_model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar el LabelEncoder
    try:
        label_encoder = joblib.load("output/label_encoder.pkl")
        st.success("LabelEncoder cargado correctamente.")
    except FileNotFoundError:
        st.error("ERROR: No se encontró 'output/label_encoder.pkl'. Asegúrate de que el archivo exista en la ruta correcta.")
        st.stop()
    except Exception as e:
        st.error(f"ERROR al cargar LabelEncoder: {e}")
        st.stop()

    # Definir la arquitectura del modelo MLP
    INPUT_DIM = 768 * len(TEMAS_PREGUNTAS)
    OUTPUT_DIM = len(label_encoder.classes_) if label_encoder else 0

    # Instanciar el modelo y cargar los pesos guardados
    mlp_model = MLPModel(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM).to(device)

    try:
        mlp_model.load_state_dict(torch.load("output/resultados/mlp_model.pth", map_location=device))
        mlp_model.eval() # Poner el modelo en modo de evaluación
        st.success(f"Modelo MLP cargado correctamente en {device}.")
    except FileNotFoundError:
        st.error("ERROR: No se encontró el archivo 'output/resultados/mlp_model.pth'.")
        st.stop()
    except Exception as e:
        st.error(f"ERROR al cargar el modelo MLP: {e}")
        st.stop()

    # --- Limpieza de GPU después de la carga inicial ---
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        st.info("Memoria de la GPU limpiada después de la carga de modelos.")

    return label_encoder, mlp_model, device

# Definición de las preguntas, movida fuera de la función principal
TEMAS_PREGUNTAS = {
    'familiar': [
        "Cuéntame lo que más te gustaba de tu madre:",
        "Cuéntame lo que más te gustaba de tu padre:",
        "En tu familia, ¿qué se esperaba de ti cuando eras joven?",
        "¿Cómo describirías tu ambiente familiar actual?"
    ],
    'laboral': [
        "Describe la ambición de tu vida en el ámbito profesional:",
        "En tu trabajo actual, ¿qué es lo que más te molesta?",
        "¿Qué tipo de ambiente laboral te permite dar lo mejor de ti?",
        "Si pudieras elegir, ¿cómo sería tu día de trabajo ideal?"
    ],
    'emocional': [
        "¿Qué es lo que más temes en la vida?",
        "Describe la peor experiencia que has tenido:",
        "¿Cuál es tu mayor deseo o anhelo?",
        "Cuando te sientes estresado/a, ¿cómo sueles manejarlo?"
    ],
    'social': [
        "¿Cómo son tus relaciones con tus amigos más cercanos?",
        "¿Cómo crees que la gente piensa de ti en general?",
        "¿Cómo te relacionas con personas nuevas?",
        "¿Prefieres pasar tiempo solo/a o en grupo?"
    ]
}

# Cargar los modelos al inicio de la aplicación
label_encoder, mlp_model, device = load_models()


# --- 2. FUNCIONES DE PROCESAMIENTO Y PREDICCIÓN ---
# ... (el resto de las funciones 'procesar_y_predecir_streamlit' y 'generar_interpretacion_ollama' sin cambios) ...
def procesar_y_predecir_streamlit(respuestas_usuario_raw: dict):
    """
    Toma las respuestas del usuario, las procesa para generar embeddings,
    usa el modelo MLP para predecir un perfil, infiere rasgos y genera
    una interpretación humanizada con Ollama, mostrando los resultados en Streamlit.
    """
    st.subheader("Paso 2: Procesando tus respuestas...")
    with st.spinner("Analizando tus palabras..."):
        df_usuario = pd.DataFrame([respuestas_usuario_raw])
        columnas_respuestas = list(respuestas_usuario_raw.keys())
        df_temas_usuario = agrupar_respuestas_por_tema(df_usuario, columnas_respuestas)

        embedding_final = []
        for tema in ['familiar', 'laboral', 'emocional', 'social']:
            if tema in df_temas_usuario.columns and df_temas_usuario.iloc[0][tema]:
                respuestas_tema = df_temas_usuario.iloc[0][tema]
                vectores = [obtener_embedding(r) for r in respuestas_tema if isinstance(r, str) and r.strip()]

                if vectores:
                    vectores_cpu = [v.cpu() for v in vectores]
                    promedio = sum(vectores_cpu) / len(vectores_cpu)
                    embedding_final.extend(promedio.tolist())
                else:
                    embedding_final.extend([0.0] * 768)
            else:
                embedding_final.extend([0.0] * 768)

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        st.info("Memoria de la GPU limpiada después de la generación de embeddings.")

    if not embedding_final:
        st.warning("No se pudo generar un embedding válido. Asegúrate de haber ingresado respuestas significativas.")
        return

    st.subheader("Paso 3: Realizando la predicción del perfil...")
    with st.spinner("Conectando con la inteligencia profunda..."):
        embedding_np = np.array([embedding_final])
        embedding_tensor = torch.tensor(embedding_np, dtype=torch.float32).to(device)

        with torch.no_grad():
            output = mlp_model(embedding_tensor)
            prediccion_numerica = torch.argmax(output, dim=1).item()

        etiqueta_predicha = label_encoder.inverse_transform([prediccion_numerica])[0]
        st.markdown(f"### Tu perfil predicho es: **{etiqueta_predicha}**")

    st.subheader("Paso 4: Analizando rasgos psicológicos por área...")
    with st.spinner("Desenterrando tus rasgos..."):
        df_rasgos = inferir_rasgos_por_area(df_temas_usuario)

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        st.info("Memoria de la GPU limpiada después de la predicción de perfil y rasgos.")

    if not df_rasgos.empty:
        st.markdown("---")
        #st.markdown("#### **Análisis de Rasgos (Probabilidades de 0 a 1)**")
        #st.dataframe(df_rasgos.set_index('ID'))

        rasgos_data = df_rasgos.to_string(index=False)

        st.subheader("Paso 5: Generando una interpretación detallada con IA...")
        with st.spinner("El psicólogo de IA está redactando tu informe..."):
            try:
                interpretacion_humanizada = generar_interpretacion_ollama(
                    perfil_predicho=etiqueta_predicha,
                    rasgos_df_str=rasgos_data,
                    respuestas_usuario=respuestas_usuario_raw
                )
                st.markdown("---")
                st.markdown("### **INTERPRETACIÓN PROFUNDA DE TU PERFIL**")
                st.write(interpretacion_humanizada)
            except Exception as e:
                st.error(f"ERROR: No se pudo generar la interpretación con Ollama: {e}")
                st.info("Asegúrate de que el servidor de Ollama esté en ejecución y el modelo 'gemma3:1b' esté descargado y disponible. Puedes verificarlo con `ollama run gemma3:1b` en tu terminal.")
    else:
        st.warning("No se pudo generar un análisis de rasgos detallado. Las respuestas podrían ser insuficientes.")


def generar_interpretacion_ollama(perfil_predicho: str, rasgos_df_str: str, respuestas_usuario: dict) -> str:
    """
    Construye un prompt para Ollama y obtiene una interpretación humanizada.
    """
    respuestas_formateadas = "\n".join([f"- {k.replace('_', ' ').title()}: {v}" for k, v in respuestas_usuario.items()])

    prompt = f"""Eres un psicólogo analítico y empático. Tu tarea es interpretar un perfil psicológico y un análisis de rasgos detallado, y presentarlo de manera comprensiva y humanizada para un usuario, manteniendo un tono profesional pero cercano.
            
            Aquí tienes los resultados del análisis:
            
            **Perfil General Predicho:** {perfil_predicho}
            
            **Análisis Cuantitativo de Rasgos por Área (Probabilidades de 0 a 1):**
            {rasgos_df_str}
            
            **Respuestas Originales del Usuario (para contexto adicional y referencias):**
            {respuestas_formateadas}
            
            Por favor, crea una interpretación detallada y sensible, cubriendo los siguientes puntos esenciales:
            1.  **Introducción Empática:** Comienza reconociendo el esfuerzo y la apertura del usuario al compartir sus respuestas.
            2.  **Análisis del Perfil General:** Explica de forma sencilla y clara qué implica el perfil '{perfil_predicho}'. Ayuda al usuario a entender su clasificación general.
            3.  **Interpretación por Áreas (Familiar, Laboral, Emocional, Social):**
                * Para cada área, analiza los rasgos más predominantes (aquellos con probabilidades más altas) y, si es relevante, los rasgos con probabilidades muy bajas.
                * Explica brevemente qué significa cada rasgo en el contexto específico de esa área (ej., "una alta proactividad en el ámbito laboral podría indicar...").
                * **Crucial:** Utiliza las respuestas originales del usuario (mencionadas en "Respuestas Originales del Usuario") para contextualizar o ejemplificar tus observaciones. Esto hace el análisis mucho más personal y creíble.
            4.  **Patrones y Conexiones:** Identifica y resume cualquier patrón o tendencia interesante que observes en los rasgos o entre las diferentes áreas.
            5.  **Inferir Rasgos del "Big Five":** Basado en toda la información proporcionada (perfil general, análisis de rasgos por área y respuestas originales), haz una inferencia sobre cuál de los "Cinco Grandes" rasgos de personalidad (Apertura a la Experiencia, Conciencia, Extraversión, Amabilidad, Neuroticismo/Estabilidad Emocional) parece ser el más **predominante** o **relevante** en el usuario. Justifica brevemente por qué consideras ese rasgo el más destacado, haciendo referencia a los datos anteriores. Si más de uno es relevante, puedes mencionar hasta dos.
            6.  **Cierre Constructivo y Empoderador:** Finaliza con un mensaje de apoyo, que fomente la auto-reflexión o sugiera áreas de posible crecimiento personal, sin ser prescriptivo ni diagnosticar. El objetivo es empoderar al usuario.
            
            Asegúrate de que la interpretación tenga un **tono cercano, comprensivo y no técnico**, como si un profesional humano estuviera hablando. No uses jerga psicológica compleja. La extensión total debe ser de al menos 300 palabras para asegurar un análisis completo y matizado.
            """

    try:
        response = ollama.chat(model='gemma3:1b', messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content']
    except Exception as e:
        raise Exception(f"Error al conectar o interactuar con Ollama: {e}. "
                        "Verifica que el servidor de Ollama esté en ejecución y el modelo especificado esté descargado.")


# --- INTERFAZ DE USUARIO DE STREAMLIT (parte principal de la aplicación) ---
# Inicializar st.session_state para almacenar las respuestas
if 'respuestas' not in st.session_state:
    st.session_state.respuestas = {}
if 'analisis_realizado' not in st.session_state:
    st.session_state.analisis_realizado = False

st.title("Análisis de Perfil Psicológico con IA")
st.markdown("Bienvenido al sistema que te ayudará a comprender mejor tu perfil psicológico y tus rasgos. Por favor, responde las preguntas en los campos de texto.")

st.markdown("---")
st.subheader("Paso 1: ¡Cuéntanos sobre ti!")
st.info("Por favor, sé lo más descriptivo posible en tus respuestas para obtener un análisis más preciso.")

# Crear pestañas para cada área temática
tab_familiar, tab_laboral, tab_emocional, tab_social = st.tabs(list(TEMAS_PREGUNTAS.keys()))

with tab_familiar:
    st.header("Área Familiar")
    for i, pregunta in enumerate(TEMAS_PREGUNTAS['familiar']):
        key = f"familiar_{i+1}"
        st.session_state.respuestas[key] = st.text_area(f"{i+1}. {pregunta}", value=st.session_state.respuestas.get(key, ""), key=key, height=100)

with tab_laboral:
    st.header("Área Laboral")
    for i, pregunta in enumerate(TEMAS_PREGUNTAS['laboral']):
        key = f"laboral_{i+1}"
        st.session_state.respuestas[key] = st.text_area(f"{i+1}. {pregunta}", value=st.session_state.respuestas.get(key, ""), key=key, height=100)

with tab_emocional:
    st.header("Área Emocional")
    for i, pregunta in enumerate(TEMAS_PREGUNTAS['emocional']):
        key = f"emocional_{i+1}"
        st.session_state.respuestas[key] = st.text_area(f"{i+1}. {pregunta}", value=st.session_state.respuestas.get(key, ""), key=key, height=100)

with tab_social:
    st.header("Área Social")
    for i, pregunta in enumerate(TEMAS_PREGUNTAS['social']):
        key = f"social_{i+1}"
        st.session_state.respuestas[key] = st.text_area(f"{i+1}. {pregunta}", value=st.session_state.respuestas.get(key, ""), key=key, height=100)

st.markdown("---")

# Botón para iniciar el análisis
if st.button("Obtener mi Análisis Psicológico", type="primary"):
    respuestas_validas = {k: v for k, v in st.session_state.respuestas.items() if v and v.strip() != ""}

    if not respuestas_validas:
        st.warning("Por favor, ingresa al menos una respuesta en cualquiera de las áreas para iniciar el análisis.")
    else:
        st.session_state.analisis_realizado = True
        procesar_y_predecir_streamlit(respuestas_validas)
elif st.session_state.analisis_realizado:
    respuestas_validas = {k: v for k, v in st.session_state.respuestas.items() if v and v.strip() != ""}
    if respuestas_validas:
        procesar_y_predecir_streamlit(respuestas_validas)