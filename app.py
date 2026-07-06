import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
import google.generativeai as genai

# Configuración inicial
pd.set_option("styler.render.max_elements", 2000000)
st.set_page_config(page_title="Prospectividad Alausí", layout="wide", page_icon="🌋")
sns.set_theme(style="whitegrid")

# Configuración API Gemini
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    st.error("API Key no configurada en los secretos de Streamlit.")

# --- INICIALIZACIÓN DE SESIÓN (PERSISTENCIA) ---
if 'df_input' not in st.session_state:
    st.session_state.df_input = None
if 'reporte_gemini' not in st.session_state:
    st.session_state.reporte_gemini = ""

@st.cache_resource
def cargar_modelos():
    scaler = joblib.load('scaler_geoquimico.pkl')
    modelo = joblib.load('modelo_fertilidad_rf.pkl')
    return scaler, modelo

scaler, modelo_rf = cargar_modelos()

elementos_requeridos = ['AU', 'AG', 'CU', 'PB', 'ZN', 'MO', 'NI', 'CO', 'CD', 'BI',
                        'FE', 'MN', 'TE', 'BA', 'CR', 'V', 'SN', 'W', 'LA', 'AL',
                        'MG', 'CA', 'NA', 'K', 'SR', 'Y', 'GA', 'LI', 'NB', 'SC',
                        'TA', 'TI', 'ZR', 'AS', 'SB', 'HG', 'PT', 'PD']

# =====================================================================
# INTERFAZ Y SIDEBAR
# =====================================================================
st.title("Clasificador de Fertilidad Magmática")
st.sidebar.header("⚙️ Panel de Control")

archivo_subido = st.sidebar.file_uploader("Cargar archivo", type=['csv', 'xlsx'])

# Botón principal de ejecución
if st.sidebar.button("🚀 Ejecutar Modelo Predictivo"):
    if archivo_subido:
        with st.spinner('Procesando datos...'):
            try:
                df = pd.read_csv(archivo_subido) if archivo_subido.name.endswith('.csv') else pd.read_excel(archivo_subido)
                datos_modelo = df[elementos_requeridos].copy()
                datos_escalados = scaler.transform(np.log10(datos_modelo + 1e-5))
                
                df['Prob_Fertilidad'] = modelo_rf.predict_proba(datos_escalados)[:, 1]
                df['Clasificacion_IA'] = np.where(df['Prob_Fertilidad'] > 0.5, 'Fértil', 'Estéril/Artefacto')
                df['Sr_Y'] = df['SR'] / df['Y']
                
                # Guardar en sesión
                st.session_state.df_input = df
                st.session_state.reporte_gemini = "" # Resetear reporte si se carga archivo nuevo
            except Exception as e:
                st.error(f"Error: {e}")

# =====================================================================
# LÓGICA PRINCIPAL (MOSTRAR SI HAY DATOS)
# =====================================================================
if st.session_state.df_input is not None:
    df = st.session_state.df_input
    
    tab1, tab2, tab3 = st.tabs(["📄 Datos", "📉 Diagramas Geoquímicos", "🗺️ Mapa"])
    
    with tab1:
        st.subheader("Resumen Analítico")
        total = len(df)
        fertiles = len(df[df['Clasificacion_IA'] == 'Fértil'])
        st.metric("Total Muestras", total)
        st.dataframe(df.head(1000).style.applymap(lambda x: 'background-color: #ffcccc' if x == 'Fértil' else '', subset=['Clasificacion_IA']))
    
    with tab2:
        st.subheader("Análisis Geoquímico")
        
        if st.button("Generar Reporte con Gemini"):
            resumen = df.groupby('Clasificacion_IA')[elementos_requeridos].mean().to_string()
            prompt = f"Analiza estos promedios geoquímicos: {resumen}. Eres un geólogo experto en pórfidos. Dame un reporte de 100 palabras."
            st.session_state.reporte_gemini = model.generate_content(prompt).text
        
        if st.session_state.reporte_gemini:
            st.info(st.session_state.reporte_gemini)
        
        # Gráficos
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        sns.scatterplot(data=df, x='Y', y='Sr_Y', hue='Prob_Fertilidad', ax=axs[0,0], palette='coolwarm')
        sns.scatterplot(data=df, x='CR', y='FE', hue='Prob_Fertilidad', ax=axs[0,1], palette='coolwarm')
        sns.scatterplot(data=df, x='K', y='CU', hue='Prob_Fertilidad', ax=axs[1,0], palette='coolwarm')
        sns.histplot(data=df, x='Prob_Fertilidad', ax=axs[1,1])
        plt.tight_layout()
        st.pyplot(fig)
