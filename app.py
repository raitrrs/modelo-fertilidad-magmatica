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
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-1.5-flash')

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

# Función Gemini
def generar_interpretacion_llm(df_resumen):
    contexto = df_resumen.to_string()
    prompt = f"Eres un geólogo experto en pórfidos. Analiza este resumen estadístico: {contexto}. Genera un reporte técnico de 100 palabras."
    return model.generate_content(prompt).text

# =====================================================================
# INTERFAZ Y SIDEBAR
# =====================================================================
st.title("Clasificador de Fertilidad Magmática")
st.sidebar.header("⚙️ Panel de Control")

with st.sidebar.expander("ℹ️ Guía de Formato"):
    st.markdown("Sube un CSV/Excel con las 38 columnas: " + ", ".join(elementos_requeridos))

archivo_subido = st.sidebar.file_uploader("Cargar archivo", type=['csv', 'xlsx'])
ejecutar_modelo = st.sidebar.button("🚀 Ejecutar Modelo Predictivo") if archivo_subido else False

# Contador
st.sidebar.markdown("---")
st.sidebar.markdown(f'<div style="text-align: center;"><img src="https://api.visitorbadge.io/api/visitors?path=modelo_fertilidad_alausi_uce&label=Visitas&countColor=%23d9534f" alt="Contador"></div>', unsafe_allow_html=True)

# =====================================================================
# LÓGICA PRINCIPAL
# =====================================================================
if ejecutar_modelo:
    with st.spinner('Procesando...'):
        try:
            df_input = pd.read_csv(archivo_subido) if archivo_subido.name.endswith('.csv') else pd.read_excel(archivo_subido)
            datos_modelo = df_input[elementos_requeridos].copy()
            datos_escalados = scaler.transform(np.log10(datos_modelo + 1e-5))
            
            df_input['Prob_Fertilidad'] = modelo_rf.predict_proba(datos_escalados)[:, 1]
            df_input['Clasificacion_IA'] = np.where(df_input['Prob_Fertilidad'] > 0.5, 'Fértil', 'Estéril/Artefacto')
            
            tab1, tab2, tab3 = st.tabs(["📄 Datos", "📉 Diagramas Geoquímicos", "🗺️ Mapa"])
            
            with tab1:
                st.metric("Total Muestras", len(df_input))
                st.dataframe(df_input.head(1000).style.applymap(lambda x: 'background-color: #ffcccc' if x == 'Fértil' else '', subset=['Clasificacion_IA']))
            
            with tab2:
                if st.button("Generar Reporte Geológico con Gemini"):
                    st.write(generar_interpretacion_llm(df_input.groupby('Clasificacion_IA')[elementos_requeridos].mean()))
                
                fig = plt.figure(figsize=(12, 8))
                sns.scatterplot(data=df_input, x='Y', y=df_input['SR']/df_input['Y'], hue='Prob_Fertilidad', palette='coolwarm')
                plt.xscale('log'); plt.yscale('log'); plt.title('Fertilidad: Sr/Y vs Y')
                st.pyplot(fig)
            
            with tab3:
                if 'LONGITUD' in df_input.columns:
                    grid_x, grid_y = np.mgrid[df_input['LONGITUD'].min():df_input['LONGITUD'].max():200j, df_input['LATITUD'].min():df_input['LATITUD'].max():200j]
                    plt.imshow(griddata((df_input['LONGITUD'], df_input['LATITUD']), df_input['Prob_Fertilidad'], (grid_x, grid_y), method='linear').T, origin='lower', cmap='coolwarm')
                    st.pyplot(plt.gcf())
            
            st.sidebar.download_button("📥 Descargar Resultados", data=df_input.to_csv(index=False).encode('utf-8'), file_name="Resultados.csv")
            
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
