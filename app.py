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

# Inicializar estado para que el reporte no desaparezca
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
ejecutar_modelo = st.sidebar.button("🚀 Ejecutar Modelo Predictivo") if archivo_subido else False

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
            df_input['Sr_Y'] = df_input['SR'] / df_input['Y']
            
            tab1, tab2, tab3 = st.tabs(["📄 Datos", "📉 Diagramas Geoquímicos", "🗺️ Mapa"])
            
            with tab1:
                st.subheader("📊 Resumen Analítico")
                col1, col2, col3 = st.columns(3)
                total = len(df_input)
                fertiles = len(df_input[df_input['Clasificacion_IA'] == 'Fértil'])
                col1.metric("Total Muestras", total)
                col2.metric("Blancos Fértiles", fertiles)
                col3.metric("Tasa de Anomalía", f"{(fertiles/total)*100:.2f}%")
                st.dataframe(df_input.head(1000).style.applymap(lambda x: 'background-color: #ffcccc' if x == 'Fértil' else '', subset=['Clasificacion_IA']))
            
            with tab2:
                st.subheader("Análisis Geoquímico Completo")
                # Botón Gemini
                if st.button("Generar Reporte con Gemini"):
                    resumen = df_input.groupby('Clasificacion_IA')[elementos_requeridos].mean().to_string()
                    prompt = f"Eres geólogo experto en pórfidos. Analiza: {resumen}. Reporte técnico de 100 palabras."
                    st.session_state.reporte_gemini = model.generate_content(prompt).text
                
                if st.session_state.reporte_gemini:
                    st.info(st.session_state.reporte_gemini)
                
                # Gráficos (6 paneles)
                fig, axs = plt.subplots(2, 3, figsize=(16, 10))
                axs = axs.flatten()
                
                sns.scatterplot(data=df_input, x='Y', y='Sr_Y', hue='Prob_Fertilidad', ax=axs[0]); axs[0].set_xscale('log'); axs[0].set_yscale('log')
                sns.scatterplot(data=df_input, x='CR', y='FE', hue='Prob_Fertilidad', ax=axs[1]); axs[1].set_xscale('log'); axs[1].set_yscale('log')
                sns.scatterplot(data=df_input, x='K', y='CU', hue='Prob_Fertilidad', ax=axs[2]); axs[2].set_xscale('log'); axs[2].set_yscale('log')
                sns.scatterplot(data=df_input, x='TI', y='V', hue='Prob_Fertilidad', ax=axs[3]); axs[3].set_xscale('log'); axs[3].set_yscale('log')
                sns.histplot(data=df_input, x='Prob_Fertilidad', ax=axs[4])
                sns.boxplot(data=df_input, x='Clasificacion_IA', y='AU', ax=axs[5])
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with tab3:
                if 'LONGITUD' in df_input.columns:
                    grid_x, grid_y = np.mgrid[df_input['LONGITUD'].min():df_input['LONGITUD'].max():200j, df_input['LATITUD'].min():df_input['LATITUD'].max():200j]
                    plt.imshow(griddata((df_input['LONGITUD'], df_input['LATITUD']), df_input['Prob_Fertilidad'], (grid_x, grid_y), method='linear').T, origin='lower', cmap='coolwarm')
                    st.pyplot(plt.gcf())
            
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
