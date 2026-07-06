import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata

# Quitar el límite de celdas coloreadas en Pandas
pd.set_option("styler.render.max_elements", 2000000)

# =====================================================================
# CONFIGURACIÓN DE LA PÁGINA
# =====================================================================
st.set_page_config(page_title="Prospectividad Alausí", layout="wide", page_icon="🌋")
sns.set_theme(style="whitegrid")

st.title("Clasificador de Fertilidad Magmática")
st.markdown("Plataforma de inferencia multivariada y análisis geoespacial.")
st.markdown("---")

# =====================================================================
# CARGA DE MODELOS (CACHÉ)
# =====================================================================
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
# PANEL LATERAL (SIDEBAR)
# =====================================================================
st.sidebar.header("⚙️ Panel de Control")

with st.sidebar.expander("ℹ️ Guía de Formato y Encabezados"):
    st.markdown("""
    **Requisitos de la matriz:**
    * **Archivos:** `.csv` o `.xlsx`.
    * **Columnas obligatorias (mayúsculas):**
    `AU, AG, CU, PB, ZN, MO, NI, CO, CD, BI, FE, MN, TE, BA, CR, V, SN, W, LA, AL, MG, CA, NA, K, SR, Y, GA, LI, NB, SC, TA, TI, ZR, AS, SB, HG, PT, PD`
    """)

st.sidebar.write("Sube la matriz analítica del laboratorio.")
archivo_subido = st.sidebar.file_uploader("Formato CSV o Excel", type=['csv', 'xlsx'])

# Lógica del botón
ejecutar_modelo = False
if archivo_subido is not None:
    ejecutar_modelo = st.sidebar.button("🚀 Ejecutar Modelo Predictivo")

# Contador de visitas
id_unico = "modelo_fertilidad_alausi_uce"
contador_html = f"""
<div style="text-align: center;">
    <img src="https://api.visitorbadge.io/api/visitors?path={id_unico}&label=Visitas&countColor=%23d9534f&style=flat&labelStyle=upper" alt="Contador de visitas">
</div>
"""
st.sidebar.markdown("---")
st.sidebar.markdown(contador_html, unsafe_allow_html=True)

# =====================================================================
# LÓGICA PRINCIPAL DE PROCESAMIENTO
# =====================================================================
if ejecutar_modelo:
    with st.spinner('Evaluando firmas multivariadas...'):
        try:
            if archivo_subido.name.endswith('.csv'):
                df_input = pd.read_csv(archivo_subido)
            else:
                df_input = pd.read_excel(archivo_subido)
            
            # 1. Filtrado de elementos
            datos_modelo = df_input[elementos_requeridos].copy()
            datos_log = np.log10(datos_modelo + 1e-5)
            datos_escalados = scaler.transform(datos_log)
            
            # 2. Predicción
            probabilidades = modelo_rf.predict_proba(datos_escalados)[:, 1]
            df_input['Prob_Fertilidad'] = probabilidades
            df_input['Clasificacion_IA'] = np.where(df_input['Prob_Fertilidad'] > 0.5, 'Fértil', 'Estéril/Artefacto')
            
            if 'SR' in df_input.columns and 'Y' in df_input.columns:
                df_input['Sr_Y'] = df_input['SR'] / df_input['Y']
            
            # 3. Visualización en pestañas
            tab1, tab2, tab3 = st.tabs(["📄 Resumen y Datos", "📉 Diagramas Geoquímicos", "🗺️ Mapa de Calor Espacial"])
            
            with tab1:
                st.subheader("📊 Resumen Analítico")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Muestras", len(df_input))
                col2.metric("Blancos Fértiles", len(df_input[df_input['Clasificacion_IA'] == 'Fértil']))
                col3.metric("Tasa de Anomalía", f"{(len(df_input[df_input['Clasificacion_IA'] == 'Fértil'])/len(df_input))*100:.2f}%")
                
                def colorear_fertiles(val):
                    return 'background-color: #ffcccc' if val == 'Fértil' else ''
                
                st.dataframe(df_input.head(1000).style.applymap(colorear_fertiles, subset=['Clasificacion_IA']), use_container_width=True)
            
            with tab2:
                fig = plt.figure(figsize=(12, 8))
                # Ejemplo de un gráfico rápido de los diagramas (Sr/Y vs Y)
                sns.scatterplot(data=df_input, x='Y', y='Sr_Y', hue='Prob_Fertilidad', palette='coolwarm')
                plt.xscale('log'); plt.yscale('log')
                st.pyplot(fig)
            
            with tab3:
                if 'LONGITUD' in df_input.columns and 'LATITUD' in df_input.columns:
                    X_coord, Y_coord, Z_prob = df_input['LONGITUD'].values, df_input['LATITUD'].values, df_input['Prob_Fertilidad'].values
                    grid_x, grid_y = np.mgrid[X_coord.min():X_coord.max():200j, Y_coord.min():Y_coord.max():200j]
                    grid_z = griddata((X_coord, Y_coord), Z_prob, (grid_x, grid_y), method='linear')
                    
                    fig_map = plt.figure(figsize=(8, 6))
                    plt.imshow(grid_z.T, extent=(X_coord.min(), X_coord.max(), Y_coord.min(), Y_coord.max()), origin='lower', cmap='coolwarm')
                    st.pyplot(fig_map)
                else:
                    st.warning("Faltan coordenadas LATITUD/LONGITUD para el mapa.")
            
            # Descarga final
            st.sidebar.markdown("---")
            st.sidebar.download_button("📥 Descargar Resultados", data=df_input.to_csv(index=False).encode('utf-8'), file_name="Resultados.csv", mime="text/csv")
            
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
else:
    st.info("👈 Sube un archivo en el panel lateral para comenzar.")
