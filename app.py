import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================================================================
# CONFIGURACIÓN DE LA PÁGINA
# =====================================================================
st.set_page_config(page_title="Prospectividad Alausí", layout="wide", page_icon="🌋")

# Título principal y descripción
st.title("Clasificador de Fertilidad Magmática")
st.markdown("Plataforma de inferencia multivariada para caracterización litogeoquímica.")
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

# Lista estricta de elementos (Garantiza que no entren coordenadas al modelo)
elementos_requeridos = ['AU', 'AG', 'CU', 'PB', 'ZN', 'MO', 'NI', 'CO', 'CD', 'BI',
                        'FE', 'MN', 'TE', 'BA', 'CR', 'V', 'SN', 'W', 'LA', 'AL',
                        'MG', 'CA', 'NA', 'K', 'SR', 'Y', 'GA', 'LI', 'NB', 'SC',
                        'TA', 'TI', 'ZR', 'AS', 'SB', 'HG', 'PT', 'PD']

# =====================================================================
# PANEL LATERAL (SIDEBAR) - INGESTA DE DATOS
# =====================================================================
st.sidebar.header("⚙️ Panel de Control")
st.sidebar.write("Sube la matriz analítica del laboratorio para su evaluación.")
archivo_subido = st.sidebar.file_uploader("Formato CSV o Excel", type=['csv', 'xlsx'])

# =====================================================================
# ÁREA PRINCIPAL - PROCESAMIENTO Y RESULTADOS
# =====================================================================
if archivo_subido is not None:
    # Leer el archivo
    if archivo_subido.name.endswith('.csv'):
        df_input = pd.read_csv(archivo_subido)
    else:
        df_input = pd.read_excel(archivo_subido)
        
    st.sidebar.success(f"Archivo cargado: {len(df_input)} muestras.")
    
    # Botón de ejecución en el sidebar
    if st.sidebar.button("🚀 Ejecutar Modelo Predictivo"):
        with st.spinner('Evaluando firmas multivariadas...'):
            try:
                # 1. FILTRADO ESTRICTO: Seleccionar solo las 38 columnas químicas
                datos_modelo = df_input[elementos_requeridos].copy()
                
                # 2. Preprocesamiento (Log10 + StandardScaler)
                datos_log = np.log10(datos_modelo + 1e-5)
                datos_escalados = scaler.transform(datos_log)
                
                # 3. Inferencia Random Forest
                probabilidades = modelo_rf.predict_proba(datos_escalados)[:, 1]
                
                # 4. Asignación de resultados
                df_input['Prob_Fertilidad'] = probabilidades
                df_input['Clasificacion_IA'] = np.where(df_input['Prob_Fertilidad'] > 0.5, 'Fértil', 'Estéril/Artefacto')
                
                # =====================================================================
                # KPIS Y MÉTRICAS DE IMPACTO VISUAL
                # =====================================================================
                st.subheader("📊 Resumen Analítico")
                
                total_muestras = len(df_input)
                muestras_fertiles = len(df_input[df_input['Clasificacion_IA'] == 'Fértil'])
                porcentaje_anomalias = (muestras_fertiles / total_muestras) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label="Total Muestras Analizadas", value=total_muestras)
                with col2:
                    st.metric(label="Blancos Fértiles Detectados", value=muestras_fertiles)
                with col3:
                    st.metric(label="Tasa de Anomalía", value=f"{porcentaje_anomalias:.2f}%")
                
                st.markdown("---")
                
                # =====================================================================
                # TABLA DE RESULTADOS Y DESCARGA
                # =====================================================================
                st.subheader("📄 Base de Datos Clasificada")
                
                # Mostrar el DataFrame destacando las últimas columnas añadidas
                st.dataframe(df_input, use_container_width=True)
                
                # Preparar descarga
                csv_buffer = df_input.to_csv(index=False).encode('utf-8')
                st.sidebar.markdown("---")
                st.sidebar.download_button(
                    label="📥 Descargar Resultados (CSV)",
                    data=csv_buffer,
                    file_name="Alausí_Resultados_Predictivos.csv",
                    mime="text/csv"
                )
                
            except KeyError as e:
                st.error(f"⚠️ Error de formato: El archivo subido no contiene la columna {e}. Verifica que estén los 38 elementos químicos requeridos.")
            except Exception as e:
                st.error(f"⚠️ Error inesperado: {e}")
else:
    st.info("👈 Por favor, carga una matriz de datos en el panel lateral para iniciar el análisis.")
