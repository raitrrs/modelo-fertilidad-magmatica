import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

# =====================================================================
# CONFIGURACIÓN Y CARGA DE COMPONENTES
# =====================================================================
st.set_page_config(page_title="Geoquímica Predictiva", layout="wide")
st.title("Clasificador de Fertilidad Magmática multivariado")
st.markdown("Plataforma de procesamiento masivo para bases de datos litogeoquímicas.")

@st.cache_resource
def cargar_modelos():
    # 1. Cargar el escalador matemático correcto
    scaler = joblib.load('scaler_geoquimico.pkl')
    # 2. Cargar el modelo Random Forest (No Keras)
    modelo = joblib.load('modelo_fertilidad_rf.pkl')
    return scaler, modelo

scaler, modelo_rf = cargar_modelos()

# =====================================================================
# INTERFAZ DE CARGA DE DATOS
# =====================================================================
st.subheader("1. Ingesta de Datos Analíticos")
archivo_subido = st.file_uploader("Sube tu matriz geoquímica (formato CSV o Excel)", type=['csv', 'xlsx'])

if archivo_subido is not None:
    # Leer el archivo dependiendo de su extensión
    if archivo_subido.name.endswith('.csv'):
        df_input = pd.read_csv(archivo_subido)
    else:
        df_input = pd.read_excel(archivo_subido)
        
    st.write(f"Archivo cargado exitosamente. Total de muestras detectadas: {len(df_input)}")
    
    # =====================================================================
    # MOTOR DE INFERENCIA
    # =====================================================================
    st.subheader("2. Procesamiento y Predicción")
    if st.button("Ejecutar Modelo Predictivo"):
        with st.spinner('Evaluando firmas multivariadas...'):
            try:
                # Extraer solo las variables numéricas
                datos_numericos = df_input.select_dtypes(include=[np.number])
                
                # Preprocesamiento idéntico al entrenamiento (Log10 + StandardScaler)
                datos_log = np.log10(datos_numericos + 1e-5)
                datos_escalados = scaler.transform(datos_log)
                
                # Predicción del modelo Random Forest
                # predict_proba devuelve dos columnas [prob_esteril, prob_fertil], tomamos la segunda (índice 1)
                probabilidades = modelo_rf.predict_proba(datos_escalados)[:, 1]
                
                # Añadir los resultados al DataFrame original
                df_input['Probabilidad_Fertilidad'] = probabilidades
                df_input['Clasificacion_IA'] = np.where(df_input['Probabilidad_Fertilidad'] > 0.5, 'Fértil', 'Estéril/Artefacto')
                
                st.success("¡Clasificación completada con éxito!")
                
                # Mostrar la tabla interactiva en la web
                st.dataframe(df_input.head(15))
                
                # =====================================================================
                # EXPORTACIÓN DE RESULTADOS
                # =====================================================================
                st.subheader("3. Descarga de Resultados")
                
                # Preparar el CSV en memoria para la descarga
                csv_buffer = df_input.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="Descargar Base de Datos Clasificada (CSV)",
                    data=csv_buffer,
                    file_name="predicciones_fertilidad.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error de procesamiento: Asegúrate de que el archivo contenga los mismos elementos químicos del entrenamiento. Detalles: {e}")
