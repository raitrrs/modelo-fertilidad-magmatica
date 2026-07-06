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
                st.subheader("Análisis Multivariado de Fertilidad Magmática")
                
                # --- LÓGICA DE INTERPRETACIÓN AUTOMÁTICA ---
                st.markdown("### 🤖 Interpretación Geológica Asistida")
                prob_promedio = df_input['Prob_Fertilidad'].mean()
                num_anomalias = len(df_input[df_input['Clasificacion_IA'] == 'Fértil'])
                
                if prob_promedio > 0.6:
                    resumen_nlp = f"El conjunto de datos presenta una **alta prospectividad magmática**, con un promedio de fertilidad del {prob_promedio:.1%} y {num_anomalias} muestras clasificadas como fértiles. Se sugiere priorizar las zonas con razones Sr/Y elevadas."
                elif prob_promedio > 0.3:
                    resumen_nlp = "Se detectó una **prospectividad moderada**. El sistema muestra firmas geoquímicas mixtas, indicando zonas de transición que requieren un análisis más detallado de la alteración potásica (Cu/K)."
                else:
                    resumen_nlp = "El sistema presenta una **baja fertilidad magmática** predominante. Los indicadores de fraccionamiento Fe/Cr sugieren una firma de arco volcánico poco evolucionado o estéril para depósitos tipo pórfido."
                
                st.info(resumen_nlp)
                
        
                st.subheader("Análisis Multivariado de Fertilidad Magmática")
                fig = plt.figure(figsize=(16, 12))
                cmap_prob = 'coolwarm'
                
                # 1. Sr/Y vs Y
                ax1 = fig.add_subplot(231)
                sns.scatterplot(data=df_input, x='Y', y='Sr_Y', hue='Prob_Fertilidad', palette=cmap_prob, s=20, alpha=0.7, ax=ax1)
                ax1.set_xscale('log'); ax1.set_yscale('log'); ax1.axhline(20, color='gray', linestyle='--')
                ax1.set_title('Fertilidad: Sr/Y vs Y')

                # 2. Spider
                ax2 = fig.add_subplot(232)
                trace_elems = ['LA', 'SR', 'Y', 'ZR', 'TI', 'V', 'SC', 'CU', 'ZN', 'PB']
                means_fert = df_input[df_input['Clasificacion_IA'] == 'Fértil'][trace_elems].mean()
                means_inf = df_input[df_input['Clasificacion_IA'] != 'Fértil'][trace_elems].mean()
                ax2.plot(trace_elems, np.log10(means_fert + 1e-5), marker='o', color='darkred', label='Fértil')
                ax2.plot(trace_elems, np.log10(means_inf + 1e-5), marker='s', color='darkblue', label='Estéril')
                ax2.set_title('Spider de Elementos (Log10)')
                ax2.legend()

                # 3. Ternario AFM
                ax3 = fig.add_subplot(233)
                A, F, M = df_input['NA'] + df_input['K'], df_input['FE'], df_input['MG']
                Total = A + F + M + 1e-5
                X_tern, Y_tern = (M/Total) + (F/Total)/2.0, (F/Total) * np.sqrt(3)/2.0
                ax3.scatter(X_tern, Y_tern, c=df_input['Prob_Fertilidad'], cmap=cmap_prob, s=20)
                ax3.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], 'k-', lw=1.5)
                ax3.set_title('Diagrama Ternario AFM')
                ax3.axis('off')

                # 4. Fe vs Cr
                ax4 = fig.add_subplot(234)
                sns.scatterplot(data=df_input, x='CR', y='FE', hue='Prob_Fertilidad', palette=cmap_prob, s=20, ax=ax4)
                ax4.set_xscale('log'); ax4.set_yscale('log'); ax4.set_title('IA: Fe vs Cr')

                # 5. Cu vs K
                ax5 = fig.add_subplot(235)
                sns.scatterplot(data=df_input, x='K', y='CU', hue='Prob_Fertilidad', palette=cmap_prob, s=20, ax=ax5)
                ax5.set_xscale('log'); ax5.set_yscale('log'); ax5.set_title('IA: Cu vs K')

                # 6. V vs Ti
                ax6 = fig.add_subplot(236)
                sns.scatterplot(data=df_input, x='TI', y='V', hue='Prob_Fertilidad', palette=cmap_prob, s=20, ax=ax6)
                ax6.set_xscale('log'); ax6.set_yscale('log'); ax6.set_title('IA: V vs Ti')

                plt.tight_layout()
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
