import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata

# Nuevas importaciones para el chatbot de imágenes
import io
import os
import base64
import requests  # se usa en la plantilla HTTP. Asegúrate de tener requests instalado.

# Opcional: si usas el cliente oficial de Google Generative API (si está disponible)
try:
    import google.generativeai as genai  # opcional, solo si usas este SDK
    HAS_GENAI = True
except Exception:
    HAS_GENAI = False

# =====================================================================
# CONFIGURACIÓN DE LA PÁGINA
# =====================================================================
st.set_page_config(page_title="Fertilidad Geoquímica", layout="wide", page_icon="🌋")

# Configuración de estilo visual para Seaborn
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

# Lista estricta de elementos requeridos
elementos_requeridos = ['AU', 'AG', 'CU', 'PB', 'ZN', 'MO', 'NI', 'CO', 'CD', 'BI',
                        'FE', 'MN', 'TE', 'BA', 'CR', 'V', 'SN', 'W', 'LA', 'AL',
                        'MG', 'CA', 'NA', 'K', 'SR', 'Y', 'GA', 'LI', 'NB', 'SC',
                        'TA', 'TI', 'ZR', 'AS', 'SB', 'HG', 'PT', 'PD']

# =====================================================================
# FUNCIONES AUXILIARES PARA DESCRIPCIÓN DE IMÁGENES (GEMINI)
# =====================================================================

def fig_to_bytes(fig):
    """Convierte una figura matplotlib a bytes PNG en memoria."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    return buf.getvalue()

def describe_with_gemini_http(image_bytes: bytes, prompt: str, api_key: str, model: str = "gemini-image-placeholder"):
    """
    Plantilla: realiza una petición HTTP a tu endpoint Gemini.
    - Reemplaza `url` y la estructura de `payload` según la API que uses (Vertex AI, PaLM REST, etc.).
    - api_key: pone tu clave en la cabecera Authorization: Bearer <KEY> o según tu método.
    """
    # --- EJEMPLO GENÉRICO (REEMPLAZAR) ---
    # Nota: Esta URL y payload son de ejemplo/placeholder. Cámbialos por el endpoint correcto.
    url = "https://GENERATIVE_API_ENDPOINT/v1/models/{model}:predict".format(model=model)
    headers = {
        "Authorization": f"Bearer {api_key}",
        # si la API requiere otro content-type, actualizarlo
    }
    # Si la API acepta archivos multipart:
    files = {
        "image": ("plot.png", image_bytes, "image/png")
    }
    data = {
        "prompt": prompt,
        # otros parámetros que tu endpoint requiera
    }
    try:
        resp = requests.post(url, headers=headers, files=files, data=data, timeout=60)
        resp.raise_for_status()
        return resp.json()  # adapta según la estructura real de la respuesta
    except Exception as e:
        return {"error": str(e)}

def describe_with_genai_sdk(image_bytes: bytes, prompt: str, model: str = "gemini-image-alpha-1"):
    """
    Ejemplo usando `google.generativeai` si lo tienes configurado.
    - Necesitas: export GEMINI_API_KEY=tu_api_key
    - Este bloque es un ejemplo; adapta según la versión del SDK que estés usando.
    """
    if not HAS_GENAI:
        return {"error": "google.generativeai no está instalado"}
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"error": "No se encontró GEMINI_API_KEY en variables de entorno"}
    genai.configure(api_key=api_key)
    # El uso exacto del SDK para multimodal puede variar. Este es un esqueleto ilustrativo.
    try:
        # Convertir a base64 (sólo si el SDK necesita la imagen inline)
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        # Ejemplo hipotético de llamada multimodal tipo chat (adaptar a tu SDK)
        response = genai.chat.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
                # Algunos SDKs permiten anexar image data en una estructura separada
            ],
            # Si el SDK soporta adjuntar la imagen en la petición, úsalo:
            # media=[{"type":"image", "data": img_b64}]
        )
        return response  # adapta según respuesta real
    except Exception as e:
        return {"error": str(e)}

# =====================================================================
# PANEL LATERAL (SIDEBAR)
# =====================================================================
st.sidebar.header("⚙️ Panel de Control")
st.sidebar.write("Sube la matriz analítica del laboratorio.")
archivo_subido = st.sidebar.file_uploader("Formato CSV o Excel", type=['csv', 'xlsx'])

# =====================================================================
# ÁREA PRINCIPAL
# =====================================================================
if archivo_subido is not None:
    if archivo_subido.name.endswith('.csv'):
        df_input = pd.read_csv(archivo_subido)
    else:
        df_input = pd.read_excel(archivo_subido)
        
    st.sidebar.success(f"Archivo cargado: {len(df_input)} muestras.")
    
    if st.sidebar.button("🚀 Ejecutar Modelo Predictivo"):
        with st.spinner('Procesando datos y renderizando gráficos espaciales...'):
            try:
                # 1. Inferencia Predictiva
                datos_modelo = df_input[elementos_requeridos].copy()
                datos_log = np.log10(datos_modelo + 1e-5)
                datos_escalados = scaler.transform(datos_log)
                probabilidades = modelo_rf.predict_proba(datos_escalados)[:, 1]
                
                df_input['Prob_Fertilidad'] = probabilidades
                df_input['Clasificacion_IA'] = np.where(df_input['Prob_Fertilidad'] > 0.5, 'Fértil', 'Estéril/Artefacto')
                
                # Calcular Sr/Y para los gráficos
                if 'SR' in df_input.columns and 'Y' in df_input.columns:
                    df_input['Sr_Y'] = df_input['SR'] / df_input['Y']
                
                # =====================================================================
                # CREACIÓN DE PESTAÑAS (TABS) PARA ORGANIZAR LA VISTA
                # =====================================================================
                tab1, tab2, tab3 = st.tabs(["📄 Resumen y Datos", "📉 Diagramas Geoquímicos", "🗺️ Mapa de Calor Espacial"])
                
                # ----- PESTAÑA 1: DATOS Y KPIS -----
                with tab1:
                    st.subheader("📊 Resumen Analítico")
                    total_muestras = len(df_input)
                    muestras_fertiles = len(df_input[df_input['Clasificacion_IA'] == 'Fértil'])
                    porcentaje_anomalias = (muestras_fertiles / total_muestras) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Muestras Analizadas", total_muestras)
                    col2.metric("Blancos Fértiles Detectados", muestras_fertiles)
                    col3.metric("Tasa de Anomalía", f"{porcentaje_anomalias:.2f}%")
                    
                    st.markdown("---")
                    
                    st.markdown("### 📋 Vista Previa de Muestras (Primeras 500 filas)")
                    
                    def colorear_fertiles(val):
                        color = '#ffcccc' if val == 'Fértil' else ''
                        return f'background-color: {color}'
                    
                    df_preview = df_input.head(500)
                    st.dataframe(
                        df_preview.style.applymap(colorear_fertiles, subset=['Clasificacion_IA']), 
                        use_container_width=True
                    )
                    
                    st.caption(f"💡 Mostrando una vista previa optimizada de 500 de las {total_muestras} muestras totales. El archivo de descarga contendrá el dataset completo.")
                
                # ----- PESTAÑA 2: DIAGRAMAS GEOQUÍMICOS -----
                with tab2:
                    st.subheader("Análisis Multivariado de Elementos Traza y Mayores")
                    fig = plt.figure(figsize=(16, 12))
                    cmap_prob = 'coolwarm'
                    
                    # 1. Sr/Y vs Y
                    ax1 = fig.add_subplot(231)
                    sns.scatterplot(data=df_input, x='Y', y='Sr_Y', hue='Prob_Fertilidad', palette=cmap_prob, s=20, alpha=0.7, ax=ax1)
                    ax1.set_xscale('log')
                    ax1.set_yscale('log')
                    ax1.axhline(20, color='gray', linestyle='--')
                    ax1.set_title('Fertilidad: Sr/Y vs Y')
                    ax1.legend([],[], frameon=False)

                    # 2. Spider
                    ax2 = fig.add_subplot(232)
                    trace_elems = ['LA', 'SR', 'Y', 'ZR', 'TI', 'V', 'SC', 'CU', 'ZN', 'PB']
                    means_fert = df_input[df_input['Clasificacion_IA'] == 'Fértil'][trace_elems].mean()
                    means_inf = df_input[df_input['Clasificacion_IA'] != 'Fértil'][trace_elems].mean()
                    ax2.plot(trace_elems, np.log10(means_fert + 1e-5), marker='o', color='darkred', lw=2, label='Fértil (Prom)')
                    ax2.plot(trace_elems, np.log10(means_inf + 1e-5), marker='s', color='darkblue', lw=2, label='Infértil (Prom)')
                    ax2.set_title('Spider de Elementos Traza (Log10)')
                    ax2.legend()

                    # 3. Ternario AFM Simplificado
                    ax3 = fig.add_subplot(233)
                    A, F, M = df_input['NA'] + df_input['K'], df_input['FE'], df_input['MG']
                    Total = A + F + M + 1e-5
                    X_tern = (M/Total) + (F/Total) / 2.0
                    Y_tern = (F/Total) * np.sqrt(3) / 2.0
                    sc = ax3.scatter(X_tern, Y_tern, c=df_input['Prob_Fertilidad'], cmap=cmap_prob, s=20, alpha=0.7)
                    ax3.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], 'k-', lw=1.5)
                    ax3.text(-0.05, -0.05, 'Na+K', ha='center')
                    ax3.text(1.05, -0.05, 'Mg', ha='center')
                    ax3.text(0.5, np.sqrt(3)/2 + 0.05, 'Fe', ha='center')
                    ax3.set_title('Diagrama Ternario AFM')
                    ax3.axis('off')

                    # 4. Fe vs Cr
                    ax4 = fig.add_subplot(234)
                    sns.scatterplot(data=df_input, x='CR', y='FE', hue='Prob_Fertilidad', palette=cmap_prob, s=20, alpha=0.7, ax=ax4)
                    ax4.set_xscale('log')
                    ax4.set_yscale('log')
                    ax4.set_title('IA: Fe vs Cr (Fraccionamiento)')
                    ax4.legend([],[], frameon=False)

                    # 5. Cu vs K
                    ax5 = fig.add_subplot(235)
                    sns.scatterplot(data=df_input, x='K', y='CU', hue='Prob_Fertilidad', palette=cmap_prob, s=20, alpha=0.7, ax=ax5)
                    ax5.set_xscale('log')
                    ax5.set_yscale('log')
                    ax5.set_title('IA: Cu vs K (Alteración Potásica)')
                    ax5.legend([],[], frameon=False)

                    # 6. V vs Ti
                    ax6 = fig.add_subplot(236)
                    sns.scatterplot(data=df_input, x='TI', y='V', hue='Prob_Fertilidad', palette=cmap_prob, s=20, alpha=0.7, ax=ax6)
                    ax6.set_xscale('log')
                    ax6.set_yscale('log')
                    ax6.set_title('IA: V vs Ti (Evolución Magmática)')
                    ax6.legend([],[], frameon=False)

                    plt.tight_layout()
                    # Mostrar figura en Streamlit
                    st.pyplot(fig)

                    # --- INTEGRACIÓN DEL CHATBOT GEMINI ---
                    st.markdown("---")
                    st.markdown("### 🤖 Describir figura con Gemini")
                    st.markdown("Haz clic para generar una descripción automática de la figura (usa tu cuenta/endpoint de Gemini).")

                    # Botón para generar la descripción
                    if st.button("📝 Describir figura (Gemini)"):
                        with st.spinner("Enviando imagen a Gemini y esperando respuesta..."):
                            # Convertir figura a bytes
                            image_bytes = fig_to_bytes(fig)

                            # Prompt que vas a enviar al modelo (modifica según quieras)
                            prompt = (
                                "En español: Describe la figura adjunta. "
                                "Indica patrones relevantes, anomalías, diferencias entre muestras fértiles e infértiles, "
                                "y una interpretación geológica corta (3-5 puntos). Usa un tono técnico pero claro."
                            )

                            # Opción 1: usar SDK google.generativeai (si está instalado y configurado)
                            if HAS_GENAI and os.environ.get("GEMINI_API_KEY"):
                                result = describe_with_genai_sdk(image_bytes, prompt, model="gemini-image-alpha-1")
                                st.write("Respuesta (SDK):")
                                st.write(result)
                            else:
                                # Opción 2: usar petición HTTP genérica (rellena tu endpoint y API key en variable de entorno)
                                api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY") if hasattr(st, 'secrets') else None
                                if not api_key:
                                    st.error("No se encontró la variable de entorno GEMINI_API_KEY. Define tu clave de API en GEMINI_API_KEY.")
                                else:
                                    resp = describe_with_gemini_http(image_bytes, prompt, api_key, model="gemini-image-placeholder")
                                    st.write("Respuesta (HTTP):")
                                    st.json(resp)

                # ----- PESTAÑA 3: MAPA ESPACIAL -----
                with tab3:
                    st.subheader("Modelamiento Espacial de Prospectividad")
                    if 'LONGITUD' in df_input.columns and 'LATITUD' in df_input.columns:
                        X_coord = df_input['LONGITUD'].values
                        Y_coord = df_input['LATITUD'].values
                        Z_prob = df_input['Prob_Fertilidad'].values
                        
                        grid_x, grid_y = np.mgrid[X_coord.min():X_coord.max():200j, Y_coord.min():Y_coord.max():200j]
                        grid_z = griddata((X_coord, Y_coord), Z_prob, (grid_x, grid_y), method='linear')
                        
                        fig_map = plt.figure(figsize=(10, 8))
                        mapa_calor = plt.imshow(grid_z.T, extent=(X_coord.min(), X_coord.max(), Y_coord.min(), Y_coord.max()),
                                                origin='lower', cmap='coolwarm', alpha=0.85)
                        
                        contornos = plt.contour(grid_x, grid_y, grid_z, levels=[0.5, 0.7, 0.9], colors=['yellow', 'orange', 'darkred'], linewidths=1.5, linestyles='--')
                        plt.clabel(contornos, inline=True, fontsize=10, fmt='Prob: %.1f')
                        
                        plt.title('Mapa de Calor: Certeza de Fertilidad Magmática')
                        plt.xlabel('Longitud')
                        plt.ylabel('Latitud')
                        cbar = plt.colorbar(mapa_calor, shrink=0.8)
                        cbar.set_label('Índice de Certeza')
                        st.pyplot(fig_map)
                    else:
                        st.warning("⚠️ El archivo subido no contiene columnas de 'LATITUD' y 'LONGITUD'. No se puede generar el mapa espacial.")
                
                # Preparar descarga
                csv_buffer = df_input.to_csv(index=False).encode('utf-8')
                st.sidebar.markdown("---")
                st.sidebar.download_button("📥 Descargar Resultados", data=csv_buffer, file_name="Resultados_Predictivos.csv", mime="text/csv")
                
            except KeyError as e:
                st.error(f"⚠️ Error: Falta la columna {e} en el archivo. Se requieren 38 elementos químicos.")
            except Exception as e:
                st.error(f"⚠️ Error al procesar: {e}")
else:
    st.info("👈 Por favor, carga una matriz de datos en el panel lateral.")
