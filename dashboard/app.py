"""Configuración principal del dashboard de riesgo crediticio."""
import streamlit as st
import joblib
import pandas as pd
import sys
from pathlib import Path

# Asegurar que src/ esté en el path cuando se corre desde dashboard/
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Riesgo Crediticio — Colombia",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def cargar_modelo():
    """Carga el modelo serializado desde models/best_model.pkl."""
    ruta = Path(__file__).parent.parent / 'models' / 'best_model.pkl'
    return joblib.load(str(ruta))


@st.cache_data
def cargar_datos():
    """Carga el dataset con features colombianas desde data/processed/."""
    ruta = Path(__file__).parent.parent / 'data' / 'processed' / 'dataset_features.parquet'
    return pd.read_parquet(str(ruta))


# Página principal — introducción
st.title("📊 Dashboard de Riesgo Crediticio — Colombia")
st.markdown("""
Análisis de riesgo crediticio con contexto colombiano.
Navega por las secciones en la barra lateral izquierda.

| Sección | Descripción |
|---|---|
| 📈 **Resumen** | KPIs del portafolio y distribución de scores crediticios |
| 🎯 **Simulador** | Calcula el score para un perfil específico con explicación SHAP |
| 🔍 **Modelo** | Comparación técnica de los 4 modelos evaluados |

---

**Origen de los datos:** Give Me Some Credit (Kaggle) — 150.000 registros con target observado.
Variables renombradas al español y enriquecidas con features de contexto colombiano.
""")

st.info(
    "⚠️ Los datos deben estar disponibles en `data/processed/dataset_features.parquet` "
    "y el modelo en `models/best_model.pkl`. Ejecuta los notebooks 01-03 primero.",
    icon="ℹ️"
)
