"""Configuración principal del dashboard de riesgo crediticio."""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from estilo import mostrar_header, mostrar_footer

st.set_page_config(
    page_title="Riesgo Crediticio — Colombia",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

mostrar_header(
    titulo="Dashboard de Riesgo Crediticio — Colombia",
    emoji="📊",
    descripcion=(
        "150,000 registros · XGBoost AUC 0.86 · KS 0.57 · "
        "Datos: Give Me Some Credit (Kaggle)"
    ),
)

st.markdown(
    """
    Navega por las secciones en la barra lateral izquierda.

    | Sección | Descripción |
    |---|---|
    | 📈 **Resumen** | KPIs del portafolio y distribución de scores crediticios |
    | 🎯 **Simulador** | Calcula el score para un perfil específico con explicación SHAP |
    | 🔍 **Modelo** | Comparación técnica de los 4 modelos evaluados |
    """
)

mostrar_footer()
