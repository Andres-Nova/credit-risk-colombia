"""Página 2: Simulador de score crediticio individual."""
import streamlit as st
from estilo import aplicar_estilo
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.evaluate import probabilidad_a_score, calcular_shap_values
from src.features.build_features import construir_features

st.title("🎯 Simulador de Score Crediticio")
st.markdown(
    "Ingresa el perfil del solicitante para calcular su score crediticio "
    "y entender qué factores lo determinan."
)

ruta_modelo = Path(__file__).parent.parent.parent / 'models' / 'best_model.pkl'


@st.cache_resource
def _cargar_modelo():
    return joblib.load(str(ruta_modelo))


try:
    modelo = _cargar_modelo()
except FileNotFoundError:
    st.error("Modelo no encontrado. Ejecuta el notebook 03 primero.")
    st.stop()

# ── Formulario ────────────────────────────────────────────────────────────────
with st.form("formulario_score"):
    st.subheader("Perfil del solicitante")
    col1, col2 = st.columns(2)

    with col1:
        ingreso = st.number_input(
            "Ingreso mensual (USD)", min_value=0, max_value=50000,
            value=3000, step=100,
            help="Ingreso mensual declarado del solicitante"
        )
        ratio_deuda = st.slider(
            "Relación deuda/ingreso", 0.0, 1.0, 0.3, 0.01,
            help="Proporción de los ingresos destinada al pago de deudas"
        )
        edad = st.number_input("Edad", min_value=18, max_value=100, value=35)
        num_lineas = st.number_input(
            "Número de líneas de crédito", min_value=0, max_value=50, value=5
        )
        utilizacion = st.slider(
            "Utilización crédito rotativo", 0.0, 1.0, 0.3, 0.01,
            help="Porcentaje del cupo de tarjetas usado actualmente"
        )

    with col2:
        mora_90 = st.number_input("Veces en mora +90 días (histórico)", 0, 20, 0)
        mora_60 = st.number_input("Veces en mora 60-89 días (histórico)", 0, 20, 0)
        mora_30 = st.number_input("Veces en mora 30-59 días (histórico)", 0, 20, 0)
        num_hipotecas = st.number_input("Créditos hipotecarios activos", 0, 20, 0)
        num_dependientes = st.number_input("Número de dependientes", 0, 20, 1)

    enviado = st.form_submit_button("Calcular Score", type="primary", use_container_width=True)

# ── Resultado ─────────────────────────────────────────────────────────────────
if enviado:
    # Construir registro del solicitante con todas las columnas
    registro = pd.DataFrame([{
        'default': 0,
        'utilizacion_credito_rotativo': utilizacion,
        'edad': edad,
        'veces_mora_30_59_dias': mora_30,
        'relacion_deuda_ingreso': ratio_deuda,
        'ingreso_mensual': ingreso,
        'num_lineas_credito': num_lineas,
        'veces_mora_90_dias': mora_90,
        'num_creditos_hipotecarios': num_hipotecas,
        'veces_mora_60_89_dias': mora_60,
        'num_dependientes': num_dependientes,
    }])
    registro = construir_features(registro)
    X = registro.drop(columns=['default'])

    proba = modelo.predict_proba(X)[0, 1]
    score = probabilidad_a_score(proba)

    # Semáforo
    if score > 650:
        color_hex = "#4CAF50"
        nivel = "BAJO RIESGO"
        emoji = "🟢"
    elif score > 500:
        color_hex = "#FF9800"
        nivel = "RIESGO MEDIO"
        emoji = "🟡"
    else:
        color_hex = "#F44336"
        nivel = "ALTO RIESGO"
        emoji = "🔴"

    col_score, col_shap = st.columns([1, 2])

    with col_score:
        st.markdown(f"""
        <div style='text-align:center; padding:24px; border-radius:12px;
                    background:{color_hex}22; border: 2px solid {color_hex}; margin-top:16px'>
            <p style='font-size:0.9em; color:#666; margin:0'>Score crediticio</p>
            <h1 style='color:{color_hex}; margin:8px 0; font-size:3.5em'>{score}</h1>
            <p style='font-size:1.2em; color:{color_hex}; margin:0'>{emoji} {nivel}</p>
            <hr style='border-color:{color_hex}44; margin:16px 0'>
            <p style='margin:0; color:#555'>Probabilidad de default:</p>
            <p style='font-size:1.4em; font-weight:bold; color:{color_hex}; margin:4px 0'>
                {proba:.1%}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.caption("**Escala de referencia:**")
        st.markdown("""
        - 🟢 > 650 — Riesgo bajo
        - 🟡 501-650 — Riesgo medio
        - 🔴 ≤ 500 — Riesgo alto
        """)

    with col_shap:
        st.subheader("Factores que determinaron el score")
        try:
            shap_values, explainer, X_trans = calcular_shap_values(modelo, X)
            feature_names = modelo.named_steps['prep'].get_feature_names_out()
            importancias = pd.Series(
                shap_values[0] if shap_values.ndim > 1 else shap_values,
                index=feature_names
            )
            top = importancias.abs().nlargest(10)
            vals = importancias[top.index].values
            colores = ['#F44336' if v > 0 else '#4CAF50' for v in vals]

            fig = go.Figure(go.Bar(
                x=vals,
                y=[n.replace('num__', '').replace('cat__', '') for n in top.index.tolist()],
                orientation='h',
                marker_color=colores,
            ))
            fig.update_layout(
                title="Impacto de cada variable (rojo = aumenta riesgo, verde = lo reduce)",
                xaxis_title="Valor SHAP",
                height=380,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Valores SHAP: miden cuánto contribuye cada variable a aumentar (+) "
                "o reducir (-) la probabilidad de default."
            )
        except Exception as e:
            st.info(f"Explicación SHAP no disponible para este modelo: {e}")
