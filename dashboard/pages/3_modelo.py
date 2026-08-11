"""Página 3: Análisis técnico del modelo — comparación de los 4 modelos."""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.preprocess import dividir_datos
from src.models.train import obtener_modelos, entrenar_modelos
from src.models.evaluate import comparar_modelos, calcular_ks, calcular_curva_aprobacion

st.title("🔍 Análisis Técnico del Modelo")
st.markdown("Comparación de los 4 modelos evaluados con métricas estándar de scoring crediticio.")


@st.cache_data
def preparar_modelos():
    """Entrena los 4 modelos y calcula métricas — se cachea para la sesión."""
    ruta_datos = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'dataset_features.parquet'
    df = pd.read_parquet(str(ruta_datos))
    X_train, X_test, y_train, y_test = dividir_datos(df)
    modelos = obtener_modelos()
    modelos_entrenados = entrenar_modelos(X_train, y_train, modelos)
    return modelos_entrenados, X_test, y_test


try:
    with st.spinner("Entrenando modelos (puede tardar 1-2 minutos)..."):
        modelos_entrenados, X_test, y_test = preparar_modelos()

    # ── Tabla de métricas ─────────────────────────────────────────────────────
    st.subheader("Comparación de métricas")
    tabla = comparar_modelos(modelos_entrenados, X_test, y_test)
    st.dataframe(
        tabla.style.highlight_max(axis=0, color='#c8f7c5').format("{:.4f}"),
        use_container_width=True
    )
    st.caption(
        "**AUC-ROC:** métrica primaria | **KS:** separación máxima good/bad | "
        "**Gini:** = 2×AUC−1 | **PR-AUC:** precisión con desbalance"
    )

    st.divider()

    # ── ROC curves y curva de aprobación ──────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Curvas ROC")
        fig_roc = go.Figure()
        colores_modelo = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for (nombre, modelo), color in zip(modelos_entrenados.items(), colores_modelo):
            y_proba = modelo.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = tabla.loc[nombre, 'auc_roc']
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f"{nombre} ({auc})",
                line=dict(color=color, width=2)
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines',
            line=dict(dash='dash', color='gray', width=1),
            name='Azar (AUC=0.5)'
        ))
        fig_roc.update_layout(
            xaxis_title='Tasa de Falsos Positivos (FPR)',
            yaxis_title='Tasa de Verdaderos Positivos (TPR)',
            height=420, legend=dict(x=0.55, y=0.05)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col2:
        st.subheader("Aprobación vs Tasa de Mora")
        mejor = tabla.index[0]
        y_proba_mejor = modelos_entrenados[mejor].predict_proba(X_test)[:, 1]
        tasas_aprob, tasas_mora = calcular_curva_aprobacion(y_test.values, y_proba_mejor)

        fig_ap = go.Figure()
        fig_ap.add_trace(go.Scatter(
            x=tasas_aprob * 100,
            y=tasas_mora * 100,
            mode='lines',
            name=mejor,
            line=dict(color='#1f77b4', width=2)
        ))
        fig_ap.update_layout(
            xaxis_title='Tasa de aprobación (%)',
            yaxis_title='Tasa de mora en aprobados (%)',
            height=420,
        )
        st.plotly_chart(fig_ap, use_container_width=True)

    # ── KS Plot ───────────────────────────────────────────────────────────────
    st.subheader(f"KS Plot — {mejor}")
    mejor_modelo = modelos_entrenados[mejor]
    y_proba_ks = mejor_modelo.predict_proba(X_test)[:, 1]

    df_ks = pd.DataFrame({'proba': y_proba_ks, 'default': y_test.values})
    df_ks = df_ks.sort_values('proba')

    buenos = df_ks[df_ks['default'] == 0]['proba'].values
    malos = df_ks[df_ks['default'] == 1]['proba'].values

    umbral = np.linspace(0, 1, 200)
    cdf_buenos = [np.mean(buenos <= u) for u in umbral]
    cdf_malos = [np.mean(malos <= u) for u in umbral]

    fig_ks = go.Figure()
    fig_ks.add_trace(go.Scatter(x=umbral, y=cdf_buenos, name='No Default', line=dict(color='#2196F3')))
    fig_ks.add_trace(go.Scatter(x=umbral, y=cdf_malos, name='Default', line=dict(color='#F44336')))
    ks_val = tabla.loc[mejor, 'ks']
    fig_ks.update_layout(
        title=f'Distribución Acumulada — KS={ks_val:.4f}',
        xaxis_title='Probabilidad de default',
        yaxis_title='CDF acumulada',
        height=350,
    )
    st.plotly_chart(fig_ks, use_container_width=True)

except FileNotFoundError:
    st.error(
        "No se encontraron los datos procesados. "
        "Ejecuta los notebooks 01-03 primero."
    )
