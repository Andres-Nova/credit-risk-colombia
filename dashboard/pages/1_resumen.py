"""Página 1: Resumen del portafolio crediticio."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.evaluate import probabilidad_a_score, calcular_ks
from src.data.preprocess import dividir_datos
from sklearn.metrics import roc_auc_score

st.title("📈 Resumen del Portafolio")


@st.cache_data
def preparar_resumen():
    """Carga datos y calcula métricas del portafolio."""
    ruta_datos = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'dataset_features.parquet'
    ruta_modelo = Path(__file__).parent.parent.parent / 'models' / 'best_model.pkl'

    df = pd.read_parquet(str(ruta_datos))
    modelo = joblib.load(str(ruta_modelo))
    _, X_test, _, y_test = dividir_datos(df)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    scores = [probabilidad_a_score(p) for p in y_proba]
    return df, X_test, y_test, y_proba, scores


try:
    df, X_test, y_test, y_proba, scores = preparar_resumen()
    modelo = joblib.load(str(Path(__file__).parent.parent.parent / 'models' / 'best_model.pkl'))

    auc = roc_auc_score(y_test, y_proba)
    ks = calcular_ks(y_test.values, y_proba)

    # ── KPIs ────────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Registros analizados", f"{len(df):,}")
    col2.metric("Tasa de default", f"{df['default'].mean():.1%}")
    col3.metric("AUC-ROC (mejor modelo)", f"{auc:.4f}")
    col4.metric("KS Statistic", f"{ks:.4f}")

    st.divider()

    # ── Gráficas ────────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Distribución de scores")
        fig_score = px.histogram(
            x=scores,
            color=[str(d) for d in y_test.values],
            labels={'x': 'Score crediticio (300-850)', 'color': 'Default'},
            title='Score crediticio por Default',
            barmode='overlay', opacity=0.7, nbins=50,
            color_discrete_map={'0': '#2196F3', '1': '#F44336'},
        )
        fig_score.add_vline(x=500, line_dash='dash', line_color='#F44336',
                           annotation_text='Rojo ≤500')
        fig_score.add_vline(x=650, line_dash='dash', line_color='#4CAF50',
                           annotation_text='Verde >650')
        st.plotly_chart(fig_score, use_container_width=True)

    with col_b:
        st.subheader("Mora por estrato simulado")
        mora_estrato = df.groupby('estrato_simulado')['default'].mean().reset_index()
        mora_estrato.columns = ['Estrato', 'Tasa de mora']
        fig_estrato = px.bar(
            mora_estrato, x='Estrato', y='Tasa de mora',
            title='Tasa de Mora por Estrato Socioeconómico',
            color='Tasa de mora', color_continuous_scale='RdYlGn_r',
            labels={'Tasa de mora': 'Tasa de mora'},
        )
        fig_estrato.update_layout(yaxis_tickformat='.1%')
        st.plotly_chart(fig_estrato, use_container_width=True)

    # ── Top features ────────────────────────────────────────────────────────
    st.subheader("Top-10 variables — importancia del modelo")
    clasificador = modelo.named_steps['modelo']
    nombre_clase = type(clasificador).__name__
    if hasattr(clasificador, 'feature_importances_'):
        feature_names = modelo.named_steps['prep'].get_feature_names_out()
        importancias = pd.Series(
            clasificador.feature_importances_, index=feature_names
        ).nlargest(10)
        fig_fi = px.bar(
            x=importancias.values, y=importancias.index,
            orientation='h', title=f'Feature Importances — {nombre_clase}',
            color=importancias.values, color_continuous_scale='Blues',
        )
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Feature importances no disponibles para este tipo de modelo.")

except FileNotFoundError:
    st.error(
        "No se encontraron datos o modelo. "
        "Ejecuta los notebooks 01-03 primero para generar los archivos necesarios."
    )
