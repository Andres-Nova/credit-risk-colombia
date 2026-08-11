"""
Genera report/index.html con métricas y gráficas del modelo ganador.
Uso: python src/report/generate_report.py
Requiere: data/processed/dataset_features.parquet y models/best_model.pkl
"""
import sys
from pathlib import Path

# Asegurar que el módulo src/ sea importable desde cualquier directorio
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from jinja2 import Template
from sklearn.metrics import roc_curve

from src.data.preprocess import dividir_datos
from src.models.train import obtener_modelos, entrenar_modelos
from src.models.evaluate import comparar_modelos


def generar_reporte(
    ruta_datos: str = 'data/processed/dataset_features.parquet',
    ruta_modelo: str = 'models/best_model.pkl',
    ruta_template: str = 'report/template.html',
    ruta_salida: str = 'report/index.html',
):
    """
    Carga el dataset y el modelo ganador, calcula métricas,
    genera gráficas con Plotly y renderiza el reporte HTML desde la plantilla Jinja2.
    """
    print("Cargando datos...")
    df = pd.read_parquet(ruta_datos)
    X_train, X_test, y_train, y_test = dividir_datos(df)

    print("Entrenando modelos para tabla comparativa (puede tardar ~5 min)...")
    modelos = obtener_modelos()
    modelos_entrenados = entrenar_modelos(X_train, y_train, modelos)
    tabla = comparar_modelos(modelos_entrenados, X_test, y_test)

    mejor = tabla.index[0]
    modelo_ganador = modelos_entrenados[mejor]

    print(f"Modelo ganador: {mejor}")
    print(tabla.to_string())

    # ── Gráfica ROC ──────────────────────────────────────────────────────────
    y_proba = modelo_ganador.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr,
        name=f"{mejor} (AUC={tabla.loc[mejor, 'auc_roc']})",
        line=dict(color='#1D5BA6', width=2)
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        line=dict(dash='dash', color='gray'),
        name='Azar'
    ))
    fig_roc.update_layout(
        xaxis_title='FPR', yaxis_title='TPR',
        title=f'Curva ROC — {mejor}',
        height=420
    )
    grafica_roc_html = fig_roc.to_html(full_html=False, include_plotlyjs='cdn')

    # ── Gráfica EDA — distribución de ingreso por default ───────────────────
    fig_eda = go.Figure()
    lim_sup = df['ingreso_mensual'].quantile(0.99)
    for default_val, color, nombre in [
        (0, '#2196F3', 'No Default'),
        (1, '#F44336', 'Default')
    ]:
        subset = df[df['default'] == default_val]['ingreso_mensual'].clip(upper=lim_sup)
        fig_eda.add_trace(go.Histogram(
            x=subset, name=nombre, marker_color=color, opacity=0.7, nbinsx=50
        ))
    fig_eda.update_layout(
        barmode='overlay',
        title='Distribución de Ingreso Mensual por Default',
        xaxis_title='Ingreso mensual',
        height=380
    )
    grafica_eda_html = fig_eda.to_html(full_html=False, include_plotlyjs=False)

    # ── Renderizar plantilla ─────────────────────────────────────────────────
    template_path = Path(ruta_template)
    template = Template(template_path.read_text())
    html = template.render(
        total_registros=f"{len(df):,}",
        tasa_default=f"{df['default'].mean():.1%}",
        mejor_modelo=mejor,
        mejor_auc=str(tabla.loc[mejor, 'auc_roc']),
        mejor_ks=str(tabla.loc[mejor, 'ks']),
        tabla_modelos=tabla.to_html(classes='', border=0, float_format='{:.4f}'.format),
        grafica_eda=grafica_eda_html,
        grafica_roc=grafica_roc_html,
    )

    salida = Path(ruta_salida)
    salida.write_text(html)
    print(f"\nReporte generado: {salida}")
    return str(salida)


if __name__ == '__main__':
    generar_reporte()
