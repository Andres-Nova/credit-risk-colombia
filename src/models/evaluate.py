"""Métricas de evaluación estándar para modelos de riesgo crediticio."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve
)


def calcular_ks(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Estadístico KS (Kolmogorov-Smirnov).
    Mide la máxima separación entre las distribuciones acumuladas
    de solicitantes buenos y malos. Métrica estándar en scoring crediticio.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(tpr - fpr))


def calcular_metricas(modelo, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Calcula métricas completas para un modelo entrenado.

    Retorna dict con claves:
    - auc_roc: Área bajo la curva ROC (métrica primaria)
    - ks: Estadístico KS
    - gini: Coeficiente Gini = 2×AUC - 1 (estándar en reportes de riesgo)
    - pr_auc: Área bajo la curva Precision-Recall (útil con desbalance)
    """
    y_proba = modelo.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    ks = calcular_ks(y_test.values, y_proba)
    return {
        'auc_roc': round(auc, 4),
        'ks': round(ks, 4),
        'gini': round(2 * auc - 1, 4),
        'pr_auc': round(average_precision_score(y_test, y_proba), 4),
    }


def comparar_modelos(
    modelos_entrenados: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Genera tabla comparativa de métricas para todos los modelos.
    Retorna DataFrame ordenado de mayor a menor AUC-ROC.
    """
    resultados = []
    for nombre, modelo in modelos_entrenados.items():
        metricas = calcular_metricas(modelo, X_test, y_test)
        metricas['modelo'] = nombre
        resultados.append(metricas)
    df = pd.DataFrame(resultados).set_index('modelo')
    return df.sort_values('auc_roc', ascending=False)


def calcular_curva_aprobacion(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_puntos: int = 100,
) -> tuple:
    """
    Curva de trade-off entre tasa de aprobación y tasa de mora.
    A menor umbral de score → más aprobaciones → más mora.

    Retorna (tasas_aprobacion, tasas_mora) — arrays de n_puntos.
    """
    umbrales = np.linspace(0, 1, n_puntos)
    tasas_aprobacion = []
    tasas_mora = []
    for umbral in umbrales:
        aprobados = y_proba <= umbral
        tasa_aprobacion = aprobados.mean()
        tasa_mora = y_true[aprobados].mean() if aprobados.sum() > 0 else 0.0
        tasas_aprobacion.append(tasa_aprobacion)
        tasas_mora.append(tasa_mora)
    return np.array(tasas_aprobacion), np.array(tasas_mora)


def calcular_shap_values(modelo, X_muestra: pd.DataFrame) -> tuple:
    """
    Calcula SHAP values para explicar predicciones individuales.
    Usa TreeExplainer para modelos de árbol (XGBoost, LightGBM, RF)
    o LinearExplainer para Regresión Logística.

    Retorna (shap_values, explainer, X_transformado).
    """
    import shap  # import lazy — solo cuando se necesita SHAP
    # El modelo es un Pipeline sklearn — extraer pasos
    clasificador = modelo.named_steps['modelo']
    preprocesador = modelo.named_steps['prep']
    X_transformado = preprocesador.transform(X_muestra)

    nombre_clase = type(clasificador).__name__
    if nombre_clase in ('XGBClassifier', 'LGBMClassifier', 'RandomForestClassifier'):
        explainer = shap.TreeExplainer(clasificador)
        shap_values = explainer.shap_values(X_transformado)
        # RandomForest retorna lista [clase_0, clase_1] — tomar clase positiva
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        explainer = shap.LinearExplainer(clasificador, X_transformado)
        shap_values = explainer.shap_values(X_transformado)

    return shap_values, explainer, X_transformado


def probabilidad_a_score(
    proba: float,
    pdo: int = 20,
    odds_base: float = 19.0,
    base: int = 600,
) -> int:
    """
    Convierte probabilidad de default a score crediticio estilo FICO (300-850).

    Parámetros de calibración:
    - pdo: puntos que duplican las odds (20 = estándar)
    - odds_base: odds del punto de calibración (19 ≈ 95% buenos / 5% malos)
    - base: score del punto de calibración (600)

    Semáforo de riesgo:
    - Verde  > 650: riesgo bajo
    - Amarillo 501-650: riesgo medio
    - Rojo   ≤ 500: riesgo alto
    """
    factor = pdo / np.log(2)
    offset = base - factor * np.log(odds_base)
    # Evitar log(0)
    proba = np.clip(proba, 1e-6, 1 - 1e-6)
    odds = (1 - proba) / proba
    score = offset + factor * np.log(odds)
    return int(np.clip(score, 300, 850))
