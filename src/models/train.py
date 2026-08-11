"""Entrenamiento de los 4 modelos de riesgo crediticio."""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Columnas numéricas y categóricas del dataset con features colombianas
COLUMNAS_NUMERICAS = [
    'utilizacion_credito_rotativo', 'edad', 'veces_mora_30_59_dias',
    'relacion_deuda_ingreso', 'ingreso_mensual', 'num_lineas_credito',
    'veces_mora_90_dias', 'num_creditos_hipotecarios', 'veces_mora_60_89_dias',
    'num_dependientes', 'capacidad_pago', 'carga_financiera',
    'riesgo_mora_acumulado', 'tasa_dtf_vigente',
]
COLUMNAS_CATEGORICAS = ['segmento_edad', 'estrato_simulado']


def construir_preprocesador() -> ColumnTransformer:
    """
    Preprocesador sklearn: escala variables numéricas con StandardScaler
    y codifica categóricas con OneHotEncoder.
    """
    return ColumnTransformer(transformers=[
        ('num', StandardScaler(), COLUMNAS_NUMERICAS),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), COLUMNAS_CATEGORICAS),
    ])


def obtener_modelos(ratio_clases: float = 13.9) -> dict:
    """
    Retorna diccionario con 4 pipelines listos para entrenar.
    ratio_clases: n_negativos / n_positivos (93.3/6.7 ≈ 13.9 para Give Me Some Credit).
    Cada pipeline incluye preprocesador + clasificador.
    """
    modelos = {
        'Regresion Logistica': Pipeline([
            ('prep', construir_preprocesador()),
            ('modelo', LogisticRegression(
                C=0.1, max_iter=1000, class_weight='balanced', random_state=42
            )),
        ]),
        'Random Forest': Pipeline([
            ('prep', construir_preprocesador()),
            ('modelo', RandomForestClassifier(
                n_estimators=200, max_depth=10, class_weight='balanced',
                random_state=42, n_jobs=-1
            )),
        ]),
        'XGBoost': Pipeline([
            ('prep', construir_preprocesador()),
            ('modelo', XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                scale_pos_weight=ratio_clases, eval_metric='auc',
                random_state=42, n_jobs=-1, verbosity=0
            )),
        ]),
        'LightGBM': Pipeline([
            ('prep', construir_preprocesador()),
            ('modelo', LGBMClassifier(
                n_estimators=300, learning_rate=0.05, num_leaves=31,
                is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1
            )),
        ]),
    }
    return modelos


def entrenar_modelos(
    X_train: pd.DataFrame,
    y_train,
    modelos: dict,
) -> dict:
    """
    Entrena cada pipeline sobre los datos de entrenamiento.
    Retorna diccionario con modelos ajustados (mismo formato que la entrada).
    """
    modelos_entrenados = {}
    for nombre, pipeline in modelos.items():
        print(f"  Entrenando {nombre}...")
        pipeline.fit(X_train, y_train)
        modelos_entrenados[nombre] = pipeline
    return modelos_entrenados


def guardar_modelo(modelo, ruta: str = 'models/best_model.pkl'):
    """Serializa el modelo ganador con joblib."""
    joblib.dump(modelo, ruta)
    print(f"Modelo guardado en {ruta}")


def guardar_preprocesador(preprocesador, ruta: str = 'models/preprocessor.pkl'):
    """Serializa el preprocesador por separado para uso en el dashboard."""
    joblib.dump(preprocesador, ruta)
    print(f"Preprocesador guardado en {ruta}")
