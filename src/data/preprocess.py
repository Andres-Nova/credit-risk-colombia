"""Pipeline de limpieza y preprocesamiento del dataset de riesgo crediticio."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Mapeo de columnas original → español/CO
RENOMBRADO = {
    'SeriousDlqin2yrs':                       'default',
    'RevolvingUtilizationOfUnsecuredLines':    'utilizacion_credito_rotativo',
    'age':                                     'edad',
    'NumberOfTime30-59DaysPastDueNotWorse':    'veces_mora_30_59_dias',
    'DebtRatio':                               'relacion_deuda_ingreso',
    'MonthlyIncome':                           'ingreso_mensual',
    'NumberOfOpenCreditLinesAndLoans':         'num_lineas_credito',
    'NumberOfTimes90DaysLate':                 'veces_mora_90_dias',
    'NumberRealEstateLoansOrLines':            'num_creditos_hipotecarios',
    'NumberOfTime60-89DaysPastDueNotWorse':    'veces_mora_60_89_dias',
    'NumberOfDependents':                      'num_dependientes',
}

BINS_EDAD = [17, 30, 45, 60, 200]
ETIQUETAS_EDAD = ['18-30', '31-45', '46-60', '60+']


def cargar_datos(ruta: str) -> pd.DataFrame:
    """Carga el CSV raw de Kaggle."""
    return pd.read_csv(ruta, index_col=0)


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el dataset aplicando:
    - Eliminación de registros con edad fuera de rango (< 18 o > 100)
    - Imputación de MonthlyIncome con mediana por grupo de edad
    - Imputación de NumberOfDependents con 0 (moda)
    - Cap de RevolvingUtilizationOfUnsecuredLines a 1.0
    - Eliminación de duplicados exactos
    """
    df = df.copy()

    # Eliminar registros con edad fuera de rango
    df = df[(df['age'] >= 18) & (df['age'] <= 100)]

    # Imputar MonthlyIncome con mediana por grupo de edad
    df['_grupo_edad'] = pd.cut(df['age'], bins=BINS_EDAD, labels=ETIQUETAS_EDAD)
    mediana_por_grupo = df.groupby('_grupo_edad', observed=True)['MonthlyIncome'].transform('median')
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(mediana_por_grupo)
    # Si quedan NaN (grupo sin datos suficientes), usar mediana global
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
    df = df.drop(columns=['_grupo_edad'])

    # Imputar NumberOfDependents con 0 (moda)
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(0)

    # Capar RevolvingUtilization a 1.0
    df['RevolvingUtilizationOfUnsecuredLines'] = df[
        'RevolvingUtilizationOfUnsecuredLines'
    ].clip(upper=1.0)

    # Eliminar duplicados exactos
    df = df.drop_duplicates()

    return df.reset_index(drop=True)


def renombrar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra columnas al español con contexto colombiano."""
    columnas_presentes = {k: v for k, v in RENOMBRADO.items() if k in df.columns}
    return df.rename(columns=columnas_presentes)


def dividir_datos(
    df: pd.DataFrame,
    objetivo: str = 'default',
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Split estratificado 80/20 por la variable objetivo.
    Retorna (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=[objetivo])
    y = df[objetivo]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
