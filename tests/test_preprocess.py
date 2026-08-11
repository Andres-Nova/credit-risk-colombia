"""Tests unitarios del pipeline de limpieza y preprocesamiento."""
import pandas as pd
import numpy as np
import pytest
from src.data.preprocess import limpiar_datos, renombrar_columnas, dividir_datos


@pytest.fixture
def df_sucio():
    """Dataset mínimo con los problemas conocidos del dataset real."""
    return pd.DataFrame({
        'SeriousDlqin2yrs': [0, 1, 0, 1, 0],
        'RevolvingUtilizationOfUnsecuredLines': [0.5, 1.8, 0.3, 2.5, 0.1],
        'age': [35, 25, 150, 45, 30],
        'NumberOfTime30-59DaysPastDueNotWorse': [0, 1, 0, 2, 0],
        'DebtRatio': [0.3, 0.5, 0.2, 0.4, 0.1],
        'MonthlyIncome': [3000.0, np.nan, 5000.0, np.nan, 4000.0],
        'NumberOfOpenCreditLinesAndLoans': [5, 3, 8, 2, 6],
        'NumberOfTimes90DaysLate': [0, 1, 0, 0, 0],
        'NumberRealEstateLoansOrLines': [1, 0, 2, 0, 1],
        'NumberOfTime60-89DaysPastDueNotWorse': [0, 0, 0, 1, 0],
        'NumberOfDependents': [2.0, np.nan, 1.0, 3.0, 0.0],
    })


def test_nan_ingreso_imputado(df_sucio):
    """MonthlyIncome NaN debe imputarse — no debe quedar ningún NaN."""
    resultado = renombrar_columnas(limpiar_datos(df_sucio))
    assert resultado['ingreso_mensual'].isna().sum() == 0


def test_nan_dependientes_imputado(df_sucio):
    """NumberOfDependents NaN debe imputarse con 0."""
    resultado = renombrar_columnas(limpiar_datos(df_sucio))
    assert resultado['num_dependientes'].isna().sum() == 0


def test_outliers_utilizacion_capados(df_sucio):
    """RevolvingUtilization > 1.0 debe caparlo a 1.0."""
    resultado = renombrar_columnas(limpiar_datos(df_sucio))
    assert resultado['utilizacion_credito_rotativo'].max() <= 1.0


def test_edades_invalidas_eliminadas(df_sucio):
    """Registros con age < 18 o > 100 deben eliminarse (age=150 en fixture)."""
    resultado = limpiar_datos(df_sucio)
    assert len(resultado) == 4  # 5 registros - 1 con age=150


def test_split_estratificado_mantiene_proporcion(df_sucio):
    """El split 80/20 debe mantener la proporción de default ± 5%."""
    df_limpio = renombrar_columnas(limpiar_datos(df_sucio))
    # Replicar para tener suficientes datos para el split
    df_grande = pd.concat([df_limpio] * 50, ignore_index=True)
    X_tr, X_te, y_tr, y_te = dividir_datos(df_grande)
    prop_train = y_tr.mean()
    prop_test = y_te.mean()
    assert abs(prop_train - prop_test) < 0.05
