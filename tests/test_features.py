"""Tests unitarios del pipeline de feature engineering colombiano."""
import pandas as pd
import numpy as np
import pytest
from src.features.build_features import construir_features


@pytest.fixture
def df_base():
    """DataFrame con columnas ya renombradas al español."""
    return pd.DataFrame({
        'default': [0, 1, 0, 1, 0, 0],
        'utilizacion_credito_rotativo': [0.5, 0.8, 0.2, 0.9, 0.3, 0.1],
        'edad': [25, 45, 35, 55, 28, 62],
        'veces_mora_30_59_dias': [0, 2, 0, 1, 0, 0],
        'relacion_deuda_ingreso': [0.3, 0.7, 0.2, 0.8, 0.1, 0.4],
        'ingreso_mensual': [2000, 800, 5000, 1500, 3500, 4200],
        'num_lineas_credito': [4, 2, 8, 1, 5, 6],
        'veces_mora_90_dias': [0, 1, 0, 2, 0, 0],
        'num_creditos_hipotecarios': [0, 0, 1, 0, 0, 1],
        'veces_mora_60_89_dias': [0, 0, 0, 1, 0, 0],
        'num_dependientes': [1, 2, 0, 3, 0, 1],
    })


def test_estrato_simulado_categorias_validas(df_base):
    """estrato_simulado debe generar solo valores del 1 al 6."""
    # Replicar para tener suficientes registros para 6 percentiles distintos
    df_grande = pd.concat([df_base] * 20, ignore_index=True)
    resultado = construir_features(df_grande)
    estratos = resultado['estrato_simulado'].unique()
    assert set(estratos).issubset({1, 2, 3, 4, 5, 6})
    assert resultado['estrato_simulado'].notna().all()


def test_capacidad_pago_no_negativa(df_base):
    """capacidad_pago nunca debe ser negativa (ingreso disponible ≥ 0)."""
    resultado = construir_features(df_base)
    assert (resultado['capacidad_pago'] >= 0).all()


def test_riesgo_mora_acumulado_no_negativo(df_base):
    """riesgo_mora_acumulado debe ser >= 0 para todos los registros."""
    resultado = construir_features(df_base)
    assert (resultado['riesgo_mora_acumulado'] >= 0).all()
