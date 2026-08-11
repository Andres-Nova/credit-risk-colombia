"""Feature engineering con contextualización colombiana para el dataset de riesgo crediticio."""
import pandas as pd
import numpy as np

# DTF promedio histórico Banco de la República 2005-2010 (período del dataset)
DTF_HISTORICA = 4.8

# Percentiles para asignación de estrato según ingreso mensual
ESTRATOS_PERCENTILES = [0, 10, 25, 50, 70, 85, 100]


def agregar_estrato_simulado(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asigna estrato socioeconómico 1-6 según percentil de ingreso mensual.
    Simula la clasificación de estratos DANE colombiana calibrada a la
    distribución del dataset.
    """
    df = df.copy()
    try:
        # Intentar con qcut — maneja duplicados automáticamente
        df['estrato_simulado'] = pd.qcut(
            df['ingreso_mensual'],
            q=6,
            labels=False,
            duplicates='drop'
        ).add(1).astype('Int64')
    except Exception:
        # Fallback: searchsorted sobre percentiles calculados
        cortes = np.percentile(df['ingreso_mensual'].dropna(), ESTRATOS_PERCENTILES)
        cortes = np.unique(cortes)
        df['estrato_simulado'] = np.searchsorted(
            cortes[1:], df['ingreso_mensual'].values, side='right'
        ).clip(1, 6)
    return df


def agregar_capacidad_pago(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ingreso disponible después de cubrir deudas.
    Fórmula: ingreso_mensual × (1 − min(relacion_deuda_ingreso, 1))
    Indicador estándar en análisis de crédito colombiano.
    """
    df = df.copy()
    ratio_capado = df['relacion_deuda_ingreso'].clip(upper=1.0)
    df['capacidad_pago'] = df['ingreso_mensual'] * (1 - ratio_capado)
    return df


def agregar_carga_financiera(df: pd.DataFrame) -> pd.DataFrame:
    """
    Historial de mora relativo al número de líneas de crédito.
    Fórmula: (mora_30 + mora_60 + mora_90) / max(num_lineas, 1)
    """
    df = df.copy()
    total_moras = (
        df['veces_mora_30_59_dias'] +
        df['veces_mora_60_89_dias'] +
        df['veces_mora_90_dias']
    )
    lineas = df['num_lineas_credito'].clip(lower=1)
    df['carga_financiera'] = total_moras / lineas
    return df


def agregar_segmento_edad(df: pd.DataFrame) -> pd.DataFrame:
    """
    Segmenta la edad en tres categorías estándar de la banca colombiana:
    Joven (18-30), Adulto (31-55), Senior (56+).
    """
    df = df.copy()
    df['segmento_edad'] = pd.cut(
        df['edad'],
        bins=[17, 30, 55, 200],
        labels=['Joven', 'Adulto', 'Senior']
    )
    return df


def agregar_riesgo_mora_acumulado(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score ponderado de mora histórica que penaliza moras más graves:
    Fórmula: veces_mora_90d×3 + veces_mora_60d×2 + veces_mora_30d×1
    """
    df = df.copy()
    df['riesgo_mora_acumulado'] = (
        df['veces_mora_90_dias'] * 3 +
        df['veces_mora_60_89_dias'] * 2 +
        df['veces_mora_30_59_dias'] * 1
    )
    return df


def agregar_tasa_dtf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade DTF histórica como variable de contexto macroeconómico.
    Valor: promedio 2005-2010 del Banco de la República (período del dataset).
    """
    df = df.copy()
    df['tasa_dtf_vigente'] = DTF_HISTORICA
    return df


def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline completo de feature engineering.
    Recibe DataFrame con columnas en español (ya renombradas).
    Retorna DataFrame con 6 features colombianas adicionales.
    """
    df = agregar_estrato_simulado(df)
    df = agregar_capacidad_pago(df)
    df = agregar_carga_financiera(df)
    df = agregar_segmento_edad(df)
    df = agregar_riesgo_mora_acumulado(df)
    df = agregar_tasa_dtf(df)
    return df
