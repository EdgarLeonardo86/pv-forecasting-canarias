"""
Preparación del dataset supervisado para el Agente 3.

Convierte el dataset de features (una fila por hora) en un problema de
predicción directa a horizonte h: para cada hora t, el target es el
valor de generacion_horaria_estimada_mw en t+h, y las features son
todas las columnas conocidas en el momento t (nunca de t+h -- ver
config.COLUMNAS_EXCLUIDAS para las que sí filtrarían futuro).
"""

import logging
import pandas as pd

from . import config

logger = logging.getLogger("pv_models.dataset_prep")


def cargar_dataset(ruta_parquet: str) -> pd.DataFrame:
    """Carga el parquet de features generado por el Agente 2."""
    df = pd.read_parquet(ruta_parquet)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def construir_target_horizonte(df: pd.DataFrame, horas: int) -> pd.Series:
    """
    Devuelve una Serie con el target desplazado -horas posiciones:
    target_horizonte[t] = generacion_horaria_estimada_mw[t + horas]

    Las últimas 'horas' filas quedan en NaN (no existe futuro para
    ellas todavía) y se descartan más adelante en el split.
    """
    return df[config.TARGET_COL].shift(-horas)


def construir_columna_futura(df: pd.DataFrame, columna: str, horas: int) -> pd.Series:
    """
    Versión genérica de construir_target_horizonte para cualquier
    columna: útil para evaluar el modelo correctamente contra el
    estado del MOMENTO OBJETIVO (t+horas), no del momento de emisión
    de la predicción (t). Ejemplo: para saber si la hora que se está
    prediciendo es de día o de noche, hay que mirar 'es_de_dia' en
    t+horas, no en t -- con horizontes como 12h o 36h, una predicción
    emitida de día puede apuntar a una hora nocturna.
    """
    return df[columna].shift(-horas)


def seleccionar_columnas_features(df: pd.DataFrame) -> list:
    """
    Devuelve la lista de columnas a usar como features de entrada,
    excluyendo las que filtran información del futuro o son metadatos
    (ver config.COLUMNAS_EXCLUIDAS).
    """
    return [c for c in df.columns if c not in config.COLUMNAS_EXCLUIDAS]


def preparar_para_horizonte(df: pd.DataFrame, horas: int) -> pd.DataFrame:
    """
    Construye el dataset supervisado completo para un horizonte dado:
    añade la columna 'target' (desplazada) y elimina filas que no
    puedan usarse (target sin futuro disponible, o features con NaN
    por lags al principio de la serie).
    """
    # CRÍTICO: seleccionar las columnas de features ANTES de añadir la
    # columna 'target' al DataFrame. Si se hiciera al revés, 'target'
    # se colaría en la lista de features (no está en COLUMNAS_EXCLUIDAS
    # porque no existe todavía en ese momento) y el modelo vería
    # literalmente la respuesta que tiene que predecir como una
    # entrada más -- leakage total, detectado al revisar feature
    # importances y ver 'target' con un peso alto.
    columnas_features = seleccionar_columnas_features(df)

    df = df.copy()
    df["target"] = construir_target_horizonte(df, horas)

    columnas_relevantes = columnas_features + ["target", "timestamp"]

    # 'es_de_dia' desplazada al MOMENTO OBJETIVO (t+horas), no al de
    # emisión (t) -- necesaria para evaluar el modelo correctamente
    # por horas de día/noche (ver docstring de construir_columna_futura).
    if "es_de_dia" in df.columns:
        df["es_de_dia_objetivo"] = construir_columna_futura(df, "es_de_dia", horas)
        columnas_relevantes = columnas_relevantes + ["es_de_dia_objetivo"]

    filas_antes = len(df)
    df = df.dropna(subset=columnas_relevantes)
    filas_descartadas = filas_antes - len(df)

    logger.info(
        f"Horizonte {horas}h: {len(df)} filas utilizables "
        f"({filas_descartadas} descartadas por NaN en target/features)"
    )

    return df


def split_temporal(df: pd.DataFrame, fraccion_train: float = None) -> tuple:
    """
    Divide el dataset en train/test de forma CRONOLÓGICA (las primeras
    filas en el tiempo van a train, las últimas a test) -- nunca de
    forma aleatoria, porque mezclar al azar en una serie temporal deja
    que el modelo entrene con información "adyacente" al futuro que
    luego se evalúa, inflando artificialmente el rendimiento aparente.
    """
    fraccion_train = fraccion_train or config.FRACCION_TRAIN
    df = df.sort_values("timestamp").reset_index(drop=True)

    corte = int(len(df) * fraccion_train)
    train = df.iloc[:corte]
    test = df.iloc[corte:]

    return train, test
