"""
Features de lag (valores pasados) y medias móviles.

Los ramp events (caídas bruscas de generación) y la dinámica temporal
en general se detectan mejor si el modelo puede ver el valor de hace
1 hora, 1 día y 1 semana, no solo el valor actual.

IMPORTANTE: estos lags requieren que el DataFrame venga ORDENADO por
timestamp y con frecuencia horaria continua (sin huecos), porque se
calculan por desplazamiento de fila (shift), no por diferencia real
de tiempo. Si hay huecos en la serie, hay que rellenarlos antes
(ver imputation.py) o los lags quedarán desalineados.
"""

import logging
import pandas as pd

from . import config

logger = logging.getLogger("pv_features.lag_features")

# Columnas sobre las que tiene sentido calcular lags para el forecasting.
# Se incluye el propio target (generacion_horaria_estimada_mw): los
# modelos de forecasting necesitan ver su valor pasado, no solo el de
# las variables explicativas.
_COLUMNAS_PARA_LAG = ["generacion_teorica_mw", "ghi", "generacion_horaria_estimada_mw"]


def calcular_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade columnas de lag para cada combinación de columna en
    _COLUMNAS_PARA_LAG y hora en config.LAGS_HORAS.

    Ejemplo de columnas generadas: 'generacion_teorica_mw_lag_1h',
    'generacion_teorica_mw_lag_24h', 'ghi_lag_168h', etc.
    """
    if df.empty:
        return df

    df = df.sort_values("timestamp").reset_index(drop=True).copy()

    for columna in _COLUMNAS_PARA_LAG:
        if columna not in df.columns:
            logger.warning(f"Columna '{columna}' no encontrada, se omiten sus lags")
            continue
        for horas in config.LAGS_HORAS:
            df[f"{columna}_lag_{horas}h"] = df[columna].shift(horas)

    return df


def calcular_media_movil_nubosidad(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade 'nubosidad_media_movil' como media móvil de las últimas
    config.VENTANA_NUBOSIDAD_HORAS horas. Suaviza el ruido puntual de
    la variable de nubosidad, que puede ser muy volátil hora a hora.
    """
    if df.empty or "nubosidad" not in df.columns:
        return df

    df = df.sort_values("timestamp").reset_index(drop=True).copy()
    df["nubosidad_media_movil"] = (
        df["nubosidad"]
        .rolling(window=config.VENTANA_NUBOSIDAD_HORAS, min_periods=1)
        .mean()
    )
    return df
