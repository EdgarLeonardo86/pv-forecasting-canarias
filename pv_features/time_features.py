"""
Features cíclicas de tiempo.

La hora del día y el día del año son variables circulares (23:00 está
tan cerca de 00:00 como de 22:00), así que se codifican con seno/coseno
en vez de como enteros, para que el modelo de ML entienda esa
continuidad circular.
"""

import numpy as np
import pandas as pd


def calcular_features_tiempo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade al DataFrame las columnas:
    - hora_sin, hora_cos: hora del día (0-23) codificada cíclicamente
    - dia_anio_sin, dia_anio_cos: día del año (1-366) codificado cíclicamente
    - mes: mes calendario (1-12), útil para detectar estacionalidad simple
    """
    if df.empty:
        return df

    df = df.copy()
    hora = df["timestamp"].dt.hour
    dia_anio = df["timestamp"].dt.dayofyear

    df["hora_sin"] = np.sin(2 * np.pi * hora / 24)
    df["hora_cos"] = np.cos(2 * np.pi * hora / 24)
    df["dia_anio_sin"] = np.sin(2 * np.pi * dia_anio / 365.25)
    df["dia_anio_cos"] = np.cos(2 * np.pi * dia_anio / 365.25)
    df["mes"] = df["timestamp"].dt.month

    return df
