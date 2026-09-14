"""
Cálculo de métricas de error para el Agente 4.

MAPE se calcula solo sobre filas con target > UMBRAL_TARGET_MAPE
(horas de día, esencialmente): dividir por un target cercano a 0
(horas de noche) produce porcentajes de error absurdos y no
informativos, aunque el error absoluto en esas horas sea minúsculo.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from . import config


def calcular_metricas(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Devuelve un diccionario con MAE, RMSE (sobre todas las filas) y
    MAPE (solo sobre filas de día, ver docstring del módulo).
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    mascara_dia = y_true > config.UMBRAL_TARGET_MAPE
    if mascara_dia.sum() > 0:
        mape = np.mean(
            np.abs((y_true[mascara_dia] - y_pred[mascara_dia]) / y_true[mascara_dia])
        ) * 100
    else:
        mape = None

    return {"mae": mae, "rmse": rmse, "mape": mape, "n_filas": len(y_true), "n_filas_dia": int(mascara_dia.sum())}


def metricas_por_hora(df_resultado, columna_real: str, columna_pred: str, columna_hora: str) -> dict:
    """
    Agrupa las métricas por hora del día (0-23), útil para detectar si
    el modelo falla más en franjas concretas (por ejemplo, amanecer o
    atardecer, donde la irradiancia cambia más rápido).

    df_resultado debe tener las columnas indicadas, con una fila por
    predicción evaluada.
    """
    resultado = {}
    for hora, grupo in df_resultado.groupby(columna_hora):
        y_true = grupo[columna_real].values
        y_pred = grupo[columna_pred].values
        resultado[int(hora)] = calcular_metricas(y_true, y_pred)
    return resultado
