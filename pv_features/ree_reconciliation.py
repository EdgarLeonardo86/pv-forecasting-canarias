"""
Reconciliación con REE: factor de corrección diario.

Como decidimos en el Agente 1, REE solo da generación real a nivel
DIARIO para todo el sistema eléctrico de Canarias, mientras que PVGIS
da una curva teórica horaria normalizada a 1 kWp de referencia. No son
comparables en magnitud absoluta (Canarias tiene cientos de MW
instalados, nuestra referencia es 1 kWp), pero SÍ es útil comparar su
FORMA relativa día a día:

    factor_correccion_diario = generacion_real_diaria_mwh / suma_diaria(generacion_teorica_mw)

Este factor captura, de forma aproximada, cuánto se desvió la
generación real del sistema respecto a lo que el modelo teórico
esperaría para ese día (por ejemplo, un día muy nuboso tendrá un
factor más bajo que un día despejado). Se expone como feature
adicional, no como una reescala física exacta.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("pv_features.ree_reconciliation")


def calcular_factor_correccion_diario(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade la columna 'factor_correccion_diario', repetida en todas las
    horas del mismo día, calculada como el ratio entre el total diario
    real de REE y la suma diaria de la generación teórica de PVGIS.

    Si falta alguna de las dos columnas necesarias, o la suma teórica
    del día es 0 (por ejemplo, un día sin datos de PVGIS), el factor
    queda como NaN para ese día en vez de forzar una división por cero.
    """
    if df.empty:
        return df

    columnas_necesarias = {"generacion_real_diaria_mwh", "generacion_teorica_mw"}
    if not columnas_necesarias.issubset(df.columns):
        logger.warning("Faltan columnas para calcular el factor de corrección diario")
        df = df.copy()
        df["factor_correccion_diario"] = np.nan
        return df

    df = df.copy()
    df["_fecha"] = df["timestamp"].dt.date

    suma_teorica_diaria = df.groupby("_fecha")["generacion_teorica_mw"].transform("sum")
    real_diario = df["generacion_real_diaria_mwh"]  # ya viene repetido por hora desde el Agente 1

    with np.errstate(divide="ignore", invalid="ignore"):
        factor = real_diario / suma_teorica_diaria

    factor = factor.replace([np.inf, -np.inf], np.nan)
    df["factor_correccion_diario"] = factor
    # Versión log-transformada para los modelos de ML: la magnitud
    # absoluta del factor es enorme (compara MWh de todo el sistema
    # de Canarias vs una referencia de 1 kWp), lo que puede desequilibrar
    # el entrenamiento -- especialmente en Prophet, que trata sus
    # regresores externos de forma más lineal/aditiva que los árboles
    # de XGBoost/LightGBM. log1p() comprime la escala sin depender de
    # estadísticas del set de entrenamiento (a diferencia de z-score).
    df["factor_correccion_diario_log"] = np.log1p(factor)
    df = df.drop(columns=["_fecha"])

    return df
