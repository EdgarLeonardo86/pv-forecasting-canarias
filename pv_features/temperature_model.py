"""
Modelo térmico del panel fotovoltaico (modelo NOCT simplificado).

La temperatura de la célula/panel afecta directamente al rendimiento
de un sistema FV (a más temperatura, menor eficiencia), así que es
una feature relevante para el modelo de ML, no solo la irradiancia.

Fórmula NOCT estándar:
    T_panel = T_ambiente + (NOCT - T_ambiente_ref) / Irradiancia_ref * GHI

Es una aproximación razonable sin necesitar datos de viento detallados,
aunque pvlib ofrece modelos más precisos (ej. modelo Sandia/PVsyst) si
más adelante se dispone de la ficha técnica exacta del panel real.
"""

import pandas as pd

from . import config


def calcular_temp_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade la columna 'temp_panel' (°C) usando el modelo NOCT simplificado,
    a partir de 'temp_ambiente' y 'ghi' (ambas de Open-Meteo).

    Si faltan las columnas necesarias, la función no falla: deja
    'temp_panel' como NaN y registra un aviso, para que el resto del
    pipeline pueda seguir adelante con huecos explícitos.
    """
    df = df.copy()

    if "temp_ambiente" not in df.columns or "ghi" not in df.columns:
        df["temp_panel"] = pd.NA
        return df

    factor = (config.PANEL_NOCT_C - config.PANEL_NOCT_TEMP_AMBIENTE_REF) / config.PANEL_NOCT_IRRADIANCIA_REF
    df["temp_panel"] = df["temp_ambiente"] + factor * df["ghi"]

    return df
