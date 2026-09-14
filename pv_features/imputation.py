"""
Imputación de valores faltantes.

Filosofía: solo se rellenan huecos CORTOS (hasta config.MAX_HUECO_INTERPOLABLE_HORAS)
mediante interpolación lineal, porque es razonable asumir que una
variable física como la irradiancia no cambia drásticamente en 1-2
horas. Los huecos más largos se dejan como NaN deliberadamente: no
tiene sentido "inventar" varias horas seguidas de un dato que no
tenemos, y el Agente 3 (modelo) o el propio pipeline de entrenamiento
deben decidir cómo tratar esas filas (excluirlas, u otra estrategia).
"""

import logging
import pandas as pd

from . import config

logger = logging.getLogger("pv_features.imputation")

# Columnas numéricas candidatas a interpolación de huecos cortos.
# Deliberadamente NO incluye 'generacion_real_diaria_mwh' ni
# 'demanda_total_diaria_mwh': son valores diarios repetidos, no una
# serie horaria real, así que interpolar ahí no tendría sentido físico.
_COLUMNAS_INTERPOLABLES = [
    "ghi", "dni", "dhi", "temp_ambiente", "nubosidad", "viento_velocidad",
    "ghi_historico", "generacion_teorica_mw",
]


def interpolar_huecos_cortos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpola linealmente huecos de hasta config.MAX_HUECO_INTERPOLABLE_HORAS
    horas de longitud en las columnas numéricas relevantes. Los huecos
    más largos se dejan sin tocar (permanecen NaN).
    """
    if df.empty:
        return df

    df = df.sort_values("timestamp").reset_index(drop=True).copy()

    for columna in _COLUMNAS_INTERPOLABLES:
        if columna not in df.columns:
            continue
        df[columna] = df[columna].interpolate(
            method="linear",
            limit=config.MAX_HUECO_INTERPOLABLE_HORAS,
            limit_area="inside",  # nunca extrapola en los extremos de la serie
        )

    return df
