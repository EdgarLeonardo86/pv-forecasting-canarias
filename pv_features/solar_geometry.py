"""
Features de geometría solar, calculadas con pvlib a partir de la
ubicación y el timestamp. No dependen de ninguna API externa: son
puramente astronómicas/geométricas.
"""

import logging
import pandas as pd
import pvlib

from . import config

logger = logging.getLogger("pv_features.solar_geometry")


def calcular_geometria_solar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade al DataFrame las columnas:
    - angulo_cenital_solar: 0° = sol en el zénit, 90° = sol en el horizonte
    - azimut_solar: dirección del sol en grados (0°=N, 90°=E, 180°=S, 270°=O)
    - elevacion_solar: complementario del cenital (90 - cenital)
    - es_de_dia: booleano, True si el sol está sobre el horizonte

    Requiere que 'timestamp' sea tz-aware (UTC), tal como lo entrega
    el Agente 1.
    """
    if df.empty:
        return df

    posicion_solar = pvlib.solarposition.get_solarposition(
        time=df["timestamp"],
        latitude=config.LATITUDE,
        longitude=config.LONGITUDE,
        altitude=config.ELEVATION_M,
    )

    df = df.copy()
    df["angulo_cenital_solar"] = posicion_solar["zenith"].values
    df["azimut_solar"] = posicion_solar["azimuth"].values
    df["elevacion_solar"] = posicion_solar["elevation"].values
    df["es_de_dia"] = df["elevacion_solar"] > 0

    return df
