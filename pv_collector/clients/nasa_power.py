"""
Cliente para NASA POWER — irradiancia histórica.

Documentación: https://power.larc.nasa.gov/docs/services/api/
Datos abiertos, sin necesidad de API key.
"""

import logging
import pandas as pd
import requests

from .. import config
from ..retry_utils import api_retry

logger = logging.getLogger("pv_collector.nasa_power")


@api_retry
def _fetch_raw(start_date: str, end_date: str) -> dict:
    """
    Llama al endpoint horario de NASA POWER.
    Nota: NASA POWER espera las fechas en formato YYYYMMDD, sin guiones.
    """
    params = {
        "parameters": ",".join(config.NASA_POWER_PARAMETERS),
        "community": config.NASA_POWER_COMMUNITY,
        "longitude": config.LONGITUDE,
        "latitude": config.LATITUDE,
        "start": start_date.replace("-", ""),
        "end": end_date.replace("-", ""),
        "format": "JSON",
    }
    response = requests.get(config.NASA_POWER_URL, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def fetch(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Descarga irradiancia histórica (GHI) de NASA POWER entre
    start_date y end_date ('YYYY-MM-DD').

    Nota: NASA POWER suele tener un desfase de varios días respecto
    al presente (no es tiempo real), así que este cliente se usa
    principalmente para relleno de huecos y validación, no para
    los datos más recientes.
    """
    logger.info(f"Descargando NASA POWER {start_date} -> {end_date}")
    raw = _fetch_raw(start_date, end_date)

    try:
        serie = raw["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
    except KeyError:
        logger.warning("NASA POWER no devolvió la serie esperada")
        return pd.DataFrame()

    rows = [
        {"timestamp": pd.to_datetime(ts, format="%Y%m%d%H", utc=True),
         "ghi_historico": valor}
        for ts, valor in serie.items()
        if valor != -999  # código de 'sin dato' de NASA POWER
    ]

    df = pd.DataFrame(rows)
    df["fuente_datos"] = "nasa_power"
    return df
