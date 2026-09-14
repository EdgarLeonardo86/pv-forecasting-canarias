"""
Cliente para Open-Meteo — meteo horaria.

Documentación: https://open-meteo.com/en/docs
No requiere API key para uso no comercial.

IMPORTANTE (descubierto al probar contra la API real):
El endpoint /v1/forecast solo cubre pronóstico y unos pocos días de
histórico reciente. Para fechas pasadas (por ejemplo, al recoger
histórico para entrenar los modelos) hay que usar el endpoint de
archivo /v1/archive, que sirve datos de reanálisis (ERA5) con las
mismas variables. Este cliente elige automáticamente el endpoint
correcto según lo antiguo que sea end_date.
"""

import logging
from datetime import date, timedelta
import pandas as pd
import requests

from .. import config
from ..retry_utils import api_retry

logger = logging.getLogger("pv_collector.open_meteo")

# Mapeo de nombres de la API a nuestro esquema interno
_COLUMN_MAP = {
    "shortwave_radiation": "ghi",
    "direct_normal_irradiance": "dni",
    "diffuse_radiation": "dhi",
    "temperature_2m": "temp_ambiente",
    "cloud_cover": "nubosidad",
    "wind_speed_10m": "viento_velocidad",
}

# Margen de seguridad: si end_date es más antiguo que esto, usamos el
# endpoint de archivo en vez del de forecast.
_DIAS_MARGEN_FORECAST = 5


def _elegir_url(end_date: str) -> str:
    """Decide qué endpoint usar según lo antigua que sea la fecha final."""
    fecha_fin = pd.to_datetime(end_date).date()
    limite = date.today() - timedelta(days=_DIAS_MARGEN_FORECAST)
    if fecha_fin < limite:
        return config.OPEN_METEO_ARCHIVE_URL
    return config.OPEN_METEO_URL


@api_retry
def _fetch_raw(start_date: str, end_date: str) -> dict:
    """Llama al endpoint de Open-Meteo (forecast o archive) y devuelve el JSON crudo."""
    url = _elegir_url(end_date)
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "hourly": ",".join(config.OPEN_METEO_HOURLY_VARS),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
    }
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    return response.json()


def fetch(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Descarga meteo horaria de Open-Meteo entre start_date y end_date
    (formato 'YYYY-MM-DD') y devuelve un DataFrame ya normalizado
    al esquema interno. Usa automáticamente el endpoint de archivo
    histórico si la fecha es antigua, o el de forecast si es reciente.
    """
    logger.info(f"Descargando Open-Meteo {start_date} -> {end_date}")
    raw = _fetch_raw(start_date, end_date)

    hourly = raw.get("hourly", {})
    df = pd.DataFrame(hourly)

    if df.empty:
        logger.warning("Open-Meteo devolvió un dataframe vacío")
        return df

    df = df.rename(columns=_COLUMN_MAP)
    df["timestamp"] = pd.to_datetime(df["time"], utc=True)
    df = df.drop(columns=["time"])
    df["fuente_datos"] = "open_meteo"

    return df
