"""
Cliente para PVGIS — producción fotovoltaica teórica.

Documentación: https://joint-research-centre.ec.europa.eu/pvgis-photovoltaic-geographical-information-system_en
Servicio público del JRC (Comisión Europea), sin API key.

IMPORTANTE (descubierto al probar contra la API real):
PVGIS solo admite años entre 2005 y 2020 en este endpoint (la base
de radiación satelital que usa no tiene cobertura más reciente para
esta ubicación). Como el valor que buscamos es una producción TEÓRICA
(una climatología de referencia, no una medición del año en curso),
la solución es fijar un año de referencia dentro del rango permitido
(PVGIS_REFERENCE_YEAR) y reproyectar el calendario devuelto (mismo
mes/día/hora) al año que realmente se está pidiendo. Esto es una
aproximación razonable para un valor teórico/baseline, pero conviene
tenerlo presente: no es la irradiancia real del año solicitado.
"""

import logging
import pandas as pd
import requests

from .. import config
from ..retry_utils import api_retry

logger = logging.getLogger("pv_collector.pvgis")


@api_retry
def _fetch_raw(year: int) -> dict:
    """
    Llama al endpoint 'seriescalc' de PVGIS.

    Nota: PVGIS trabaja con años completos de la base histórica,
    no con rangos de fecha arbitrarios como las otras APIs. Por eso
    este cliente recibe un año y se filtra después al rango deseado.
    """
    params = {
        "lat": config.LATITUDE,
        "lon": config.LONGITUDE,
        "startyear": year,
        "endyear": year,
        "pvcalculation": 1,
        "peakpower": config.PVGIS_PEAKPOWER_KW,
        "loss": config.PVGIS_LOSS_PCT,
        "outputformat": "json",
    }
    response = requests.get(config.PVGIS_URL, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def fetch(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Descarga producción FV teórica de PVGIS usando el año de referencia
    permitido (config.PVGIS_REFERENCE_YEAR) y reproyecta el calendario
    (mismo mes/día/hora) al rango [start_date, end_date] realmente
    solicitado ('YYYY-MM-DD').

    La potencia teórica se escala fuera de este cliente (en el
    Agente 2) según la potencia pico real de la instalación de
    referencia; aquí se devuelve normalizada a PVGIS_PEAKPOWER_KW.
    """
    año_solicitado = pd.to_datetime(start_date).year
    año_referencia = config.PVGIS_REFERENCE_YEAR

    logger.info(
        f"Descargando PVGIS (año de referencia {año_referencia}, "
        f"reproyectado a {año_solicitado})"
    )
    raw = _fetch_raw(año_referencia)

    try:
        registros = raw["outputs"]["hourly"]
    except KeyError:
        logger.warning("PVGIS no devolvió la serie horaria esperada")
        return pd.DataFrame()

    df = pd.DataFrame(registros)
    df["timestamp_referencia"] = pd.to_datetime(df["time"], format="%Y%m%d:%H%M", utc=True)
    # PVGIS marca el punto medio de la hora (ej. 00:09); lo redondeamos
    # a la hora en punto para que encaje con Open-Meteo/NASA POWER al
    # combinar por 'timestamp' en el colector.
    df["timestamp_referencia"] = df["timestamp_referencia"].dt.floor("h")
    df = df.rename(columns={"P": "generacion_teorica_w"})
    df["generacion_teorica_mw"] = df["generacion_teorica_w"] / 1_000_000

    # Reproyectar el año de referencia al año realmente solicitado,
    # conservando mes/día/hora. El 29 de febrero se descarta si el
    # año solicitado no es bisiesto.
    try:
        df["timestamp"] = df["timestamp_referencia"].apply(
            lambda ts: ts.replace(year=año_solicitado)
        )
    except ValueError:
        # 29 de febrero en año de referencia bisiesto -> año solicitado no bisiesto
        mask_valida = ~((df["timestamp_referencia"].dt.month == 2) & (df["timestamp_referencia"].dt.day == 29))
        df = df.loc[mask_valida].copy()
        df["timestamp"] = df["timestamp_referencia"].apply(
            lambda ts: ts.replace(year=año_solicitado)
        )

    inicio = pd.Timestamp(start_date, tz="UTC")
    fin_inclusive = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(hours=23)
    mask_rango = (df["timestamp"] >= inicio) & (df["timestamp"] <= fin_inclusive)
    df = df.loc[mask_rango, ["timestamp", "generacion_teorica_mw"]].reset_index(drop=True)
    df["fuente_datos"] = "pvgis"

    return df
