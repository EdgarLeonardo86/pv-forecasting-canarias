"""
Cliente para REE (Red Eléctrica de España) — generación real y demanda.

Usa la API pública apidatos.ree.es.

IMPORTANTE (descubierto al probar contra la API real):
- El widget 'estructura-generacion' solo admite granularidad diaria,
  mensual o anual (time_trunc). NO admite 'hour'. Por tanto, la
  generación FV "real" que devuelve este cliente es un total diario,
  no una medición horaria.
- El endpoint de demanda 'demanda-tiempo-real' que se había asumido
  inicialmente no existe como widget real de la API; el widget
  correcto es 'evolucion' dentro de la categoría 'demanda'.

Cómo se usa en el pipeline (decisión de diseño):
Como el resto del sistema trabaja a grano horario, este cliente
expande el valor diario a las 24 horas de ese día (repitiendo el
mismo valor) y lo expone como columna de REFERENCIA/VALIDACIÓN
diaria, no como generación horaria real. El Agente 4 (validador)
la usará para contrastar el total diario de las demás fuentes con
el dato oficial de REE, y el Agente 2 puede usarla como contexto,
pero la generación horaria "real" para entrenar los modelos debe
venir de PVGIS o de un proxy calculado con pvlib a partir del GHI
de Open-Meteo.
"""

import logging
import pandas as pd
import requests

from .. import config
from ..retry_utils import api_retry

logger = logging.getLogger("pv_collector.ree")


@api_retry
def _fetch_raw(endpoint: str, start_date: str, end_date: str) -> dict:
    """Llama a un endpoint de REE y devuelve el JSON crudo."""
    url = f"{config.REE_BASE_URL}{endpoint}"
    params = {
        "start_date": f"{start_date}T00:00",
        "end_date": f"{end_date}T23:59",
        "time_trunc": config.REE_TIME_TRUNC,
        "geo_trunc": config.REE_GEO_TRUNC,
        "geo_limit": config.REE_GEO_LIMIT,
        "geo_ids": config.REE_GEO_IDS,
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    response = requests.get(url, params=params, headers=headers, timeout=15)
    response.raise_for_status()
    return response.json()


def _extract_daily_series(raw: dict, tecnologia_buscada: tuple, columna: str) -> pd.DataFrame:
    """
    Extrae una serie diaria de la respuesta de REE buscando por nombre
    de tecnología (case-insensitive, coincidencia parcial).

    tecnologia_buscada: tupla de substrings válidos, p.ej. ('fotovoltaica',)
    """
    rows = []
    for bloque in raw.get("included", []):
        nombre = bloque.get("type", "").lower()
        if not any(t in nombre for t in tecnologia_buscada):
            continue
        for punto in bloque.get("attributes", {}).get("values", []):
            rows.append({
                "fecha": pd.to_datetime(punto["datetime"]).date(),
                columna: punto["value"],
            })
    return pd.DataFrame(rows)


def _expandir_a_horario(df_diario: pd.DataFrame, columna: str) -> pd.DataFrame:
    """
    Expande un DataFrame con una fila por día a 24 filas (una por hora),
    repitiendo el valor diario. Así se puede unir por 'timestamp' con
    el resto de fuentes horarias sin perder la información diaria.
    """
    if df_diario.empty:
        return pd.DataFrame(columns=["timestamp", columna])

    filas = []
    for _, fila in df_diario.iterrows():
        for hora in range(24):
            filas.append({
                "timestamp": pd.Timestamp(fila["fecha"], tz="UTC") + pd.Timedelta(hours=hora),
                columna: fila[columna],
            })
    return pd.DataFrame(filas)


def fetch(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Descarga generación FV diaria y demanda diaria de Canarias entre
    start_date y end_date ('YYYY-MM-DD'), y las expande a resolución
    horaria (mismo valor repetido en las 24 horas del día) para poder
    combinarlas con el resto del pipeline.

    Columnas devueltas:
    - generacion_real_diaria_mwh: total diario de generación FV (referencia)
    - demanda_total_diaria_mwh: total diario de demanda (referencia)

    Nota: estos NO son valores horarios reales, son el dato diario
    repartido. Úsalos como contexto/validación, no como target de
    entrenamiento horario.
    """
    logger.info(f"Descargando REE (grano diario) {start_date} -> {end_date}")

    raw_gen = _fetch_raw(config.REE_GENERATION_ENDPOINT, start_date, end_date)
    raw_dem = _fetch_raw(config.REE_DEMAND_ENDPOINT, start_date, end_date)

    df_gen_diario = _extract_daily_series(raw_gen, ("fotovoltaica",), "generacion_real_diaria_mwh")
    df_dem_diario = _extract_daily_series(raw_dem, ("demanda",), "demanda_total_diaria_mwh")

    if df_gen_diario.empty and df_dem_diario.empty:
        logger.warning("REE devolvió generación y demanda vacías")
        return pd.DataFrame()

    df_gen_horario = _expandir_a_horario(df_gen_diario, "generacion_real_diaria_mwh")
    df_dem_horario = _expandir_a_horario(df_dem_diario, "demanda_total_diaria_mwh")

    if df_gen_horario.empty:
        df = df_dem_horario
    elif df_dem_horario.empty:
        df = df_gen_horario
    else:
        df = pd.merge(df_gen_horario, df_dem_horario, on="timestamp", how="outer")

    df["fuente_datos"] = "ree"
    return df
