"""
Almacenamiento del dataset combinado en Parquet, particionado por fecha.

Estructura resultante:
    data/raw/year=2026/month=09/day=08/data.parquet
"""

import logging
import pandas as pd

from . import config

logger = logging.getLogger("pv_collector.storage")


def guardar_particionado(df: pd.DataFrame) -> list[str]:
    """
    Guarda el DataFrame en Parquet, particionando por año/mes/día
    según la columna 'timestamp'. Cada partición se guarda como un
    fichero independiente para poder reprocesar días sueltos sin
    tocar el resto del histórico.

    Devuelve la lista de rutas escritas.
    """
    if df.empty:
        logger.warning("Nada que guardar: DataFrame vacío")
        return []

    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["_fecha"] = df["timestamp"].dt.date

    rutas_escritas = []
    for fecha, grupo in df.groupby("_fecha"):
        carpeta = (
            config.RAW_DATA_DIR
            / f"year={fecha.year}"
            / f"month={fecha.month:02d}"
            / f"day={fecha.day:02d}"
        )
        carpeta.mkdir(parents=True, exist_ok=True)
        ruta = carpeta / "data.parquet"

        grupo = grupo.drop(columns=["_fecha"])
        grupo.to_parquet(ruta, index=False, engine="pyarrow")
        rutas_escritas.append(str(ruta))
        logger.info(f"Guardado: {ruta} ({len(grupo)} filas)")

    return rutas_escritas


def leer_rango(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Lee y concatena todas las particiones dentro del rango de fechas
    dado. Útil para que el Agente 2 (feature engineer) cargue el
    histórico sin conocer el detalle de la partición.
    """
    import glob

    patrones = str(config.RAW_DATA_DIR / "year=*" / "month=*" / "day=*" / "data.parquet")
    ficheros = sorted(glob.glob(patrones))

    if not ficheros:
        logger.warning("No hay ficheros Parquet en storage todavía")
        return pd.DataFrame()

    dfs = [pd.read_parquet(f) for f in ficheros]
    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    inicio = pd.Timestamp(start_date, tz="UTC")
    fin_inclusive = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(hours=23)
    mask = (df["timestamp"] >= inicio) & (df["timestamp"] <= fin_inclusive)
    return df.loc[mask].sort_values("timestamp").reset_index(drop=True)
