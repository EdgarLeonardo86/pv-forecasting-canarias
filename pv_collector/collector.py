"""
Agente 1 — Colector.

Orquesta las cuatro fuentes de datos (Open-Meteo, REE, NASA POWER,
PVGIS), las combina en un único DataFrame según el esquema acordado,
las valida y las guarda en storage particionado.

Este es el único módulo que las capas superiores (Agente 2 en
adelante) necesitan conocer: `ejecutar_coleccion(...)`.
"""

import logging
import pandas as pd

from .clients import open_meteo, ree, nasa_power, pvgis
from . import validation, storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("pv_collector.collector")

# Cada fuente puede fallar de forma independiente sin tumbar las demás
_FUENTES = {
    "open_meteo": open_meteo.fetch,
    "ree": ree.fetch,
    "nasa_power": nasa_power.fetch,
    "pvgis": pvgis.fetch,
}


def _combinar_fuentes(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Llama a cada cliente de forma independiente (si uno falla, se
    registra el error y se continúa con las demás fuentes en vez de
    abortar toda la colección) y las une por 'timestamp'.
    """
    dataframes = []

    for nombre, funcion_fetch in _FUENTES.items():
        try:
            df_fuente = funcion_fetch(start_date, end_date)
            if not df_fuente.empty:
                dataframes.append(df_fuente)
            else:
                logger.warning(f"Fuente '{nombre}' devolvió datos vacíos")
        except Exception as error:
            # No relanzamos: una fuente caída no debe tumbar las demás
            logger.error(f"Fallo al recoger '{nombre}': {error}", exc_info=True)

    if not dataframes:
        logger.error("Ninguna fuente devolvió datos. Colección abortada.")
        return pd.DataFrame()

    df_final = dataframes[0]
    for df_extra in dataframes[1:]:
        # Cada fuente puede traer su propia columna 'fuente_datos';
        # las combinamos en una sola columna de texto separada por coma
        df_final = pd.merge(
            df_final, df_extra, on="timestamp", how="outer", suffixes=("", "_dup")
        )
        if "fuente_datos_dup" in df_final.columns:
            df_final["fuente_datos"] = (
                df_final["fuente_datos"].fillna("") + ","
                + df_final["fuente_datos_dup"].fillna("")
            ).str.strip(",")
            df_final = df_final.drop(columns=["fuente_datos_dup"])

    return df_final.sort_values("timestamp").reset_index(drop=True)


def ejecutar_coleccion(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Punto de entrada único del Agente 1.

    1. Descarga las cuatro fuentes para el rango [start_date, end_date]
    2. Combina en un único DataFrame por timestamp
    3. Valida (huecos, rangos, duplicados)
    4. Guarda en Parquet particionado

    Devuelve el DataFrame final (ya validado) para inspección o
    encadenado directo con el Agente 2 si se desea.
    """
    logger.info(f"=== Iniciando colección {start_date} -> {end_date} ===")

    df_combinado = _combinar_fuentes(start_date, end_date)
    if df_combinado.empty:
        return df_combinado

    df_validado = validation.validar(df_combinado)
    rutas = storage.guardar_particionado(df_validado)

    logger.info(
        f"=== Colección completada: {len(df_validado)} filas, "
        f"{len(rutas)} particiones escritas ==="
    )
    return df_validado
