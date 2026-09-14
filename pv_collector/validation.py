"""
Validación de calidad de datos antes de guardar en storage.

Aplica cuatro tipos de comprobación:
1. Huecos temporales (timestamps faltantes en la serie horaria)
2. Fuentes individuales faltantes (una fuente falló pero las demás no)
3. Valores fuera de rango físico razonable
4. Timestamps duplicados

Añade dos columnas:
- 'calidad_dato': 'ok', 'parcial', 'imputado', 'sospechoso'
- 'fuentes_faltantes': lista de fuentes sin dato en esa fila (vacía si ninguna falta)
"""

import logging
import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger("pv_collector.validation")

# Rangos físicos razonables por columna (min, max)
_RANGO_VALIDO = {
    "ghi": (0, 1400),
    "dni": (0, 1100),
    "dhi": (0, 800),
    "temp_ambiente": (-5, 45),
    "nubosidad": (0, 100),
    "viento_velocidad": (0, 60),
    "generacion_real_diaria_mwh": (0, 14400),  # 600 MW máx * 24h como cota superior
    "generacion_teorica_mw": (0, 600),
    "demanda_total_diaria_mwh": (0, 60000),  # 2500 MW máx * 24h como cota superior
}

# Qué columnas pertenecen a cada fuente, para poder detectar si una
# fuente concreta falta en una fila (aunque las demás sí tengan dato).
_COLUMNAS_POR_FUENTE = {
    "open_meteo": ["ghi", "dni", "dhi", "temp_ambiente", "nubosidad", "viento_velocidad"],
    "ree": ["generacion_real_diaria_mwh", "demanda_total_diaria_mwh"],
    "nasa_power": ["ghi_historico"],
    "pvgis": ["generacion_teorica_mw"],
}


def _marcar_fuera_de_rango(df: pd.DataFrame) -> pd.Series:
    """Devuelve una máscara booleana de filas con algún valor fuera de rango."""
    fuera_de_rango = pd.Series(False, index=df.index)
    for columna, (minimo, maximo) in _RANGO_VALIDO.items():
        if columna not in df.columns:
            continue
        fuera_de_rango |= (df[columna] < minimo) | (df[columna] > maximo)
    return fuera_de_rango


def _detectar_fuentes_faltantes(df: pd.DataFrame) -> pd.Series:
    """
    Para cada fila, devuelve la lista de fuentes cuyas columnas están
    TODAS a NaN en esa fila (es decir, esa fuente concreta no aportó
    dato), aunque otras fuentes sí lo hayan hecho.
    """
    faltantes_por_fila = [[] for _ in range(len(df))]

    for fuente, columnas in _COLUMNAS_POR_FUENTE.items():
        columnas_presentes = [c for c in columnas if c in df.columns]
        if not columnas_presentes:
            # La fuente ni siquiera aparece en el DataFrame combinado
            for lista in faltantes_por_fila:
                lista.append(fuente)
            continue

        fuente_ausente = df[columnas_presentes].isna().all(axis=1)
        for idx_pos, ausente in enumerate(fuente_ausente):
            if ausente:
                faltantes_por_fila[idx_pos].append(fuente)

    return pd.Series(faltantes_por_fila, index=df.index)


def _rellenar_huecos_temporales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reindexa a frecuencia horaria completa entre el min y el max
    timestamp, dejando NaN donde no había dato. Así los huecos
    quedan explícitos en vez de simplemente 'no existir'.
    """
    if df.empty:
        return df

    rango_completo = pd.date_range(
        start=df["timestamp"].min(),
        end=df["timestamp"].max(),
        freq="h",
        tz="UTC",
    )
    df = df.set_index("timestamp").reindex(rango_completo)
    df.index.name = "timestamp"
    return df.reset_index()


def validar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ejecuta la validación completa sobre un DataFrame ya combinado
    de las 4 fuentes y devuelve el mismo DataFrame con las columnas
    'calidad_dato' y 'fuentes_faltantes' añadidas.

    Prioridad de 'calidad_dato' (de más a menos grave):
    sospechoso > imputado (hueco temporal completo) > parcial (falta
    alguna fuente, no todas) > ok
    """
    if df.empty:
        logger.warning("DataFrame vacío recibido en validación")
        return df

    df = df.drop_duplicates(subset="timestamp", keep="first")
    df = _rellenar_huecos_temporales(df)

    era_nulo_total = df.drop(columns=["timestamp"]).isna().all(axis=1)
    fuentes_faltantes = _detectar_fuentes_faltantes(df)
    hay_alguna_faltante = fuentes_faltantes.apply(len) > 0
    fuera_de_rango = _marcar_fuera_de_rango(df)

    df["calidad_dato"] = "ok"
    df.loc[hay_alguna_faltante, "calidad_dato"] = "parcial"
    df.loc[era_nulo_total, "calidad_dato"] = "imputado"
    df.loc[fuera_de_rango, "calidad_dato"] = "sospechoso"
    df["fuentes_faltantes"] = fuentes_faltantes.apply(lambda lst: ",".join(lst))

    n_sospechosos = int(fuera_de_rango.sum())
    n_huecos = int(era_nulo_total.sum())
    n_parciales = int(hay_alguna_faltante.sum())
    if n_sospechosos or n_huecos or n_parciales:
        logger.warning(
            f"Validación: {n_huecos} huecos totales, {n_parciales} filas con "
            f"alguna fuente faltante, {n_sospechosos} valores fuera de rango"
        )

    df["fecha_ingesta"] = pd.Timestamp.utcnow()
    return df
