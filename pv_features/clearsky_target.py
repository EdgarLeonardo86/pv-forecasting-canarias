"""
Target de generación horaria estimada, vía índice de cielo despejado (kt).

Motivación (decisión de diseño acordada): no disponemos de generación FV
horaria real de ninguna fuente gratuita (REE solo da grano diario del
sistema completo de Canarias). En su lugar, construimos un proxy
realista combinando:

1. GHI de cielo despejado (envolvente EMPÍRICA, ver clearsky_envelope.py):
   percentil 97 del GHI real observado, por mes y hora del día.
2. GHI real observado (Open-Meteo): incluye el efecto real de las nubes.
3. kt = GHI_real / GHI_cielo_despejado: el índice de cielo despejado,
   ~1.0 en un día despejado, más bajo cuanto más nublado.
4. target = generacion_teorica_mw (PVGIS, curva "limpia") * kt

El resultado es una curva de generación que SÍ refleja el paso de
nubes reales, y por tanto puede contener ramp events genuinos que los
modelos del Agente 3 aprendan a predecir -- a diferencia de
'generacion_teorica_mw' sola, que es climatología sin variabilidad
diaria real.

HISTORIAL DE DISEÑO (por qué se usa una envolvente empírica y no un
modelo físico de pvlib): se probaron los modelos Ineichen y Haurwitz
de pvlib, y ambos dieron un sesgo sistemático demasiado grande para
Canarias -- 10-20% de las horas de sol con kt pegado al límite superior
(1.2), frente al 2-5% esperable por variabilidad meteorológica real
(cloud enhancement). Se intentó corregir con un desplazamiento de
timestamp (asumiendo que Open-Meteo promedia la hora completa mientras
pvlib da un valor instantáneo), pero el ajuste empeoró el resultado en
vez de mejorarlo, señal de que la causa no era solo un desfase temporal
simple. Sin datos de un pirómetro real en Gran Canaria para calibrar la
turbidez atmosférica correctamente, se optó por un enfoque empírico que
se autocalibra con los propios datos históricos, evitando toda esta
incertidumbre de modelado físico.
"""

import logging
import numpy as np
import pandas as pd

from . import config, clearsky_envelope

logger = logging.getLogger("pv_features.clearsky_target")


def calcular_target_generacion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade al DataFrame las columnas:
    - ghi_cielo_despejado: envolvente empírica (percentil 97 por mes/hora)
    - indice_cielo_despejado: kt = ghi / ghi_cielo_despejado, recortado
      al rango [config.KT_MINIMO, config.KT_MAXIMO]
    - generacion_horaria_estimada_mw: target final = generacion_teorica_mw * kt

    Requiere que la envolvente ya esté construida (ver
    clearsky_envelope.construir_y_guardar) y que existan las columnas
    'ghi' (Open-Meteo) y 'generacion_teorica_mw' (PVGIS).
    """
    if df.empty:
        return df

    columnas_necesarias = {"ghi", "generacion_teorica_mw"}
    if not columnas_necesarias.issubset(df.columns):
        logger.warning("Faltan columnas para calcular el target de generación")
        df = df.copy()
        df["ghi_cielo_despejado"] = np.nan
        df["indice_cielo_despejado"] = np.nan
        df["generacion_horaria_estimada_mw"] = np.nan
        return df

    df = clearsky_envelope.aplicar(df)

    with np.errstate(divide="ignore", invalid="ignore"):
        kt = df["ghi"] / df["ghi_cielo_despejado"]

    kt = kt.replace([np.inf, -np.inf], np.nan)
    kt = kt.clip(lower=config.KT_MINIMO, upper=config.KT_MAXIMO)
    kt = kt.fillna(0.0)  # de noche, o si la envolvente es 0 -> kt=0 (generación 0 igualmente)
    df["indice_cielo_despejado"] = kt

    df["generacion_horaria_estimada_mw"] = df["generacion_teorica_mw"] * kt

    return df
