"""
Detección de ramp events (caídas bruscas de generación por nubes),
usando el índice de cielo despejado (kt) en vez de la generación en
bruto -- la generación cae de forma natural y esperada cada atardecer,
eso no es un ramp event; un ramp event real es una caída de kt, que
aísla el efecto de las nubes de la geometría solar.

Dos modos de uso:
1. detectar_historico(df): analiza datos YA OBSERVADOS para contar y
   listar los ramp events ocurridos en un periodo (útil para
   análisis/reporting, y para conectar con el TFB).
2. evaluar_prediccion_ramp(...): evalúa una predicción del Agente 3
   (modelo de horizonte 1h) para emitir una alerta ANTES de que ocurra
   el ramp event -- el caso de uso de "aviso temprano" real.
"""

import logging
import pandas as pd

from . import config

logger = logging.getLogger("pv_alertas.deteccion")


def _clasificar_confianza(magnitud_caida_kt: float) -> str:
    """
    Heurística simple de confianza, NO una probabilidad calibrada:
    cuanto más lejos esté la caída del umbral mínimo, más confianza se
    asigna. Es una forma honesta y simple de priorizar alertas, no un
    modelo estadístico de incertidumbre real.
    """
    ratio = magnitud_caida_kt / config.UMBRAL_CAIDA_KT
    if ratio >= config.UMBRAL_CONFIANZA_ALTA:
        return "alta"
    elif ratio >= config.UMBRAL_CONFIANZA_MEDIA:
        return "media"
    else:
        return "baja"


def detectar_historico(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analiza un dataset de features YA OBSERVADO (columnas
    'indice_cielo_despejado', 'es_de_dia', 'timestamp',
    'generacion_teorica_mw') y devuelve un DataFrame con un ramp event
    por fila detectada: timestamp, caída de kt, magnitud estimada en
    MW (referencia 1 kWp) y nivel de confianza heurístico.

    Un ramp event se define como: caída de kt >= UMBRAL_CAIDA_KT entre
    la hora anterior y la actual, estando ambas horas dentro del día
    (para no confundir con la caída natural del atardecer).
    """
    columnas_necesarias = {"indice_cielo_despejado", "es_de_dia", "timestamp", "generacion_teorica_mw"}
    if not columnas_necesarias.issubset(df.columns):
        raise ValueError(f"Faltan columnas necesarias: {columnas_necesarias - set(df.columns)}")

    df = df.sort_values("timestamp").reset_index(drop=True).copy()
    df["kt_anterior"] = df["indice_cielo_despejado"].shift(1)
    df["es_de_dia_anterior"] = df["es_de_dia"].shift(1)
    df["caida_kt"] = df["kt_anterior"] - df["indice_cielo_despejado"]

    mascara_ramp = (
        (df["caida_kt"] >= config.UMBRAL_CAIDA_KT)
        & (df["es_de_dia"] == True)
        & (df["es_de_dia_anterior"] == True)
    )

    eventos = df.loc[mascara_ramp, ["timestamp", "caida_kt", "generacion_teorica_mw"]].copy()
    eventos["magnitud_estimada_mw"] = eventos["caida_kt"] * eventos["generacion_teorica_mw"]
    eventos["confianza"] = eventos["caida_kt"].apply(_clasificar_confianza)
    eventos = eventos.rename(columns={"caida_kt": "caida_indice_cielo_despejado"})

    logger.info(f"Ramp events detectados en el histórico: {len(eventos)} de {len(df)} horas analizadas")

    return eventos.reset_index(drop=True)


def evaluar_prediccion_ramp(
    kt_actual: float,
    generacion_predicha_siguiente_hora: float,
    generacion_teorica_siguiente_hora: float,
    timestamp_siguiente_hora,
) -> dict | None:
    """
    Evalúa si la predicción del Agente 3 para la PRÓXIMA hora implica
    un ramp event respecto al estado actual, y devuelve un diccionario
    con la alerta si es así (o None si no hay alerta).

    Reconstruye el kt implícito en la predicción a partir de
    generacion_predicha_siguiente_hora / generacion_teorica_siguiente_hora
    (la curva teórica es determinista y conocida de antemano, así que
    no hace falta que el modelo la prediga -- solo su generación total,
    de la que se puede despejar el kt implícito).

    Si generacion_teorica_siguiente_hora es prácticamente 0 (noche), no
    tiene sentido evaluar kt y se devuelve None sin alerta.
    """
    if generacion_teorica_siguiente_hora <= 1e-9:
        return None  # próxima hora es de noche, no aplica

    kt_predicho_siguiente = generacion_predicha_siguiente_hora / generacion_teorica_siguiente_hora
    caida_kt = kt_actual - kt_predicho_siguiente

    if caida_kt < config.UMBRAL_CAIDA_KT:
        return None

    magnitud_estimada_mw = caida_kt * generacion_teorica_siguiente_hora
    confianza = _clasificar_confianza(caida_kt)

    return {
        "hora_prevista": timestamp_siguiente_hora,
        "caida_indice_cielo_despejado": round(caida_kt, 4),
        "magnitud_estimada_mw": round(magnitud_estimada_mw, 6),
        "confianza": confianza,
        "kt_actual": round(kt_actual, 4),
        "kt_predicho_siguiente": round(kt_predicho_siguiente, 4),
    }
