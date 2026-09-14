"""
Agente 4 — Validador.

Carga un modelo ya entrenado (XGBoost o LightGBM) de los guardados por
el Agente 3, lo evalúa contra un dataset de features (puede ser el
mismo histórico de entrenamiento u otro más reciente), calcula
métricas globales y por hora del día, y decide si el rendimiento se ha
degradado lo suficiente como para recomendar un reentrenamiento.

Funciona reutilizando dataset_prep.py del Agente 3 tal cual: la
preparación del dataset supervisado (target por horizonte, exclusión
de columnas con leakage, etc.) debe ser idéntica a la usada en
entrenamiento, o la comparación no sería justa.
"""

import json
import logging

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from pv_models import dataset_prep, config as config_models
from . import config, metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("pv_validator.validador")


def _cargar_modelo(tipo_modelo: str, horas: int):
    """Carga un modelo ya entrenado y guardado por el Agente 3."""
    if tipo_modelo == "xgboost":
        ruta = config_models.MODELS_DIR / f"xgboost_h{horas}.json"
        modelo = xgb.XGBRegressor()
        modelo.load_model(str(ruta))
        return modelo
    elif tipo_modelo == "lightgbm":
        ruta = config_models.MODELS_DIR / f"lightgbm_h{horas}.txt"
        return lgb.Booster(model_file=str(ruta))
    else:
        raise ValueError(f"Tipo de modelo desconocido: {tipo_modelo}")


def evaluar_modelo(tipo_modelo: str, horas: int, df: pd.DataFrame) -> dict:
    """
    Evalúa un modelo ya entrenado contra el dataset dado, usando
    exactamente la misma preparación de datos que en entrenamiento
    (dataset_prep.preparar_para_horizonte), para que la comparación
    sea justa.

    Devuelve un diccionario con las métricas globales, por hora del
    día, y la recomendación de reentrenamiento (comparando contra la
    referencia guardada, si existe).
    """
    df_preparado = dataset_prep.preparar_para_horizonte(df, horas)
    if df_preparado.empty:
        logger.error(f"No hay datos utilizables para evaluar el horizonte {horas}h")
        return {"error": "sin datos utilizables"}

    columnas_features = dataset_prep.seleccionar_columnas_features(df_preparado)
    modelo = _cargar_modelo(tipo_modelo, horas)

    X = df_preparado[columnas_features]
    y_true = df_preparado["target"].values

    if tipo_modelo == "lightgbm":
        y_pred = modelo.predict(X)
    else:
        y_pred = modelo.predict(X)

    metricas_globales = metrics.calcular_metricas(y_true, y_pred)

    df_resultado = df_preparado[["timestamp"]].copy()
    df_resultado["real"] = y_true
    df_resultado["prediccion"] = y_pred
    df_resultado["hora"] = df_resultado["timestamp"].dt.hour
    metricas_hora = metrics.metricas_por_hora(df_resultado, "real", "prediccion", "hora")

    clave_referencia = f"{tipo_modelo}_h{horas}"
    referencia = _cargar_metricas_referencia().get(clave_referencia)

    recomendar_reentrenar = False
    motivo = None
    if referencia is not None:
        umbral = referencia["mae"] * config.UMBRAL_DEGRADACION_RELATIVA
        if metricas_globales["mae"] > umbral:
            recomendar_reentrenar = True
            motivo = (
                f"MAE actual ({metricas_globales['mae']:.6f}) supera el umbral de "
                f"degradación ({umbral:.6f} = {config.UMBRAL_DEGRADACION_RELATIVA}x "
                f"la referencia de {referencia['mae']:.6f})"
            )
    else:
        motivo = "No había referencia guardada -- no se puede evaluar degradación todavía"

    resultado = {
        "tipo_modelo": tipo_modelo,
        "horas": horas,
        "metricas_globales": metricas_globales,
        "metricas_por_hora": metricas_hora,
        "referencia": referencia,
        "recomendar_reentrenar": recomendar_reentrenar,
        "motivo": motivo,
    }

    if metricas_globales["mape"] is not None:
        logger.info(
            f"[{tipo_modelo} h{horas}] MAE={metricas_globales['mae']:.6f}, "
            f"RMSE={metricas_globales['rmse']:.6f}, MAPE={metricas_globales['mape']:.1f}%"
        )
    else:
        logger.info(
            f"[{tipo_modelo} h{horas}] MAE={metricas_globales['mae']:.6f}, "
            f"RMSE={metricas_globales['rmse']:.6f}"
        )
    if recomendar_reentrenar:
        logger.warning(f"[{tipo_modelo} h{horas}] REENTRENAMIENTO RECOMENDADO: {motivo}")
    else:
        logger.info(f"[{tipo_modelo} h{horas}] {motivo or 'Rendimiento dentro de lo esperado'}")

    return resultado


def _cargar_metricas_referencia() -> dict:
    """Carga el JSON de métricas de referencia, o {} si no existe todavía."""
    if not config.RUTA_METRICAS_REFERENCIA.exists():
        return {}
    with open(config.RUTA_METRICAS_REFERENCIA) as f:
        return json.load(f)


def establecer_referencia(tipo_modelo: str, horas: int, metricas_globales: dict) -> None:
    """
    Guarda las métricas actuales como nueva referencia para este
    modelo/horizonte. Se usa la primera vez que se valida un modelo
    (bootstrap), o deliberadamente tras un reentrenamiento exitoso,
    para que las próximas validaciones comparen contra el rendimiento
    ya aceptado como bueno.
    """
    referencias = _cargar_metricas_referencia()
    clave = f"{tipo_modelo}_h{horas}"
    referencias[clave] = {"mae": metricas_globales["mae"], "rmse": metricas_globales["rmse"]}

    config.RUTA_METRICAS_REFERENCIA.parent.mkdir(parents=True, exist_ok=True)
    with open(config.RUTA_METRICAS_REFERENCIA, "w") as f:
        json.dump(referencias, f, indent=2)

    logger.info(f"Referencia actualizada para {clave}: MAE={metricas_globales['mae']:.6f}")
