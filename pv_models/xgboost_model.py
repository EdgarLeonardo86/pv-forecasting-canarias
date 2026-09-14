"""
Entrenamiento y evaluación de un modelo XGBoost por horizonte de
predicción (1-6h), usando el dataset preparado por dataset_prep.py.
"""

import logging
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from . import config, dataset_prep

logger = logging.getLogger("pv_models.xgboost_model")


def entrenar_modelo_horizonte(df: pd.DataFrame, horas: int) -> dict:
    """
    Entrena un XGBRegressor para predecir el target 'horas' horas por
    delante, y lo evalúa en un conjunto de test separado
    cronológicamente (nunca mezclado al azar con el train).

    Devuelve un diccionario con el modelo entrenado, las métricas de
    evaluación, y metadatos (columnas usadas, tamaño de train/test).
    """
    df_preparado = dataset_prep.preparar_para_horizonte(df, horas)

    if df_preparado.empty:
        logger.error(f"Horizonte {horas}h: no quedan filas utilizables tras la preparación")
        return {"horas": horas, "modelo": None, "error": "sin datos utilizables"}

    columnas_features = dataset_prep.seleccionar_columnas_features(df_preparado)
    train, test = dataset_prep.split_temporal(df_preparado)

    if train.empty or test.empty:
        logger.error(f"Horizonte {horas}h: train o test vacío tras el split")
        return {"horas": horas, "modelo": None, "error": "train/test vacío"}

    X_train, y_train = train[columnas_features], train["target"]
    X_test, y_test = test[columnas_features], test["target"]

    modelo = XGBRegressor(**config.XGBOOST_PARAMS)
    modelo.fit(X_train, y_train)

    predicciones = modelo.predict(X_test)

    importancias = pd.Series(
        modelo.feature_importances_, index=columnas_features
    ).sort_values(ascending=False)

    mae = mean_absolute_error(y_test, predicciones)
    rmse = np.sqrt(mean_squared_error(y_test, predicciones))

    # Métricas separadas para horas de DÍA: de noche el target es
    # trivialmente 0 y cualquier modelo lo acierta sin esfuerzo, lo
    # que puede inflar el MAE global y ocultar el rendimiento real
    # donde de verdad importa (variabilidad de nubes, ramp events).
    # Se filtra por 'es_de_dia_objetivo' (el momento que se está
    # prediciendo, t+horas), NO por 'es_de_dia' (el momento de emisión,
    # t) -- con horizontes como 12h o 36h, una predicción emitida de
    # día puede apuntar a una hora nocturna, y usar el filtro equivocado
    # infla artificialmente el resultado en esos horizontes concretos.
    if "es_de_dia_objetivo" in test.columns:
        mascara_dia = test["es_de_dia_objetivo"].astype(bool).values
        if mascara_dia.sum() > 0:
            mae_dia = mean_absolute_error(y_test[mascara_dia], predicciones[mascara_dia])
            rmse_dia = np.sqrt(mean_squared_error(y_test[mascara_dia], predicciones[mascara_dia]))
        else:
            mae_dia = rmse_dia = None
    else:
        mae_dia = rmse_dia = None

    # Baseline de persistencia ingenua (predecir "igual que ahora mismo",
    # usando el lag_1h del target como referencia si existe): sirve para
    # saber si el modelo realmente aporta algo sobre la opción más simple.
    columna_lag_target = f"{config.TARGET_COL}_lag_1h"
    if columna_lag_target in test.columns:
        baseline_persistencia = test[columna_lag_target]
        mae_baseline = mean_absolute_error(y_test, baseline_persistencia)
        if "es_de_dia_objetivo" in test.columns and mascara_dia.sum() > 0:
            mae_baseline_dia = mean_absolute_error(y_test[mascara_dia], baseline_persistencia[mascara_dia])
        else:
            mae_baseline_dia = None
    else:
        mae_baseline = None
        mae_baseline_dia = None

    resultado = {
        "horas": horas,
        "modelo": modelo,
        "columnas_features": columnas_features,
        "n_train": len(train),
        "n_test": len(test),
        "mae": mae,
        "rmse": rmse,
        "mae_dia": mae_dia,
        "rmse_dia": rmse_dia,
        "mae_baseline_persistencia": mae_baseline,
        "mae_baseline_persistencia_dia": mae_baseline_dia,
        "importancias": importancias,
    }

    mensaje_baseline = f", baseline MAE={mae_baseline:.6f}" if mae_baseline is not None else ""
    mensaje_dia = f", MAE_dia={mae_dia:.6f} (baseline_dia={mae_baseline_dia:.6f})" if mae_dia is not None else ""
    logger.info(
        f"Horizonte {horas}h: MAE={mae:.6f}, RMSE={rmse:.6f} "
        f"(train={len(train)}, test={len(test)}){mensaje_baseline}{mensaje_dia}"
    )

    return resultado


def entrenar_todos_horizontes(df: pd.DataFrame) -> dict:
    """
    Entrena un modelo independiente para cada horizonte en
    config.HORIZONTES_HORAS y devuelve un diccionario {horas: resultado}.
    """
    resultados = {}
    for horas in config.HORIZONTES_HORAS:
        resultados[horas] = entrenar_modelo_horizonte(df, horas)
    return resultados


def guardar_modelos(resultados: dict) -> None:
    """Guarda cada modelo entrenado en data/models/xgboost_h{horas}.json"""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for horas, resultado in resultados.items():
        if resultado.get("modelo") is None:
            continue
        ruta = config.MODELS_DIR / f"xgboost_h{horas}.json"
        resultado["modelo"].save_model(str(ruta))
        logger.info(f"Modelo guardado: {ruta}")
