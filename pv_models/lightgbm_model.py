"""
Entrenamiento y evaluación de un modelo LightGBM por horizonte de
predicción medio (6-48h), usando el mismo dataset_prep.py que XGBoost.

La lógica es deliberadamente un espejo de xgboost_model.py: mismo
patrón de preparación, split cronológico, baseline de persistencia y
métricas separadas por horas de día -- solo cambia el algoritmo y el
rango de horizontes.
"""

import logging
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from . import config, dataset_prep

logger = logging.getLogger("pv_models.lightgbm_model")


def entrenar_modelo_horizonte(df: pd.DataFrame, horas: int) -> dict:
    """
    Entrena un LGBMRegressor para predecir el target 'horas' horas por
    delante. Misma metodología que xgboost_model.entrenar_modelo_horizonte
    (ver ese módulo para el razonamiento detallado de cada decisión).
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

    modelo = LGBMRegressor(**config.LIGHTGBM_PARAMS)
    modelo.fit(X_train, y_train)

    predicciones = modelo.predict(X_test)

    importancias = pd.Series(
        modelo.feature_importances_, index=columnas_features
    ).sort_values(ascending=False)

    mae = mean_absolute_error(y_test, predicciones)
    rmse = np.sqrt(mean_squared_error(y_test, predicciones))

    if "es_de_dia_objetivo" in test.columns:
        mascara_dia = test["es_de_dia_objetivo"].astype(bool).values
        if mascara_dia.sum() > 0:
            mae_dia = mean_absolute_error(y_test[mascara_dia], predicciones[mascara_dia])
            rmse_dia = np.sqrt(mean_squared_error(y_test[mascara_dia], predicciones[mascara_dia]))
        else:
            mae_dia = rmse_dia = None
    else:
        mae_dia = rmse_dia = None

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
    config.LIGHTGBM_HORIZONTES_HORAS y devuelve {horas: resultado}.
    """
    resultados = {}
    for horas in config.LIGHTGBM_HORIZONTES_HORAS:
        resultados[horas] = entrenar_modelo_horizonte(df, horas)
    return resultados


def guardar_modelos(resultados: dict) -> None:
    """Guarda cada modelo entrenado en data/models/lightgbm_h{horas}.txt"""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for horas, resultado in resultados.items():
        if resultado.get("modelo") is None:
            continue
        ruta = config.MODELS_DIR / f"lightgbm_h{horas}.txt"
        resultado["modelo"].booster_.save_model(str(ruta))
        logger.info(f"Modelo guardado: {ruta}")
