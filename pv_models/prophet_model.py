"""
Entrenamiento y evaluación de Prophet (estacionalidad anual/semanal/
diaria y festivos de España).

A diferencia de xgboost_model.py y lightgbm_model.py, Prophet no sigue
el enfoque de "un modelo por horizonte": ajusta una única curva de
tendencia + estacionalidad + festivos sobre todo el histórico de train,
y predice directamente sobre las fechas de test (que pueden ser
cualquier horizonte, desde 1h hasta meses vista, sin re-entrenar).

Prophet requiere sus propias columnas 'ds' (fecha, sin zona horaria) e
'y' (target). Se le añade 'generacion_teorica_mw' como regresor externo
porque es determinista (climatología PVGIS) y por tanto legítimamente
conocido de antemano para cualquier fecha futura -- a diferencia del
GHI real, que en producción real dependería de un pronóstico
meteorológico con su propia incertidumbre.
"""

import logging
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

from . import config, dataset_prep

logger = logging.getLogger("pv_models.prophet_model")


def preparar_datos_prophet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte el dataset de features al formato que espera Prophet:
    columna 'ds' (timestamp SIN zona horaria -- Prophet no admite
    tz-aware) y 'y' (el target), más los regresores configurados.
    """
    columnas = ["timestamp", config.TARGET_COL] + config.PROPHET_REGRESORES
    df_prophet = df[columnas].copy()
    df_prophet = df_prophet.rename(columns={"timestamp": "ds", config.TARGET_COL: "y"})
    df_prophet["ds"] = df_prophet["ds"].dt.tz_localize(None)
    df_prophet = df_prophet.dropna()
    return df_prophet


def entrenar_y_evaluar(df: pd.DataFrame) -> dict:
    """
    Entrena Prophet sobre el primer FRACCION_TRAIN del histórico y lo
    evalúa sobre el resto (split cronológico, igual que en
    xgboost_model.py / lightgbm_model.py).
    """
    df_prophet = preparar_datos_prophet(df)
    if df_prophet.empty:
        logger.error("No hay datos utilizables para Prophet tras la preparación")
        return {"modelo": None, "error": "sin datos utilizables"}

    corte = int(len(df_prophet) * config.FRACCION_TRAIN)
    train = df_prophet.iloc[:corte]
    test = df_prophet.iloc[corte:]

    modelo = Prophet(
        # Estacionalidad ANUAL desactivada deliberadamente: con solo 1
        # año de histórico, Prophet avisa de que este componente queda
        # mal identificado y se confunde con la tendencia (ver warning
        # en el log). No hace falta que Prophet la estime de todos
        # modos: el regresor 'generacion_teorica_mw' (climatología
        # PVGIS) ya captura la forma anual real de manera exacta y
        # determinista, sin ese problema de datos insuficientes.
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True,
    )
    for regresor in config.PROPHET_REGRESORES:
        modelo.add_regressor(regresor)
    modelo.add_country_holidays(country_name=config.PROPHET_PAIS_FESTIVOS)

    logger.info(f"Entrenando Prophet con {len(train)} filas...")
    modelo.fit(train)

    columnas_prediccion = ["ds"] + config.PROPHET_REGRESORES
    prediccion = modelo.predict(test[columnas_prediccion])

    y_test = test["y"].values
    y_pred = prediccion["yhat"].values

    # Prophet no sabe que la generación FV nunca puede ser negativa --
    # su modelo de tendencia + estacionalidad es aditivo y sin
    # restricciones. Sin este recorte, las predicciones nocturnas
    # (donde el valor real es siempre 0) pueden salir negativas y
    # disparar el error de forma desproporcionada.
    n_negativas = int((y_pred < 0).sum())
    if n_negativas > 0:
        logger.warning(
            f"Prophet predijo {n_negativas} valores negativos "
            f"({n_negativas / len(y_pred) * 100:.1f}% del test) -- recortados a 0"
        )
    y_pred = np.clip(y_pred, a_min=0, a_max=None)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Reconstruir 'es_de_dia' para separar métricas de día/noche,
    # usando el propio target: de noche el target original es 0.
    es_de_dia_test = y_test > 0
    if es_de_dia_test.sum() > 0:
        mae_dia = mean_absolute_error(y_test[es_de_dia_test], y_pred[es_de_dia_test])
        rmse_dia = np.sqrt(mean_squared_error(y_test[es_de_dia_test], y_pred[es_de_dia_test]))
    else:
        mae_dia = rmse_dia = None

    # Baseline: media del target en el propio train (Prophet no tiene
    # un "lag_1h" tan natural como los otros modelos, así que se usa
    # la media histórica como referencia mínima de sentido común).
    baseline_media = train["y"].mean()
    mae_baseline = mean_absolute_error(y_test, np.full_like(y_test, baseline_media))

    resultado = {
        "modelo": modelo,
        "n_train": len(train),
        "n_test": len(test),
        "mae": mae,
        "rmse": rmse,
        "mae_dia": mae_dia,
        "rmse_dia": rmse_dia,
        "mae_baseline_media": mae_baseline,
    }

    mensaje_dia = f", MAE_dia={mae_dia:.6f}" if mae_dia is not None else ""
    logger.info(
        f"Prophet: MAE={mae:.6f}, RMSE={rmse:.6f} "
        f"(train={len(train)}, test={len(test)}), baseline media MAE={mae_baseline:.6f}{mensaje_dia}"
    )

    return resultado


def guardar_modelo(resultado: dict) -> None:
    """Guarda el modelo Prophet entrenado (formato JSON propio de Prophet)."""
    if resultado.get("modelo") is None:
        return
    from prophet.serialize import model_to_json

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = config.MODELS_DIR / "prophet.json"
    with open(ruta, "w") as f:
        f.write(model_to_json(resultado["modelo"]))
    logger.info(f"Modelo guardado: {ruta}")
