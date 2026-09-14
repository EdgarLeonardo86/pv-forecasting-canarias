"""
Agente 2 — Feature engineer.

Lee el histórico ya combinado y validado del Agente 1 (storage.leer_rango),
y construye el dataset final listo para entrenar los modelos del
Agente 3, aplicando en orden:

1. Imputación de huecos cortos (interpolación lineal)
2. Geometría solar (pvlib): ángulo cenital, azimut, elevación
3. Modelo térmico del panel (NOCT simplificado)
4. Features cíclicas de tiempo (hora/día del año en seno/coseno)
5. Target de generación horaria estimada (índice de cielo despejado, kt)
6. Lags de generación teórica, GHI y target (1h, 24h, 168h)
7. Media móvil de nubosidad
8. Factor de corrección diario (reconciliación con REE)

El orden importa: la imputación va primero para que los lags y medias
móviles no arrastren huecos innecesarios; el target se calcula antes
de los lags porque estos necesitan poder desplazar también la propia
columna objetivo (generacion_horaria_estimada_mw).
"""

import logging
import pandas as pd

from pv_collector import storage as storage_agente1

from . import (
    imputation,
    solar_geometry,
    temperature_model,
    time_features,
    clearsky_target,
    lag_features,
    ree_reconciliation,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("pv_features.feature_engineer")


def construir_features(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Punto de entrada único del Agente 2.

    Lee el rango [start_date, end_date] del storage del Agente 1 y
    devuelve el DataFrame final con todas las features añadidas,
    listo para el Agente 3.
    """
    logger.info(f"=== Construyendo features {start_date} -> {end_date} ===")

    df = storage_agente1.leer_rango(start_date, end_date)
    if df.empty:
        logger.error("No hay datos crudos en ese rango. Ejecuta primero el Agente 1.")
        return df

    filas_iniciales = len(df)

    df = imputation.interpolar_huecos_cortos(df)
    df = solar_geometry.calcular_geometria_solar(df)
    df = temperature_model.calcular_temp_panel(df)
    df = time_features.calcular_features_tiempo(df)
    df = clearsky_target.calcular_target_generacion(df)
    df = lag_features.calcular_lags(df)
    df = lag_features.calcular_media_movil_nubosidad(df)
    df = ree_reconciliation.calcular_factor_correccion_diario(df)

    logger.info(
        f"=== Features construidas: {len(df)} filas ({filas_iniciales} de entrada), "
        f"{len(df.columns)} columnas ==="
    )
    return df
