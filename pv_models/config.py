"""
Configuración del Agente 3 - Modelo ML (XGBoost, horizonte 1-6h).
"""

from pathlib import Path

TARGET_COL = "generacion_horaria_estimada_mw"

# Horizontes de predicción, en horas (h=1 significa "predecir la
# generación de dentro de 1 hora, usando solo datos de ahora mismo")
HORIZONTES_HORAS = [1, 2, 3, 4, 5, 6]

# Columnas que NUNCA deben usarse como feature de entrada, porque
# filtran información del futuro (data leakage) o son metadatos sin
# valor predictivo:
COLUMNAS_EXCLUIDAS = [
    "timestamp",
    "fuente_datos",
    "calidad_dato",
    "fuentes_faltantes",
    "fecha_ingesta",
    # Filtran el total del DÍA COMPLETO de REE -- no se conoce hasta
    # las 23:59 de ese día, así que usarlo a las 8:00 sería leakage:
    "generacion_real_diaria_mwh",
    "demanda_total_diaria_mwh",
    "factor_correccion_diario",
    "factor_correccion_diario_log",
    # NASA POWER llega con varios días de desfase de publicación en
    # producción real -- no estaría disponible a tiempo:
    "ghi_historico",
    # El propio target y sus componentes intermedios se excluyen de
    # las FEATURES (aunque sus LAGS sí se usan, ver dataset_prep.py):
    TARGET_COL,
    # Blindaje adicional: 'target' es el nombre literal de la columna
    # que dataset_prep.py añade temporalmente al construir el dataset
    # supervisado por horizonte. Nunca debe usarse como feature de
    # entrada (sería literalmente la respuesta), así que se excluye
    # aquí también por si se recalculan las columnas de features en
    # un punto del código donde esa columna ya existe en el DataFrame.
    "target",
    # Mismo motivo: 'es_de_dia_objetivo' se añade para poder evaluar
    # el modelo correctamente por horas de día/noche del momento
    # OBJETIVO (t+horas), no debe colarse como feature de entrada.
    "es_de_dia_objetivo",
]

# --- División temporal train/test ---
# División CRONOLÓGICA, no aleatoria: en series temporales, mezclar al
# azar dejaría que el modelo "viera" el futuro durante el entrenamiento
# a través de filas cercanas en el tiempo a las de test.
FRACCION_TRAIN = 0.8

# --- Hiperparámetros XGBoost (punto de partida razonable, no optimizado) ---
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": 42,
}

# --- LightGBM: horizonte medio (6-48h) ---
LIGHTGBM_HORIZONTES_HORAS = [6, 12, 18, 24, 30, 36, 42, 48]

LIGHTGBM_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,  # silenciar warnings de LightGBM sobre datasets pequeños
}

# --- Prophet: estacionalidad anual/semanal/diaria y festivos ---
# A diferencia de XGBoost/LightGBM, Prophet no usa el enfoque de
# "un modelo por horizonte" -- ajusta una curva continua de
# tendencia + estacionalidad y predice directamente sobre fechas
# futuras. Se le pasa 'generacion_teorica_mw' como regresor externo
# porque es determinista (climatología PVGIS, conocida de antemano
# para cualquier fecha futura), a diferencia del GHI real que
# dependería de un pronóstico meteorológico no disponible aquí.
PROPHET_REGRESORES = ["generacion_teorica_mw"]
PROPHET_PAIS_FESTIVOS = "ES"

# --- Rutas ---
MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
