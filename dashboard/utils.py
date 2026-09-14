"""
Funciones de datos para el dashboard de Streamlit.

Reutiliza directamente los módulos de los 5 agentes (mismo repo) en
vez de duplicar lógica: pv_collector.config para la ubicación,
pv_models.dataset_prep para preparar datos y cargar modelos,
pv_validator para métricas, pv_alertas para ramp events.
"""

import sys
from pathlib import Path

# Streamlit ejecuta el script desde dashboard/, así que hay que añadir
# la raíz del proyecto al sys.path para poder importar pv_collector,
# pv_models, pv_validator y pv_alertas como si se ejecutara desde ahí.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
import xgboost as xgb
import lightgbm as lgb

from pv_collector import config as config_collector
from pv_models import dataset_prep, config as config_models
from pv_validator import validador as validador_agente4
from pv_alertas import deteccion

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@st.cache_data
def listar_datasets_features() -> list:
    """Lista los parquet de features disponibles en data/features/."""
    carpeta = DATA_DIR / "features"
    if not carpeta.exists():
        return []
    return sorted([str(p) for p in carpeta.glob("*.parquet")], reverse=True)


@st.cache_data
def cargar_features(ruta: str) -> pd.DataFrame:
    return dataset_prep.cargar_dataset(ruta)


@st.cache_data
def listar_modelos_disponibles() -> dict:
    """
    Devuelve {tipo_modelo: [horizontes disponibles]} según lo que haya
    guardado en data/models/.
    """
    carpeta = config_models.MODELS_DIR
    disponibles = {"xgboost": [], "lightgbm": []}
    if not carpeta.exists():
        return disponibles

    for fichero in carpeta.glob("xgboost_h*.json"):
        horas = int(fichero.stem.replace("xgboost_h", ""))
        disponibles["xgboost"].append(horas)
    for fichero in carpeta.glob("lightgbm_h*.txt"):
        horas = int(fichero.stem.replace("lightgbm_h", ""))
        disponibles["lightgbm"].append(horas)

    disponibles["xgboost"].sort()
    disponibles["lightgbm"].sort()
    return disponibles


@st.cache_data
def generar_predicciones(ruta_features: str, tipo_modelo: str, horas: int) -> pd.DataFrame:
    """
    Carga el dataset de features, prepara el problema supervisado para
    el horizonte dado, carga el modelo ya entrenado y devuelve un
    DataFrame con timestamp (del momento OBJETIVO, no de emisión),
    valor real y predicción -- listo para graficar.
    """
    df = cargar_features(ruta_features)
    df_prep = dataset_prep.preparar_para_horizonte(df, horas)
    columnas_features = dataset_prep.seleccionar_columnas_features(df_prep)

    if tipo_modelo == "xgboost":
        modelo = xgb.XGBRegressor()
        modelo.load_model(str(config_models.MODELS_DIR / f"xgboost_h{horas}.json"))
    else:
        modelo = lgb.Booster(model_file=str(config_models.MODELS_DIR / f"lightgbm_h{horas}.txt"))

    predicciones = modelo.predict(df_prep[columnas_features])

    resultado = pd.DataFrame({
        "timestamp_objetivo": df_prep["timestamp"] + pd.Timedelta(hours=horas),
        "real": df_prep["target"].values,
        "prediccion": predicciones,
    })
    return resultado


@st.cache_data
def cargar_metricas_referencia() -> dict:
    return validador_agente4.cargar_metricas_referencia()


@st.cache_data
def evaluar_todos_los_modelos(ruta_features: str) -> pd.DataFrame:
    """
    Evalúa todos los modelos/horizontes disponibles contra el dataset
    dado y devuelve una tabla resumen (una fila por modelo/horizonte).
    """
    df = cargar_features(ruta_features)
    disponibles = listar_modelos_disponibles()

    filas = []
    for tipo_modelo, horizontes in disponibles.items():
        for horas in horizontes:
            resultado = validador_agente4.evaluar_modelo(tipo_modelo, horas, df)
            if "error" in resultado:
                continue
            m = resultado["metricas_globales"]
            filas.append({
                "modelo": tipo_modelo,
                "horizonte_horas": horas,
                "mae": m["mae"],
                "rmse": m["rmse"],
                "mape_dia": m["mape"],
                "recomendar_reentrenar": resultado["recomendar_reentrenar"],
            })

    return pd.DataFrame(filas)


@st.cache_data
def cargar_eventos_historicos(ruta_features: str) -> pd.DataFrame:
    df = cargar_features(ruta_features)
    return deteccion.detectar_historico(df)


def ubicacion_referencia() -> tuple:
    """Latitud/longitud de referencia usada en todo el proyecto (Agente 1)."""
    return config_collector.LATITUDE, config_collector.LONGITUDE
