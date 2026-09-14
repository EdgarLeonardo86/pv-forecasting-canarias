"""
Punto de entrada del Agente 5 - Alertas.

Modo histórico (ramp events ya ocurridos, sobre datos observados):
    python -m pv_alertas.main --modo historico --features data/features/2024-01-01_2024-12-31.parquet

Modo backtest (simula el aviso TEMPRANO recorriendo el histórico como
si fuera tiempo real, usando el modelo XGBoost de 1h ya entrenado):
    python -m pv_alertas.main --modo backtest --features data/features/2024-01-01_2024-12-31.parquet
"""

import argparse
import logging

import pandas as pd
import xgboost as xgb

from pv_models import dataset_prep, config as config_models
from . import deteccion, notificador

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("pv_alertas.main")


def modo_historico(ruta_features: str):
    df = dataset_prep.cargar_dataset(ruta_features)
    eventos = deteccion.detectar_historico(df)

    print()
    print(f"=== Ramp events detectados en el histórico: {len(eventos)} ===")
    if not eventos.empty:
        print(eventos.head(20).to_string(index=False))
        if len(eventos) > 20:
            print(f"... y {len(eventos) - 20} más")

        ruta_salida = config_models.MODELS_DIR.parent / "alertas" / "eventos_historicos.parquet"
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        eventos.to_parquet(ruta_salida, index=False, engine="pyarrow")
        print(f"\nGuardado: {ruta_salida}")


def modo_backtest(ruta_features: str):
    df = dataset_prep.cargar_dataset(ruta_features)

    ruta_modelo = config_models.MODELS_DIR / "xgboost_h1.json"
    modelo = xgb.XGBRegressor()
    modelo.load_model(str(ruta_modelo))

    df_prep = dataset_prep.preparar_para_horizonte(df, horas=1)
    columnas_features = dataset_prep.seleccionar_columnas_features(df_prep)

    predicciones = modelo.predict(df_prep[columnas_features])

    # Teórica de la HORA OBJETIVO (t+1): determinista (PVGIS), por eso
    # se puede conocer de antemano sin necesitar ningún pronóstico.
    df_teorica_futura = df.copy()
    df_teorica_futura["teorica_objetivo"] = dataset_prep.construir_columna_futura(
        df_teorica_futura, "generacion_teorica_mw", 1
    )
    teorica_objetivo = df_teorica_futura.loc[df_prep.index, "teorica_objetivo"].values

    kt_actual = df_prep["indice_cielo_despejado"].values
    timestamps = df_prep["timestamp"].values

    n_alertas = 0
    for i in range(len(df_prep)):
        alerta = deteccion.evaluar_prediccion_ramp(
            kt_actual=kt_actual[i],
            generacion_predicha_siguiente_hora=predicciones[i],
            generacion_teorica_siguiente_hora=teorica_objetivo[i],
            timestamp_siguiente_hora=pd.Timestamp(timestamps[i]) + pd.Timedelta(hours=1),
        )
        if alerta is not None:
            notificador.notificar(alerta)
            n_alertas += 1

    print()
    print(f"=== Backtest completado: {n_alertas} alertas emitidas de {len(df_prep)} horas simuladas ===")
    print(f"Registro completo en: {notificador.config.RUTA_LOG_ALERTAS}")


def main():
    parser = argparse.ArgumentParser(description="Agente 5 - Alertas")
    parser.add_argument("--modo", choices=["historico", "backtest"], required=True)
    parser.add_argument("--features", type=str, required=True, help="Ruta al parquet de features")
    args = parser.parse_args()

    if args.modo == "historico":
        modo_historico(args.features)
    else:
        modo_backtest(args.features)


if __name__ == "__main__":
    main()
