"""
Diagnóstico puntual: compara la predicción del modelo XGBoost h=1
contra la realidad, justo en las horas de los 2 ramp events reales
confirmados por el modo histórico, y también mira el error general
del modelo alrededor de esas fechas.

Uso: python diagnostico_ramp.py
"""

import pandas as pd
import xgboost as xgb

from pv_models import dataset_prep, config as config_models
from pv_alertas import deteccion

RUTA_FEATURES = "data/features/2024-01-01_2024-12-31.parquet"
EVENTOS_REALES = [
    pd.Timestamp("2024-02-29 09:00:00", tz="UTC"),
    pd.Timestamp("2024-12-15 13:00:00", tz="UTC"),
]

df = dataset_prep.cargar_dataset(RUTA_FEATURES)

modelo = xgb.XGBRegressor()
modelo.load_model(str(config_models.MODELS_DIR / "xgboost_h1.json"))

df_prep = dataset_prep.preparar_para_horizonte(df, horas=1)
columnas_features = dataset_prep.seleccionar_columnas_features(df_prep)
predicciones = modelo.predict(df_prep[columnas_features])
df_prep = df_prep.copy()
df_prep["prediccion"] = predicciones

df_teorica_futura = df.copy()
df_teorica_futura["teorica_objetivo"] = dataset_prep.construir_columna_futura(
    df_teorica_futura, "generacion_teorica_mw", 1
)
df_prep["teorica_objetivo"] = df_teorica_futura.loc[df_prep.index, "teorica_objetivo"].values

for evento in EVENTOS_REALES:
    # La hora de "emisión" de la alerta sería 1h antes del evento real
    hora_emision = evento - pd.Timedelta(hours=1)
    fila = df_prep[df_prep["timestamp"] == hora_emision]

    print(f"\n=== Evento real en {evento} (emisión de alerta esperada en {hora_emision}) ===")
    if fila.empty:
        print("  No hay fila preparada para esa hora (puede haber sido descartada por NaN)")
        continue

    fila = fila.iloc[0]
    kt_actual = fila["indice_cielo_despejado"]
    pred = fila["prediccion"]
    teorica_obj = fila["teorica_objetivo"]
    real_obj = fila["target"]

    kt_predicho = pred / teorica_obj if teorica_obj > 1e-9 else None
    kt_real_objetivo = real_obj / teorica_obj if teorica_obj > 1e-9 else None

    print(f"  kt_actual (hora de emisión):      {kt_actual:.4f}")
    print(f"  kt_predicho (hora objetivo):      {kt_predicho:.4f}" if kt_predicho is not None else "  kt_predicho: N/A")
    print(f"  kt_REAL (hora objetivo):          {kt_real_objetivo:.4f}" if kt_real_objetivo is not None else "  kt_real: N/A")
    print(f"  Caída de kt que el modelo VIO:    {kt_actual - kt_predicho:.4f}" if kt_predicho is not None else "")
    print(f"  Caída de kt que REALMENTE pasó:   {kt_actual - kt_real_objetivo:.4f}" if kt_real_objetivo is not None else "")
    print(f"  generacion predicha:  {pred:.6f} MW")
    print(f"  generacion real:      {real_obj:.6f} MW")
    print(f"  error absoluto:       {abs(pred - real_obj):.6f} MW")

# Error medio general del modelo, para contextualizar si el error de
# arriba es "normal" o anormalmente grande
error_medio_general = (df_prep["prediccion"] - df_prep["target"]).abs().mean()
print(f"\n=== Referencia: MAE general del modelo en todo 2024: {error_medio_general:.6f} MW ===")
