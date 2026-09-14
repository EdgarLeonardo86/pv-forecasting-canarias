# Agente 4 — Validador

Evalúa un modelo ya entrenado por el Agente 3 (XGBoost o LightGBM)
contra un dataset de features, calcula métricas (MAE, RMSE, MAPE en
horas de día) globales y por hora del día, y recomienda reentrenar si
el rendimiento se ha degradado respecto a una referencia guardada.

## Estructura

```
pv_validator/
├── config.py       # umbral de degradación, rutas
├── metrics.py        # cálculo de MAE/RMSE/MAPE
├── validador.py       # carga de modelos, evaluación, comparación con referencia
└── main.py             # CLI
```

## Requisito previo

Necesitas al menos un modelo ya entrenado y guardado por el Agente 3
(`data/models/xgboost_h{N}.json` o `data/models/lightgbm_h{N}.txt`).

## Uso

**Primera vez** (establece la referencia inicial, ya que sin ella no
hay nada contra lo que comparar la degradación):

```bash
python -m pv_validator.main --features data/features/2024-01-01_2024-12-31.parquet \
    --modelo xgboost --horas 6 --establecer-referencia
```

**Validaciones posteriores** (por ejemplo, con datos más recientes a
medida que el Colector va acumulando histórico nuevo):

```bash
python -m pv_validator.main --features data/features/<nuevo_rango>.parquet \
    --modelo xgboost --horas 6
```

Si el MAE actual supera `UMBRAL_DEGRADACION_RELATIVA` (1.5x por
defecto, es decir, un 50% peor) respecto a la referencia guardada, el
validador recomienda reentrenar.

## Decisiones de diseño

- **Reutiliza `dataset_prep.py` del Agente 3 tal cual**: la
  preparación del dataset de evaluación (target por horizonte,
  exclusión de columnas con leakage) es idéntica a la de entrenamiento,
  para que la comparación de métricas sea justa y no se comparen
  peras con manzanas.
- **MAPE solo en horas de día**: dividir por un target casi-cero
  (horas de noche) produce porcentajes de error absurdos que no
  aportan información real sobre el rendimiento del modelo.
- **Umbral relativo, no absoluto**: se compara contra el MAE de
  referencia de ESE modelo y horizonte concretos, no contra un valor
  fijo — cada horizonte tiene una dificultad distinta (ver resultados
  del Agente 3), así que un umbral único no tendría sentido para todos.
- **La referencia se establece explícitamente**, nunca de forma
  automática: evita que el sistema "se acostumbre" silenciosamente a
  un rendimiento cada vez peor sin que nadie se entere.

## Próximo paso

Con el Validador funcionando, el siguiente paso natural del sistema
completo original es el **Agente 5 — Alertas** (detección de ramp
events y notificación), y finalmente el dashboard de visualización.
