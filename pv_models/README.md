# Agente 3 — Modelos ML (XGBoost 1-6h, LightGBM 6-48h)

Entrena un modelo independiente por cada horizonte de predicción,
usando el dataset de features del Agente 2. Predicción directa (un
modelo por horizonte), no recursiva. Dos algoritmos disponibles:
**XGBoost** para el horizonte corto (1-6h) y **LightGBM** para el
horizonte medio (6-48h), compartiendo toda la lógica de preparación
de datos.

## Estructura

```
pv_models/
├── config.py             # columnas excluidas, horizontes, hiperparámetros
├── dataset_prep.py         # target por horizonte, selección de features, split temporal
├── xgboost_model.py         # entrenamiento y evaluación (1-6h)
├── lightgbm_model.py        # entrenamiento y evaluación (6-48h)
└── main.py                  # CLI (elige modelo con --modelo)
```

## Requisito previo

Necesitas un parquet de features ya generado por el Agente 2 (que a su
vez requiere haber construido antes la envolvente de cielo despejado).

## Instalación

```bash
pip install -r pv_models/requirements.txt
```

En Mac, XGBoost necesita además la librería del sistema `libomp`:
`brew install libomp` (no se instala vía pip).

## Uso

```bash
python -m pv_models.main --features data/features/2024-01-01_2024-12-31.parquet --modelo xgboost
python -m pv_models.main --features data/features/2024-01-01_2024-12-31.parquet --modelo lightgbm
```

Guarda un modelo por horizonte en `data/models/xgboost_h{1..6}.json` o
`data/models/lightgbm_h{6,12,...,48}.txt`, y muestra un resumen de
MAE/RMSE en test (global y solo horas de día), comparado contra un
baseline de persistencia ingenua.

## Decisiones de diseño importantes

- **Sin data leakage**: se excluyen explícitamente las columnas que
  filtrarían información del futuro -- el total diario de REE (no se
  conoce hasta las 23:59 de ese día) y `ghi_historico` de NASA POWER
  (llega con varios días de desfase en producción real). Ver
  `config.COLUMNAS_EXCLUIDAS`.
- **Split cronológico, no aleatorio**: train = primer 80% de la serie
  temporal, test = último 20%. Mezclar al azar inflaría
  artificialmente el rendimiento aparente del modelo.
- **Predicción directa, no recursiva**: cada horizonte (1h, 2h...) tiene
  su propio modelo entrenado específicamente para esa distancia
  temporal, en vez de encadenar predicciones de 1h para llegar a 6h
  (evita que los errores se acumulen y amplifiquen).
- **Baseline de persistencia**: se compara siempre contra "predecir el
  mismo valor que hace 1 hora" (`generacion_horaria_estimada_mw_lag_1h`).
  Si el modelo no bate claramente este baseline tan simple, no está
  aportando valor real.
- **El target es un proxy** (ver README y `clearsky_target.py` del
  Agente 2), no una medición real de una instalación FV. Los resultados
  de este modelo deben interpretarse con esa salvedad.
- **Bug de leakage real detectado y corregido**: en una primera versión,
  la columna `target` se añadía al DataFrame antes de seleccionar las
  columnas de features, y se colaba a sí misma como feature de entrada
  -- el modelo veía literalmente la respuesta que tenía que predecir.
  Se detectó revisando `feature_importances_` (aparecía `target` con
  peso alto) y se corrigió reordenando `dataset_prep.preparar_para_horizonte`
  para seleccionar las features ANTES de añadir la columna target, más
  una exclusión explícita en `config.COLUMNAS_EXCLUIDAS` como blindaje
  adicional. Sirve de recordatorio: un MAE sospechosamente bueno y
  plano en todos los horizontes es señal de alarma, no de éxito.

## Próximo paso

Con XGBoost (1-6h) y LightGBM (6-48h) ya entrenados y validados, el
siguiente paso es **Prophet** (estacionalidad anual y festivos), que
tiene una interfaz distinta a scikit-learn/XGBoost/LightGBM (no usa
`dataset_prep.py` de la misma forma, ya que Prophet espera sus propias
columnas `ds`/`y` y trabaja mejor con series continuas en vez del
enfoque de horizonte directo por filas).
