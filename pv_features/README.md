# Agente 2 — Feature engineer

Lee el histórico crudo ya combinado y validado por el Agente 1 (storage
en `data/raw/`) y construye el dataset final de features, listo para
entrenar los modelos del Agente 3.

## Estructura

```
pv_features/
├── config.py               # parámetros del panel, lags, ventanas
├── imputation.py             # interpolación de huecos cortos
├── solar_geometry.py         # ángulo cenital/azimut/elevación (pvlib)
├── temperature_model.py      # temp_panel (modelo NOCT simplificado)
├── time_features.py          # hora/día del año en seno/coseno
├── lag_features.py           # lags (1h/24h/168h) y media móvil de nubosidad
├── ree_reconciliation.py     # factor de corrección diario vs REE
├── feature_engineer.py       # orquestador: construir_features(...)
└── main.py                   # punto de entrada CLI
```

## Requisito previo

Este agente **no llama a ninguna API** — lee directamente de
`data/raw/`, el storage que genera el Agente 1. Asegúrate de haber
ejecutado el Colector para el rango de fechas que quieras procesar
antes de lanzar el feature engineer.

**Paso adicional obligatorio la primera vez**: antes de generar
features, hay que construir la envolvente empírica de cielo despejado
(usada para el target de generación), con el máximo histórico posible:

```bash
python -m pv_features.main --construir-envolvente --start 2024-01-01 --end 2024-12-31
```

Esto se hace **una sola vez** (o cada vez que quieras recalibrar con
más histórico acumulado) — no hace falta repetirlo cada vez que
generes features para un rango nuevo. Ver `clearsky_envelope.py` para
el porqué de este enfoque.

## Instalación

```bash
pip install -r pv_features/requirements.txt
```

(pvlib es la única dependencia nueva respecto al Agente 1)

## Uso

```bash
python -m pv_features.main --start 2024-01-01 --end 2024-01-31
```

Guarda el resultado en `data/features/2024-01-01_2024-01-31.parquet`.

O desde código:

```python
from pv_features.feature_engineer import construir_features

df = construir_features("2024-01-01", "2024-01-31")
print(df.columns.tolist())
```

## Features generadas

| Categoría | Columnas | Notas |
|---|---|---|
| Geometría solar | `angulo_cenital_solar`, `azimut_solar`, `elevacion_solar`, `es_de_dia` | Puramente astronómico, calculado con pvlib |
| Térmica | `temp_panel` | Modelo NOCT simplificado a partir de `temp_ambiente` y `ghi` |
| Tiempo cíclico | `hora_sin`, `hora_cos`, `dia_anio_sin`, `dia_anio_cos`, `mes` | Codificación circular para el modelo de ML |
| **Target de generación** | `ghi_cielo_despejado`, `indice_cielo_despejado`, `generacion_horaria_estimada_mw` | **Es el target a predecir por el Agente 3.** Ver `clearsky_target.py` y `clearsky_envelope.py`: combina la curva teórica de PVGIS con el índice de cielo despejado (kt = GHI real / envolvente empírica de GHI) para obtener un proxy realista de generación horaria, con variabilidad de nubes real. La envolvente es un percentil 97 del GHI histórico por mes/hora, no un modelo físico (ver historial de diseño en el docstring de `clearsky_target.py`) |
| Lags | `generacion_teorica_mw_lag_{1,24,168}h`, `ghi_lag_{1,24,168}h`, `generacion_horaria_estimada_mw_lag_{1,24,168}h` | Requieren serie horaria continua (ver imputación) |
| Media móvil | `nubosidad_media_movil` | Ventana de 3h por defecto |
| Reconciliación REE | `factor_correccion_diario`, `factor_correccion_diario_log` | Ratio diario real(REE)/teórico(PVGIS); la versión `_log` (log1p) es la recomendada para entrenar modelos, ya que la escala absoluta es muy grande (ver docstring de `ree_reconciliation.py`) |

## Notas importantes

- **La imputación es deliberadamente conservadora**: solo interpola
  huecos de hasta `MAX_HUECO_INTERPOLABLE_HORAS` (3h por defecto).
  Huecos más largos se dejan como `NaN` a propósito — el Agente 3
  decide cómo tratarlos, en vez de que este agente invente datos.
- **Los lags requieren serie horaria continua**: se calculan por
  `shift()` de fila, no por diferencia real de tiempo. Si el rango
  pedido tiene huecos largos sin imputar, los lags de las primeras
  filas después del hueco quedarán mal alineados temporalmente.
  Revisar esto quedaría bien como validación adicional en el Agente 4.
- **`factor_correccion_diario` no es una reescala física exacta**:
  compara la forma relativa de la generación diaria real (todo el
  sistema de Canarias, vía REE) con la curva teórica normalizada a
  1 kWp (PVGIS). Es una feature útil de contexto/nubosidad agregada,
  no un factor de escala para convertir MW teóricos en MW reales.
- **El target (`generacion_horaria_estimada_mw`) es un proxy, no una
  medición real.** Se construye multiplicando la curva teórica de
  PVGIS por el índice de cielo despejado (kt) calculado a partir del
  GHI real de Open-Meteo y el GHI de cielo despejado de pvlib. Es una
  técnica estándar del sector para estimar generación horaria sin un
  contador real, pero conviene tenerlo presente al interpretar
  resultados del Agente 3: los ramp events que el modelo aprenda a
  predecir son ramp events de este proxy, no de una instalación real
  medida.

## Próximo paso

Con el dataset de features y el target ya construidos, el siguiente
paso es el **Agente 3 — Modelo ML**: entrenar XGBoost (1-6h), LightGBM
(6-48h) y Prophet (estacionalidad) para predecir
`generacion_horaria_estimada_mw`, usando el resto de columnas como
variables explicativas.
