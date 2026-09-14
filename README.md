# Sistema Multiagente de Previsión Fotovoltaica — Canarias

Sistema de 5 agentes en pipeline que predicen la generación fotovoltaica
horaria en Gran Canaria y detectan caídas bruscas de generación por
nubosidad (*ramp events*), usando únicamente fuentes de datos públicas
y gratuitas (Open-Meteo, REE, NASA POWER, PVGIS).

Proyecto de portfolio orientado a roles de *energy data analyst*,
conectado con un TFB sobre predicción temprana de ramp events.

📄 **[Resumen ejecutivo completo](./resumen_ejecutivo_proyecto.md)** —
arquitectura, decisiones de diseño, bugs encontrados, resultados y
limitaciones.

🚀 **[Dashboard en vivo](https://pv-forecasting-canarias-24cfz37h9x4wnkmajonzwl.streamlit.app/)** —
pruébalo directamente, sin instalar nada.

## Arquitectura

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────┐    ┌────────────┐
│  Agente 1   │───▶│  Agente 2    │───▶│  Agente 3   │───▶│  Agente 4  │───▶│  Agente 5  │
│  Colector   │    │  Features    │    │  Modelos ML │    │ Validador  │    │  Alertas   │
└─────────────┘    └──────────────┘    └─────────────┘    └────────────┘    └────────────┘
  4 APIs →           geometría solar,    XGBoost 1-6h       MAE/RMSE vs      ramp events
  Parquet             target FV,         LightGBM 6-48h     referencia       (kt) + aviso
                       envolvente kt      Prophet             guardada        temprano
```

| Agente | Carpeta | README |
|---|---|---|
| 1 — Colector | [`pv_collector/`](./pv_collector) | [README](./pv_collector/README.md) |
| 2 — Feature engineer | [`pv_features/`](./pv_features) | [README](./pv_features/README.md) |
| 3 — Modelos ML | [`pv_models/`](./pv_models) | [README](./pv_models/README.md) |
| 4 — Validador | [`pv_validator/`](./pv_validator) | [README](./pv_validator/README.md) |
| 5 — Alertas | [`pv_alertas/`](./pv_alertas) | [README](./pv_alertas/README.md) |

## Instalación rápida

```bash
git clone <url-del-repo>
cd pv-forecasting-canarias
python -m venv venv
source venv/bin/activate
pip install -r pv_collector/requirements.txt
pip install -r pv_features/requirements.txt
pip install -r pv_models/requirements.txt
pip install -r pv_validator/requirements.txt
pip install -r pv_alertas/requirements.txt
```

En Mac, XGBoost necesita además `brew install libomp` (librería del
sistema, no se instala vía pip).

## Uso de punta a punta

```bash
# 1. Recoger un año de histórico
python -m pv_collector.main --start 2024-01-01 --end 2024-12-31

# 2. Construir la envolvente de cielo despejado y generar features
python -m pv_features.main --construir-envolvente --start 2024-01-01 --end 2024-12-31
python -m pv_features.main --start 2024-01-01 --end 2024-12-31

# 3. Entrenar los tres modelos
python -m pv_models.main --features data/features/2024-01-01_2024-12-31.parquet --modelo xgboost
python -m pv_models.main --features data/features/2024-01-01_2024-12-31.parquet --modelo lightgbm
python -m pv_models.main --features data/features/2024-01-01_2024-12-31.parquet --modelo prophet

# 4. Establecer la referencia de validación
python -m pv_validator.main --features data/features/2024-01-01_2024-12-31.parquet --modelo xgboost --horas 6 --establecer-referencia

# 5. Detectar ramp events
python -m pv_alertas.main --modo historico --features data/features/2024-01-01_2024-12-31.parquet
python -m pv_alertas.main --modo backtest --features data/features/2024-01-01_2024-12-31.parquet
```

## Stack técnico

Python · pandas · pvlib · XGBoost · LightGBM · Prophet · scikit-learn ·
PyArrow (Parquet)

## Datos y modelos no incluidos en el repo

La carpeta `data/` (Parquet de histórico, features, modelos entrenados)
está excluida vía `.gitignore` — se regenera ejecutando los comandos de
arriba. Los modelos entrenados (`data/models/`) tampoco se versionan
por tamaño; considera Git LFS si quieres incluirlos en el futuro.

## Limitaciones conocidas

Ver la sección "Limitaciones y trabajo futuro" del
[resumen ejecutivo](./resumen_ejecutivo_proyecto.md) — en particular,
el sistema de aviso temprano de ramp events (Agente 5) tiene baja
precisión/recall con el enfoque actual, documentado con un análisis
cuantitativo de sensibilidad.

## Dashboard

Código en [`dashboard/`](./dashboard) — [README](./dashboard/README.md).
Desplegado en Streamlit Community Cloud (enlace arriba). Para correrlo
en local: `streamlit run dashboard/app.py` desde la raíz del proyecto.
