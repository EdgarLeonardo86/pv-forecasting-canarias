# Dashboard — Previsión FV Canarias

Dashboard interactivo en Streamlit que visualiza los resultados de los
5 agentes: generación real vs. predicha, comparativa de modelos,
ramp events detectados, y contexto geográfico.

## Requisito previo

Necesitas haber ejecutado ya al menos:
1. Agente 1 (Colector) — genera `data/raw/`
2. Agente 2 (Feature engineer) — genera `data/features/*.parquet`
3. Agente 3 (Modelos ML) — genera `data/models/*`

El Agente 4 (referencia guardada) y el Agente 5 (eventos históricos)
son opcionales: si faltan, las pestañas correspondientes lo indican
en vez de fallar.

## Instalación

```bash
pip install -r dashboard/requirements.txt
```

## Uso

Desde la **raíz del proyecto** (importante: no desde dentro de `dashboard/`):

```bash
streamlit run dashboard/app.py
```

Se abrirá automáticamente en `http://localhost:8501`.

## Pestañas

- **📈 Generación**: curva real vs. predicha para el modelo/horizonte
  elegido, con selector de rango de fechas.
- **🤖 Modelos**: comparativa de MAE/RMSE entre XGBoost y LightGBM por
  horizonte, y el estado de recomendación de reentrenamiento del
  Agente 4.
- **⚡ Ramp events**: eventos detectados en el histórico, con la
  limitación del sistema de aviso temprano señalada explícitamente.
- **📍 Contexto**: ubicación de referencia y resumen de la arquitectura.

## Notas de diseño

- Reutiliza los módulos de los 5 agentes directamente (`pv_collector`,
  `pv_models`, `pv_validator`, `pv_alertas`) en vez de duplicar lógica
  de carga de modelos o cálculo de métricas.
- Usa `@st.cache_data` para no recalcular predicciones/métricas en
  cada interacción del usuario con los selectores.
- El target sigue siendo un proxy (ver resumen ejecutivo del
  proyecto), no una medición real — el dashboard lo indica en la
  cabecera para no inducir a error a quien lo vea sin contexto previo.
