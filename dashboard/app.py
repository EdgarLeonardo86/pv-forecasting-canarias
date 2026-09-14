"""
Dashboard del Sistema Multiagente de Previsión Fotovoltaica — Canarias.

Uso:
    streamlit run dashboard/app.py

Requiere haber ejecutado ya los Agentes 1-5 al menos una vez (para
tener features, modelos entrenados y, opcionalmente, alertas
generadas). Si falta algo, cada pestaña avisa de qué comando ejecutar.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import utils

st.set_page_config(
    page_title="Previsión FV — Canarias",
    page_icon="☀️",
    layout="wide",
)

st.title("☀️ Sistema Multiagente de Previsión Fotovoltaica — Gran Canaria")
st.caption(
    "Proyecto de portfolio con datos 100% gratuitos (Open-Meteo, REE, NASA POWER, PVGIS). "
    "El target es un proxy de generación (climatología PVGIS × índice de cielo despejado), no una medición real."
)

# --- Selector de dataset compartido por todas las pestañas ---
datasets = utils.listar_datasets_features()
if not datasets:
    st.error(
        "No se encontró ningún dataset de features en `data/features/`. "
        "Ejecuta primero el Agente 1 (Colector) y el Agente 2 (Feature engineer)."
    )
    st.stop()

ruta_dataset = st.sidebar.selectbox("Dataset de features", datasets, index=0)
st.sidebar.caption(f"Usando: `{ruta_dataset}`")

tab_generacion, tab_modelos, tab_ramp, tab_contexto = st.tabs(
    ["📈 Generación", "🤖 Modelos", "⚡ Ramp events", "📍 Contexto"]
)

# ============================================================
# TAB 1 — Generación real vs. predicha
# ============================================================
with tab_generacion:
    st.subheader("Generación real vs. predicha")

    disponibles = utils.listar_modelos_disponibles()
    modelos_con_datos = {k: v for k, v in disponibles.items() if v}

    if not modelos_con_datos:
        st.warning(
            "No hay ningún modelo entrenado todavía en `data/models/`. "
            "Ejecuta el Agente 3 (`python -m pv_models.main ...`) primero."
        )
    else:
        col1, col2 = st.columns(2)
        with col1:
            tipo_modelo = st.selectbox("Modelo", list(modelos_con_datos.keys()))
        with col2:
            horizonte = st.selectbox("Horizonte (horas)", modelos_con_datos[tipo_modelo])

        with st.spinner("Generando predicciones..."):
            df_pred = utils.generar_predicciones(ruta_dataset, tipo_modelo, horizonte)

        fecha_min = df_pred["timestamp_objetivo"].min().date()
        fecha_max = df_pred["timestamp_objetivo"].max().date()
        rango = st.slider(
            "Rango de fechas",
            min_value=fecha_min,
            max_value=fecha_max,
            value=(fecha_min, min(fecha_min + pd.Timedelta(days=14), fecha_max)),
        )

        mascara = (df_pred["timestamp_objetivo"].dt.date >= rango[0]) & (
            df_pred["timestamp_objetivo"].dt.date <= rango[1]
        )
        df_vista = df_pred.loc[mascara]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_vista["timestamp_objetivo"], y=df_vista["real"],
            name="Real (proxy)", line=dict(color="#2b8a3e"),
        ))
        fig.add_trace(go.Scatter(
            x=df_vista["timestamp_objetivo"], y=df_vista["prediccion"],
            name="Predicción", line=dict(color="#e8590c", dash="dot"),
        ))
        fig.update_layout(
            xaxis_title="Fecha",
            yaxis_title="Generación (MW, referencia 1 kWp)",
            hovermode="x unified",
            height=450,
        )
        st.plotly_chart(fig, width="stretch")

        mae_vista = (df_vista["real"] - df_vista["prediccion"]).abs().mean()
        st.metric("MAE en el rango seleccionado", f"{mae_vista:.6f} MW")

# ============================================================
# TAB 2 — Comparativa de modelos
# ============================================================
with tab_modelos:
    st.subheader("Comparativa de modelos y horizontes")

    if not any(utils.listar_modelos_disponibles().values()):
        st.warning("No hay modelos entrenados todavía.")
    else:
        with st.spinner("Evaluando todos los modelos disponibles..."):
            tabla_metricas = utils.evaluar_todos_los_modelos(ruta_dataset)

        if tabla_metricas.empty:
            st.warning("No se pudo evaluar ningún modelo con este dataset.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                fig_mae = px.bar(
                    tabla_metricas, x="horizonte_horas", y="mae", color="modelo",
                    barmode="group", title="MAE por horizonte",
                    labels={"mae": "MAE (MW)", "horizonte_horas": "Horizonte (h)"},
                )
                st.plotly_chart(fig_mae, width="stretch")
            with col2:
                fig_rmse = px.bar(
                    tabla_metricas, x="horizonte_horas", y="rmse", color="modelo",
                    barmode="group", title="RMSE por horizonte",
                    labels={"rmse": "RMSE (MW)", "horizonte_horas": "Horizonte (h)"},
                )
                st.plotly_chart(fig_rmse, width="stretch")

            st.dataframe(
                tabla_metricas.style.format({"mae": "{:.6f}", "rmse": "{:.6f}", "mape_dia": "{:.1f}%"}),
                width="stretch",
            )

            alguno_recomienda_reentrenar = tabla_metricas["recomendar_reentrenar"].any()
            if alguno_recomienda_reentrenar:
                st.error("⚠️ Al menos un modelo supera el umbral de degradación respecto a su referencia guardada (Agente 4).")
            else:
                st.success("✅ Todos los modelos están dentro del rendimiento esperado según la referencia guardada (Agente 4).")

        referencias = utils.cargar_metricas_referencia()
        with st.expander("Ver referencias guardadas (Agente 4)"):
            if referencias:
                st.json(referencias)
            else:
                st.caption("Todavía no se ha establecido ninguna referencia (`--establecer-referencia`).")

# ============================================================
# TAB 3 — Ramp events
# ============================================================
with tab_ramp:
    st.subheader("Ramp events detectados (histórico)")
    st.info(
        "⚠️ **Limitación documentada**: el sistema de aviso temprano (Agente 5) tiene baja "
        "precisión/recall con el enfoque actual (solo 1 de 9 eventos reales de 2024 anticipado, "
        "incluso relajando el umbral). Ver el resumen ejecutivo del proyecto para el análisis completo."
    )

    with st.spinner("Detectando ramp events en el histórico..."):
        eventos = utils.cargar_eventos_historicos(ruta_dataset)

    if eventos.empty:
        st.success("No se detectó ningún ramp event en este dataset con el umbral configurado.")
    else:
        st.metric("Ramp events detectados", len(eventos))

        fig_eventos = px.scatter(
            eventos, x="timestamp", y="caida_indice_cielo_despejado",
            size="magnitud_estimada_mw", color="confianza",
            title="Ramp events a lo largo del tiempo",
            labels={"timestamp": "Fecha", "caida_indice_cielo_despejado": "Caída de kt"},
            color_discrete_map={"alta": "#c92a2a", "media": "#e8590c", "baja": "#f08c00"},
        )
        st.plotly_chart(fig_eventos, width="stretch")

        st.dataframe(eventos, width="stretch")

# ============================================================
# TAB 4 — Contexto / ubicación
# ============================================================
with tab_contexto:
    st.subheader("Ubicación de referencia")
    lat, lon = utils.ubicacion_referencia()
    st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=9)
    st.caption(f"Vecindario / Santa Lucía, Gran Canaria — lat {lat}, lon {lon}")

    st.subheader("Arquitectura del sistema")
    st.markdown(
        """
        1. **Colector** — 4 APIs gratuitas → Parquet particionado
        2. **Feature engineer** — geometría solar, target de generación, envolvente empírica de cielo despejado
        3. **Modelos ML** — XGBoost (1-6h), LightGBM (6-48h), Prophet (estacionalidad)
        4. **Validador** — métricas vs. referencia guardada, recomienda reentrenar
        5. **Alertas** — detección de ramp events (histórico + aviso temprano)
        """
    )
    st.caption("Ver el resumen ejecutivo completo en el repositorio para el detalle de decisiones de diseño y bugs corregidos.")
