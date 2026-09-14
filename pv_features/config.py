"""
Configuración del Agente 2 - Feature engineer.

Reutiliza la ubicación del Agente 1 (mismo punto de referencia) y
añade los parámetros propios de las features: modelo térmico del
panel, lags y ventanas de media móvil.
"""

from pv_collector.config import LATITUDE, LONGITUDE, ELEVATION_M  # mismo punto de referencia

# --- Modelo térmico del panel (modelo NOCT simplificado) ---
# NOCT: Nominal Operating Cell Temperature, dato de ficha técnica típico
# de un panel FV estándar. 45°C es un valor representativo habitual.
PANEL_NOCT_C = 45.0
PANEL_NOCT_IRRADIANCIA_REF = 800  # W/m², irradiancia de referencia del NOCT
PANEL_NOCT_TEMP_AMBIENTE_REF = 20.0  # °C, temperatura ambiente de referencia del NOCT

# --- Lags de generación teórica (en horas) ---
LAGS_HORAS = [1, 24, 168]  # 1h, 1 día, 1 semana

# --- Ventanas de media móvil (en horas) ---
VENTANA_NUBOSIDAD_HORAS = 3

# --- Umbral de imputación ---
# Huecos de hasta este tamaño (en horas) se interpolan linealmente;
# huecos mayores se dejan como NaN para que el Agente 3 decida cómo
# tratarlos (no inventamos datos en huecos largos).
MAX_HUECO_INTERPOLABLE_HORAS = 3

# --- Índice de cielo despejado (kt) para el target de generación ---
# kt = GHI_real / GHI_cielo_despejado. Cerca del amanecer/atardecer el
# denominador es muy pequeño y el ratio puede dispararse por ruido
# numérico, así que se recorta a un rango físicamente razonable.
KT_MINIMO = 0.0
KT_MAXIMO = 1.2
