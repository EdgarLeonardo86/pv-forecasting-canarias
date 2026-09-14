"""
Configuración del Agente 4 - Validador.
"""

from pathlib import Path

# Si el MAE actual es este factor (o más) peor que el MAE de
# referencia guardado, se recomienda reentrenar el modelo.
UMBRAL_DEGRADACION_RELATIVA = 1.5  # 50% peor que la referencia

# Valor mínimo del target por debajo del cual una fila se considera
# "de noche" a efectos de MAPE (evita divisiones por casi-cero que
# producen porcentajes de error absurdos y no informativos).
UMBRAL_TARGET_MAPE = 1e-6

RUTA_METRICAS_REFERENCIA = Path(__file__).resolve().parent.parent / "data" / "models" / "metricas_referencia.json"
