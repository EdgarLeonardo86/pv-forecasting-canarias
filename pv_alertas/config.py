"""
Configuración del Agente 5 - Alertas (detección de ramp events).
"""

from pathlib import Path

# Umbral de caída del índice de cielo despejado (kt) entre dos horas
# consecutivas para considerarlo un ramp event. kt va de 0 (muy
# nublado) a ~1.2 (cielo despejado con cloud enhancement); una caída
# de 0.4 significa pasar, por ejemplo, de un cielo bastante despejado
# (kt=0.9) a muy nublado (kt=0.5) en una sola hora.
UMBRAL_CAIDA_KT = 0.4

# Nivel de confianza (heurístico, no una probabilidad calibrada):
# cuanto más se aleje la caída del umbral, más "confianza" se asigna.
UMBRAL_CONFIANZA_ALTA = 2.0   # caída >= 2x el umbral -> confianza alta
UMBRAL_CONFIANZA_MEDIA = 1.2  # caída >= 1.2x el umbral -> confianza media
                                # por debajo -> confianza baja

RUTA_LOG_ALERTAS = Path(__file__).resolve().parent.parent / "data" / "alertas" / "alertas.jsonl"
