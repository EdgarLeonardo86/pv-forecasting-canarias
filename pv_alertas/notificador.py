"""
Notificación de alertas de ramp events.

Esta primera versión notifica por consola/log y deja un registro en
un fichero JSONL (una alerta por línea, fácil de consultar o volcar a
un dashboard más adelante). Está separada deliberadamente de
deteccion.py: quien detecta un ramp event no debería tener que saber
CÓMO se notifica -- así, añadir email o Telegram más adelante es
cuestión de implementar una función más aquí, sin tocar la lógica de
detección.

Para añadir un canal real en el futuro (ejemplo, Telegram):
- Crear una función `notificar_telegram(alerta: dict, bot_token: str,
  chat_id: str)` en este mismo módulo, usando la librería `requests`
  contra la API de Telegram (`https://api.telegram.org/bot<token>/sendMessage`).
- Llamarla desde `notificar(...)` junto a `notificar_consola(...)`,
  pasando el token/chat_id desde una variable de entorno (NUNCA
  hardcodeado en el código).
"""

import json
import logging
from datetime import datetime, timezone

from . import config

logger = logging.getLogger("pv_alertas.notificador")


def _formatear_mensaje(alerta: dict) -> str:
    hora = alerta["hora_prevista"]
    hora_str = hora.strftime("%Y-%m-%d %H:%M UTC") if hasattr(hora, "strftime") else str(hora)
    return (
        f"⚠️  RAMP EVENT previsto para {hora_str}\n"
        f"    Caída de índice de cielo despejado: {alerta['caida_indice_cielo_despejado']:.2f}\n"
        f"    Magnitud estimada: {alerta['magnitud_estimada_mw']:.6f} MW (referencia 1 kWp)\n"
        f"    Confianza: {alerta['confianza']}"
    )


def notificar_consola(alerta: dict) -> None:
    """Muestra la alerta por consola/log."""
    logger.warning(_formatear_mensaje(alerta))


def registrar_en_fichero(alerta: dict) -> None:
    """
    Añade la alerta como una línea JSON al fichero de registro
    (data/alertas/alertas.jsonl), creando la carpeta si no existe.
    """
    config.RUTA_LOG_ALERTAS.parent.mkdir(parents=True, exist_ok=True)

    alerta_serializable = dict(alerta)
    hora = alerta_serializable["hora_prevista"]
    alerta_serializable["hora_prevista"] = hora.isoformat() if hasattr(hora, "isoformat") else str(hora)
    alerta_serializable["fecha_registro"] = datetime.now(timezone.utc).isoformat()

    with open(config.RUTA_LOG_ALERTAS, "a") as f:
        f.write(json.dumps(alerta_serializable) + "\n")


def notificar(alerta: dict) -> None:
    """
    Punto de entrada único para notificar una alerta. Por ahora,
    consola + fichero; añadir más canales aquí (ver docstring del
    módulo) sin tocar deteccion.py.
    """
    notificar_consola(alerta)
    registrar_en_fichero(alerta)
