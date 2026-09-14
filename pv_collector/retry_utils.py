"""
Utilidades de reintento compartidas por todos los clientes de API.

Usa tenacity para no reinventar la lógica de backoff exponencial.
Instalar con: pip install tenacity
"""

import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import requests

from . import config

logger = logging.getLogger("pv_collector")


def api_retry(func):
    """
    Decorador estándar de reintentos para llamadas a APIs externas.

    - Reintenta solo en errores de red/timeout/HTTP, no en errores de
      parseo o de datos (esos deben fallar rápido y visiblemente).
    - Backoff exponencial entre RETRY_WAIT_MIN_SECONDS y
      RETRY_WAIT_MAX_SECONDS, hasta MAX_RETRIES intentos.
    """
    return retry(
        stop=stop_after_attempt(config.MAX_RETRIES),
        wait=wait_exponential(
            multiplier=1,
            min=config.RETRY_WAIT_MIN_SECONDS,
            max=config.RETRY_WAIT_MAX_SECONDS,
        ),
        retry=retry_if_exception_type(
            (requests.ConnectionError, requests.Timeout, requests.HTTPError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )(func)
