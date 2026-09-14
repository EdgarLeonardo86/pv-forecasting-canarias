"""
Envolvente empírica de cielo despejado, calculada a partir de los
propios datos históricos en vez de un modelo físico.

MOTIVACIÓN (decisión de diseño, tras varios intentos fallidos con
modelos físicos de pvlib -- ver historial en clearsky_target.py):
los modelos Ineichen y Haurwitz de pvlib dieron sesgos sistemáticos
demasiado grandes para Canarias (10-20% de horas de sol con el índice
de cielo despejado pegado al límite superior, frente al 2-5% esperable
por variabilidad meteorológica real). En vez de seguir ajustando un
modelo físico sin datos de calibración propios (turbidez real,
convención exacta de timestamp de Open-Meteo), se usa un enfoque
empírico estándar en el sector: el "envelope" o envolvente de cielo
despejado, calculado como un percentil alto (por defecto, 97) del GHI
real observado, agrupado por (mes, hora del día).

Idea: en un año completo, para cualquier franja concreta (por ejemplo,
"12:00 en julio") habrá varios días realmente despejados. El valor más
alto observado en esa franja es, por definición, una buena aproximación
del techo físico de esa franja, calibrado con las condiciones reales
del lugar (incluida la calima, la convención de timestamp de la fuente
de datos, etc.), sin depender de ningún modelo externo.

LIMITACIÓN a tener presente: la envolvente debe construirse UNA VEZ
con suficiente histórico (idealmente un año completo, para tener ~28-31
muestras por franja mes/hora) y luego reutilizarse -- no recalcularse
cada vez que se generan features para un rango pequeño, porque con
pocos días no hay muestras suficientes para que el percentil tenga
sentido estadístico.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("pv_features.clearsky_envelope")

RUTA_ENVOLVENTE = Path(__file__).resolve().parent.parent / "data" / "clearsky_envelope.parquet"

PERCENTIL = 0.97
MIN_MUESTRAS_POR_FRANJA = 10  # aviso si alguna franja mes/hora tiene menos que esto


def construir_y_guardar(df_historico: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la envolvente (percentil PERCENTIL de 'ghi' por mes y hora)
    a partir de un DataFrame histórico (se recomienda al menos un año
    completo) y la guarda en RUTA_ENVOLVENTE para uso posterior.

    Devuelve el DataFrame de la envolvente (columnas: mes, hora,
    ghi_envolvente, n_muestras).
    """
    if df_historico.empty or "ghi" not in df_historico.columns:
        raise ValueError("Se necesita un DataFrame con datos y columna 'ghi' para construir la envolvente")

    df = df_historico.copy()
    df["mes"] = df["timestamp"].dt.month
    df["hora"] = df["timestamp"].dt.hour

    envolvente = (
        df.groupby(["mes", "hora"])["ghi"]
        .agg(ghi_envolvente=lambda s: s.quantile(PERCENTIL), n_muestras="count")
        .reset_index()
    )

    franjas_pocas_muestras = envolvente[envolvente["n_muestras"] < MIN_MUESTRAS_POR_FRANJA]
    if not franjas_pocas_muestras.empty:
        logger.warning(
            f"{len(franjas_pocas_muestras)} franjas (mes, hora) tienen menos de "
            f"{MIN_MUESTRAS_POR_FRANJA} muestras -- el percentil ahí puede ser poco fiable. "
            f"Se recomienda construir la envolvente con al menos un año completo de histórico."
        )

    RUTA_ENVOLVENTE.parent.mkdir(parents=True, exist_ok=True)
    envolvente.to_parquet(RUTA_ENVOLVENTE, index=False, engine="pyarrow")
    logger.info(f"Envolvente de cielo despejado guardada en {RUTA_ENVOLVENTE} ({len(envolvente)} franjas)")

    return envolvente


def cargar() -> pd.DataFrame:
    """
    Carga la envolvente ya construida. Lanza un error claro si todavía
    no se ha construido, en vez de fallar más adelante con un error
    de fichero no encontrado poco informativo.
    """
    if not RUTA_ENVOLVENTE.exists():
        raise FileNotFoundError(
            "No existe la envolvente de cielo despejado todavía. "
            "Constrúyela primero con: python -m pv_features.main --construir-envolvente "
            "--start <fecha> --end <fecha> (usando idealmente un año completo de histórico)."
        )
    return pd.read_parquet(RUTA_ENVOLVENTE)


def aplicar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade la columna 'ghi_cielo_despejado' al DataFrame, tomando el
    valor de la envolvente correspondiente a cada fila según su mes y
    hora. Requiere que la envolvente ya esté construida (ver cargar()).
    """
    if df.empty:
        return df

    envolvente = cargar()

    df = df.copy()
    df["mes"] = df["timestamp"].dt.month
    df["hora"] = df["timestamp"].dt.hour

    df = df.merge(
        envolvente[["mes", "hora", "ghi_envolvente"]],
        on=["mes", "hora"],
        how="left",
    )
    df = df.rename(columns={"ghi_envolvente": "ghi_cielo_despejado"})
    df = df.drop(columns=["mes", "hora"])

    return df
