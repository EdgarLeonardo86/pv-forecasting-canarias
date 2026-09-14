"""
Punto de entrada para ejecutar el Agente 2 - Feature engineer.

Uso normal (requiere haber construido antes la envolvente de cielo
despejado, ver más abajo):
    python -m pv_features.main --start 2026-09-01 --end 2026-09-08

Construir/reconstruir la envolvente de cielo despejado (hacerlo una
vez, con el máximo histórico posible -- idealmente un año completo o
más -- y solo repetirlo si se quiere recalibrar con más datos):
    python -m pv_features.main --construir-envolvente --start 2024-01-01 --end 2024-12-31

Guarda el resultado en data/features/<start>_<end>.parquet
"""

import argparse
from pathlib import Path

from pv_collector import storage as storage_agente1
from . import clearsky_envelope
from .feature_engineer import construir_features

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "features"


def main():
    parser = argparse.ArgumentParser(description="Agente 2 - Feature engineer")
    parser.add_argument("--start", type=str, required=True, help="Fecha inicio YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="Fecha fin YYYY-MM-DD")
    parser.add_argument(
        "--construir-envolvente",
        action="store_true",
        help="Construye (o reconstruye) la envolvente empírica de cielo despejado a partir del rango dado, en vez de generar features",
    )
    args = parser.parse_args()

    if args.construir_envolvente:
        df_historico = storage_agente1.leer_rango(args.start, args.end)
        if df_historico.empty:
            print("No hay datos crudos en ese rango para construir la envolvente. Ejecuta primero el Agente 1.")
            return
        clearsky_envelope.construir_y_guardar(df_historico)
        print(f"Envolvente construida a partir de {args.start} -> {args.end} ({len(df_historico)} filas de entrada).")
        return

    df = construir_features(args.start, args.end)
    if df.empty:
        print("No se generaron features (dataset vacío).")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ruta_salida = OUTPUT_DIR / f"{args.start}_{args.end}.parquet"
    df.to_parquet(ruta_salida, index=False, engine="pyarrow")
    print(f"Guardado: {ruta_salida} ({len(df)} filas, {len(df.columns)} columnas)")


if __name__ == "__main__":
    main()
