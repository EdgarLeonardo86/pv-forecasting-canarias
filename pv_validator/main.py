"""
Punto de entrada del Agente 4 - Validador.

Uso normal (evaluar un modelo/horizonte contra un dataset):
    python -m pv_validator.main --features data/features/2024-01-01_2024-12-31.parquet --modelo xgboost --horas 6

Establecer las métricas actuales como referencia (la primera vez que
se valida un modelo, o tras un reentrenamiento aceptado):
    python -m pv_validator.main --features data/features/2024-01-01_2024-12-31.parquet --modelo xgboost --horas 6 --establecer-referencia
"""

import argparse

from pv_models import dataset_prep
from . import validador


def main():
    parser = argparse.ArgumentParser(description="Agente 4 - Validador")
    parser.add_argument("--features", type=str, required=True, help="Ruta al parquet de features")
    parser.add_argument("--modelo", type=str, choices=["xgboost", "lightgbm"], required=True)
    parser.add_argument("--horas", type=int, required=True, help="Horizonte del modelo a evaluar")
    parser.add_argument(
        "--establecer-referencia",
        action="store_true",
        help="Guarda las métricas de esta evaluación como nueva referencia para comparaciones futuras",
    )
    args = parser.parse_args()

    df = dataset_prep.cargar_dataset(args.features)
    resultado = validador.evaluar_modelo(args.modelo, args.horas, df)

    if "error" in resultado:
        print(f"ERROR: {resultado['error']}")
        return

    print()
    print(f"=== Validación: {args.modelo} horizonte {args.horas}h ===")
    m = resultado["metricas_globales"]
    mape_str = f"{m['mape']:.1f}%" if m["mape"] is not None else "N/A"
    print(f"MAE:  {m['mae']:.6f}")
    print(f"RMSE: {m['rmse']:.6f}")
    print(f"MAPE (horas de día): {mape_str}")
    print(f"Filas evaluadas: {m['n_filas']} ({m['n_filas_dia']} de día)")
    print()
    print(f"¿Reentrenar? {'SÍ -- ' + resultado['motivo'] if resultado['recomendar_reentrenar'] else 'No'}")
    if not resultado["recomendar_reentrenar"] and resultado["motivo"]:
        print(f"({resultado['motivo']})")

    if args.establecer_referencia:
        validador.establecer_referencia(args.modelo, args.horas, m)
        print()
        print("Métricas guardadas como nueva referencia.")


if __name__ == "__main__":
    main()
