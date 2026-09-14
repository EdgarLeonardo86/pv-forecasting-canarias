"""
Punto de entrada para entrenar el Agente 3 (XGBoost 1-6h o LightGBM 6-48h).

Uso:
    python -m pv_models.main --features data/features/2024-01-01_2024-12-31.parquet --modelo xgboost
    python -m pv_models.main --features data/features/2024-01-01_2024-12-31.parquet --modelo lightgbm
"""

import argparse
import logging

from . import dataset_prep, xgboost_model, lightgbm_model, prophet_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("pv_models.main")

MODULOS = {
    "xgboost": xgboost_model,
    "lightgbm": lightgbm_model,
}


def main():
    parser = argparse.ArgumentParser(description="Agente 3 - Entrenamiento de modelos")
    parser.add_argument("--features", type=str, required=True, help="Ruta al parquet de features del Agente 2")
    parser.add_argument(
        "--modelo",
        type=str,
        choices=["xgboost", "lightgbm", "prophet"],
        default="xgboost",
        help="Qué modelo entrenar (xgboost: 1-6h; lightgbm: 6-48h; prophet: estacionalidad)",
    )
    args = parser.parse_args()

    logger.info(f"Cargando dataset: {args.features}")
    df = dataset_prep.cargar_dataset(args.features)

    if args.modelo == "prophet":
        resultado = prophet_model.entrenar_y_evaluar(df)
        prophet_model.guardar_modelo(resultado)

        if resultado.get("modelo") is None:
            print(f"ERROR: {resultado.get('error')}")
            return

        print()
        print("=== Resumen de resultados (prophet) ===")
        print(f"MAE global:        {resultado['mae']:.6f}")
        print(f"RMSE global:       {resultado['rmse']:.6f}")
        print(f"MAE baseline media: {resultado['mae_baseline_media']:.6f}")
        if resultado["mae_dia"] is not None:
            print(f"MAE horas de día:  {resultado['mae_dia']:.6f}")
            print(f"RMSE horas de día: {resultado['rmse_dia']:.6f}")
        print(f"Train: {resultado['n_train']}, Test: {resultado['n_test']}")
        return

    modulo = MODULOS[args.modelo]

    resultados = modulo.entrenar_todos_horizontes(df)
    modulo.guardar_modelos(resultados)

    print()
    print(f"=== Resumen de resultados ({args.modelo}, todas las horas) ===")
    print(f"{'Horizonte':<12}{'MAE':<14}{'RMSE':<14}{'MAE baseline':<16}{'Train':<10}{'Test':<10}")
    for horas, resultado in resultados.items():
        if resultado.get("modelo") is None:
            print(f"{horas}h{'':<10}ERROR: {resultado.get('error')}")
            continue
        baseline = resultado["mae_baseline_persistencia"]
        baseline_str = f"{baseline:.6f}" if baseline is not None else "N/A"
        print(
            f"{horas}h{'':<10}{resultado['mae']:<14.6f}{resultado['rmse']:<14.6f}"
            f"{baseline_str:<16}{resultado['n_train']:<10}{resultado['n_test']:<10}"
        )

    print()
    print(f"=== Resumen de resultados ({args.modelo}, SOLO horas de día) ===")
    print(f"{'Horizonte':<12}{'MAE_dia':<14}{'RMSE_dia':<14}{'MAE baseline_dia':<18}")
    for horas, resultado in resultados.items():
        if resultado.get("modelo") is None:
            continue
        mae_dia = resultado.get("mae_dia")
        rmse_dia = resultado.get("rmse_dia")
        baseline_dia = resultado.get("mae_baseline_persistencia_dia")
        if mae_dia is None:
            print(f"{horas}h{'':<10}N/A (sin columna es_de_dia)")
            continue
        baseline_dia_str = f"{baseline_dia:.6f}" if baseline_dia is not None else "N/A"
        print(f"{horas}h{'':<10}{mae_dia:<14.6f}{rmse_dia:<14.6f}{baseline_dia_str:<18}")


if __name__ == "__main__":
    main()
