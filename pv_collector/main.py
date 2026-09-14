"""
Punto de entrada para ejecutar el Agente 1 - Colector.

Uso manual:
    python -m pv_collector.main --start 2026-09-01 --end 2026-09-08

Uso programado (ejecución horaria, recogiendo solo el día en curso):
    python -m pv_collector.main --hoy

Para la ejecución horaria automática, este script está pensado para
lanzarse desde APScheduler o un cron cada hora; ver comentario al
final del fichero con un ejemplo de integración con APScheduler.
"""

import argparse
from datetime import date

from .collector import ejecutar_coleccion


def main():
    parser = argparse.ArgumentParser(description="Agente 1 - Colector de datos FV")
    parser.add_argument("--start", type=str, help="Fecha inicio YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="Fecha fin YYYY-MM-DD")
    parser.add_argument(
        "--hoy", action="store_true", help="Recoger solo el día de hoy"
    )
    args = parser.parse_args()

    if args.hoy:
        hoy = date.today().isoformat()
        ejecutar_coleccion(hoy, hoy)
    elif args.start and args.end:
        ejecutar_coleccion(args.start, args.end)
    else:
        parser.error("Especifica --start/--end o usa --hoy")


if __name__ == "__main__":
    main()


# --- Ejemplo de integración con APScheduler para ejecución horaria ---
#
# from apscheduler.schedulers.blocking import BlockingScheduler
# from datetime import date
# from pv_collector.collector import ejecutar_coleccion
#
# scheduler = BlockingScheduler(timezone="UTC")
#
# @scheduler.scheduled_job("cron", minute=5)  # 5 min pasada cada hora
# def job_horario():
#     hoy = date.today().isoformat()
#     ejecutar_coleccion(hoy, hoy)
#
# scheduler.start()
