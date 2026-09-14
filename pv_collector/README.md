# Agente 1 — Colector

Recoge datos horarios de meteo, generación real e irradiancia teórica
de cuatro fuentes gratuitas, valida su calidad y los guarda en Parquet
particionado por fecha, listos para el Agente 2 (feature engineer).

## Estructura

```
pv_collector/
├── config.py           # ubicación, rutas, parámetros de cada API
├── retry_utils.py       # decorador de reintentos con backoff
├── validation.py        # huecos, rangos, duplicados, flag de calidad
├── storage.py            # escritura/lectura Parquet particionado
├── collector.py          # orquestador: combina las 4 fuentes
├── main.py                # punto de entrada CLI / scheduler
└── clients/
    ├── open_meteo.py      # meteo horaria
    ├── ree.py              # generación FV real + demanda (Canarias)
    ├── nasa_power.py       # irradiancia histórica
    └── pvgis.py            # producción FV teórica
```

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
# Recoger un rango de fechas
python -m pv_collector.main --start 2026-09-01 --end 2026-09-08

# Recoger solo el día de hoy (para ejecución horaria)
python -m pv_collector.main --hoy
```

O desde código:

```python
from pv_collector.collector import ejecutar_coleccion

df = ejecutar_coleccion("2026-09-01", "2026-09-08")
print(df.head())
```

## Notas importantes

- **Calidad de dato por fila:** la columna `calidad_dato` distingue `ok`,
  `parcial` (falta alguna fuente concreta en esa hora, pero no todas),
  `imputado` (hueco temporal completo, ninguna fuente aportó dato) y
  `sospechoso` (algún valor fuera de rango físico). La columna
  `fuentes_faltantes` lista qué fuente(s) faltan en cada fila (p.ej.
  `"nasa_power"`), para poder ver de un vistazo qué está incompleto sin
  inspeccionar columna por columna.
- **Ubicación de referencia:** Vecindario/Santa Lucía, Gran Canaria
  (lat 27.913, lon -15.546). Cambiar en `config.py` si se necesita otro punto.
- **Open-Meteo tiene dos endpoints distintos:** `/v1/forecast` solo cubre
  pronóstico y unos pocos días de histórico reciente; para fechas más
  antiguas (histórico de entrenamiento) hace falta `/v1/archive` (datos
  de reanálisis ERA5). El cliente elige automáticamente uno u otro según
  lo antigua que sea `end_date` (margen de 5 días).
- **REE solo da grano diario, no horario:** el widget `estructura-generacion`
  de la API pública de REE no admite `time_trunc=hour` (se descubrió probando
  contra la API real). Por eso `ree.py` descarga el dato diario y lo expande
  repitiendo el mismo valor en las 24 horas de ese día, en las columnas
  `generacion_real_diaria_mwh` y `demanda_total_diaria_mwh`. **Estas columnas
  son de referencia/validación, no generación horaria real** — para el target
  horario de los modelos de ML usa PVGIS o un proxy calculado con pvlib a
  partir del GHI de Open-Meteo. El Agente 4 (validador) es el lugar natural
  para contrastar el total diario agregado de PVGIS/pvlib contra este dato
  oficial de REE.
- **PVGIS trabaja por año completo**, no por rango de fechas arbitrario,
  **y solo admite años entre 2005 y 2020** para esta ubicación (descubierto
  probando contra la API real). Como el valor es una producción TEÓRICA
  (climatología de referencia, no medición del año en curso), el cliente
  fija `PVGIS_REFERENCE_YEAR = 2020` y reproyecta el calendario (mismo
  mes/día/hora) al año realmente solicitado. Es una aproximación razonable
  para un baseline teórico, pero no es la irradiancia real de ese año.
- **NASA POWER tiene varios días de desfase** respecto al presente —
  úsalo para histórico/relleno, no para el dato más reciente.
- **Tolerancia a fallos por fuente:** si una API falla, se registra el
  error y la colección continúa con las demás fuentes en vez de abortar.
- Antes de usar en producción, revisa los nombres exactos de los
  endpoints de REE (`REE_GEO_IDS`, tipos de tecnología en la respuesta
  de generación) contra una llamada real, ya que la API de REE puede
  variar ligeramente el formato de respuesta según el desglose pedido.

## Próximo paso

Con esto tienes el Agente 1 completo. El siguiente paso natural es el
**Agente 2 — Feature engineer**, que lee de `storage.leer_rango(...)`
y construye el dataset listo para ML (ángulo solar con pvlib, lags,
medias móviles, `temp_panel`, etc.).
