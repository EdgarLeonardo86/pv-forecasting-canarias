"""
Configuración central del Agente 1 - Colector.

Todas las constantes de ubicación, rutas de almacenamiento y parámetros
de las APIs externas viven aquí para que los clientes no tengan
valores 'hardcodeados' repartidos por el código.
"""

from pathlib import Path

# --- Ubicación de referencia: Vecindario / Santa Lucía, Gran Canaria ---
LATITUDE = 27.913
LONGITUDE = -15.546
ELEVATION_M = 15  # aproximada, ajustar si se conoce el valor exacto del punto

# --- Rutas de almacenamiento ---
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
LOG_DIR = BASE_DIR / "logs"

# --- Open-Meteo ---
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"  # para fechas pasadas (>~5 días)
OPEN_METEO_HOURLY_VARS = [
    "shortwave_radiation",       # -> ghi
    "direct_normal_irradiance",  # -> dni
    "diffuse_radiation",         # -> dhi
    "temperature_2m",            # -> temp_ambiente
    "cloud_cover",                # -> nubosidad
    "wind_speed_10m",            # -> viento_velocidad
]

# --- REE (Red Eléctrica de España) ---
REE_BASE_URL = "https://apidatos.ree.es/es/datos"
REE_GENERATION_ENDPOINT = "/generacion/estructura-generacion"
REE_DEMAND_ENDPOINT = "/demanda/evolucion"  # 'demanda-tiempo-real' no existe como widget real
REE_TIME_TRUNC = "day"  # 'estructura-generacion' no admite granularidad horaria
REE_GEO_TRUNC = "electric_system"  # obligatorio siempre que se envíe geo_limit/geo_ids
REE_GEO_LIMIT = "canarias"
REE_GEO_IDS = 8742  # id geográfico de Canarias en la API de REE

# --- NASA POWER ---
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
NASA_POWER_PARAMETERS = ["ALLSKY_SFC_SW_DWN"]  # GHI histórico
NASA_POWER_COMMUNITY = "re"  # renewable energy

# --- PVGIS ---
PVGIS_URL = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
PVGIS_PEAKPOWER_KW = 1.0  # kWp de referencia para la simulación teórica
PVGIS_LOSS_PCT = 14.0     # pérdidas del sistema por defecto (%)
PVGIS_REFERENCE_YEAR = 2020  # PVGIS solo admite años 2005-2020; se usa como año de referencia y se reproyecta el calendario

# --- Reintentos ---
MAX_RETRIES = 4
RETRY_WAIT_MIN_SECONDS = 2
RETRY_WAIT_MAX_SECONDS = 30

# --- Validación ---
MAX_GAP_HOURS = 2  # huecos mayores a esto se marcan como sospechosos
