# Sistema Multiagente de Previsión Fotovoltaica en Canarias

**Resumen ejecutivo del proyecto**
Edgar — Septiembre 2026

---

## 1. Objetivo del proyecto

Diseñar y construir, de principio a fin, un sistema de agentes especializados que predicen la generación fotovoltaica horaria en Gran Canaria y detectan caídas bruscas de generación por nubosidad (*ramp events*), usando exclusivamente fuentes de datos públicas y gratuitas. El proyecto conecta tres líneas de trabajo personales: el TFB sobre predicción temprana de ramp events eólicos, la experiencia previa con dashboards de generación renovable vía la API de REE, y el objetivo de transición profesional hacia el análisis de datos en el sector energético.

## 2. Arquitectura: 5 agentes en pipeline

| Agente | Función | Estado |
|---|---|---|
| **1 — Colector** | Recoge datos horarios de 4 APIs externas, valida calidad, guarda en Parquet particionado | ✅ Completo |
| **2 — Feature engineer** | Construye el dataset de ML: geometría solar, modelo térmico, lags, y el target de generación | ✅ Completo |
| **3 — Modelos ML** | Tres modelos según horizonte: XGBoost (1-6h), LightGBM (6-48h), Prophet (estacionalidad) | ✅ Completo |
| **4 — Validador** | Evalúa modelos contra una referencia guardada y recomienda reentrenar si se degradan | ✅ Completo |
| **5 — Alertas** | Detecta ramp events (histórico y aviso temprano) y notifica | ✅ Completo, con limitación documentada |

**Stack técnico**: Python puro (sin frameworks de agentes), pandas, pvlib, XGBoost, LightGBM, Prophet, scikit-learn, PyArrow (Parquet).

## 3. Fuentes de datos

| Fuente | Qué aporta | Coste |
|---|---|---|
| Open-Meteo (forecast + archive) | Meteo horaria: GHI, DNI, DHI, temperatura, nubosidad, viento | Gratis |
| REE (apidatos.ree.es) | Generación FV y demanda de Canarias (grano diario) | Gratis |
| NASA POWER | Irradiancia histórica (con desfase de publicación) | Gratis |
| PVGIS | Producción FV teórica (climatología, año de referencia 2005-2020) | Gratis |

Ninguna requiere clave de API para el volumen usado.

## 4. Decisiones de diseño clave

- **Target de generación horaria sin contador real**: no existe generación FV horaria real y gratuita para una instalación de Canarias. Se construyó un proxy físicamente fundamentado: `generación_teórica (PVGIS) × índice de cielo despejado (kt)`, donde kt se calculó inicialmente con modelos físicos de pvlib (Ineichen, luego Haurwitz), y finalmente con una **envolvente empírica** (percentil 97 del GHI real por mes/hora), tras comprobar que los modelos físicos daban sesgos sistemáticos del 10-20% para esta ubicación.
- **REE como validación diaria, no como target horario**: se descubrió en producción que la API de REE solo admite grano diario para generación FV (no horario), así que se usa como reconciliación/contexto (`factor_correccion_diario`), no como fuente de la variable a predecir.
- **Sin data leakage por diseño**: se excluyen explícitamente del entrenamiento las columnas que filtrarían información del futuro (el total diario de REE, no conocido hasta las 23:59; NASA POWER, con desfase de publicación real).
- **Split cronológico, nunca aleatorio**: en series temporales, mezclar al azar infla artificialmente el rendimiento aparente del modelo.
- **Ramp events definidos sobre el índice de cielo despejado, no sobre generación bruta**: la caída de generación al atardecer es esperada y no debe confundirse con una anomalía real por nubes.

## 5. Bugs reales encontrados y corregidos

Este proyecto se validó de forma rigurosa contra APIs y datos reales en cada paso, lo que permitió detectar y corregir más de 15 problemas concretos que no habrían aparecido en un desarrollo puramente teórico:

**Agente 1 (Colector):**
- Falta el parámetro `geo_trunc=electric_system`, obligatorio en la API de REE.
- El widget de generación de REE solo admite grano diario, no horario.
- El endpoint de demanda usado no existía (`demanda-tiempo-real` → `evolucion`).
- PVGIS solo admite años 2005-2020; se corrigió reproyectando un año de referencia.
- Open-Meteo necesita dos endpoints distintos (forecast vs. archive) según la antigüedad de la fecha.
- Errores de límites en filtros de rango de fechas (excluían la última hora del día).

**Agente 2 (Feature engineer):**
- El modelo Ineichen de pvlib dio un sesgo del 15,8% de horas con índice de cielo despejado saturado; Haurwitz lo redujo al 10,9%; ninguno resolvió el problema del todo.
- Un ajuste de desfase temporal (+30 min) empeoró el resultado, descartando la hipótesis de desalineación de timestamp.
- **Solución final**: envolvente empírica (percentil 97 por mes/hora), que redujo el sesgo al 0% de saturación y 3,0% de horas con kt>1 — dentro del rango físicamente esperado.

**Agente 3 (Modelos ML):**
- **Leakage crítico en XGBoost**: la columna `target` se colaba como feature de sí misma (el modelo veía la respuesta antes de predecirla). Detectado revisando `feature_importances_`.
- **Desalineación día/noche en LightGBM**: el filtro de "horas de día" usaba el momento de emisión de la predicción en vez del momento objetivo, dando resultados artificialmente perfectos en horizontes de 12h/36h.
- **Prophet con estacionalidad anual mal identificada**: con solo 1 año de histórico, Prophet avisó de que el componente anual quedaba mal identificado; se desactivó a favor del regresor determinista de PVGIS, mejorando el MAE en 50x.

**Agente 5 (Alertas):**
- No es un bug de código, sino un hallazgo de investigación: se comprobó rigurosamente (con análisis de sensibilidad al umbral) que el enfoque de aviso temprano basado en persistencia/lags tiene baja precisión y recall genuinos, no resolubles ajustando el umbral — ver sección de limitaciones.

## 6. Resultados

### Agente 3 — Modelos de predicción

Todos los modelos baten claramente a un baseline de persistencia ingenua ("va a hacer lo mismo que ahora"):

- **XGBoost (1-6h)**: MAE de 2-7x mejor que el baseline, con degradación progresiva y esperada según el horizonte.
- **LightGBM (6-48h)**: MAE de 1,4-3x mejor que el baseline, consistente en todos los horizontes tras corregir la evaluación día/noche.
- **Prophet**: tras las correcciones, MAE del mismo orden de magnitud que los modelos de árboles, cumpliendo su rol de capturar estacionalidad y festivos.

### Agente 5 — Detección de ramp events

- **9 ramp events reales** confirmados en 2024 (caída de kt ≥ 0,30, horas de día).
- El sistema de aviso temprano (XGBoost 1h) solo anticipó **1 de 9** eventos reales, con una tasa de falsas alarmas alta (72 alertas para 1 acierto con umbral relajado).

## 7. Limitaciones y trabajo futuro

1. **El target es un proxy, no una medición real**: construido a partir de irradiancia y climatología, no de un contador de una instalación FV real. Los resultados deben interpretarse con esa salvedad.
2. **El aviso temprano de ramp events tiene baja precisión/recall con el enfoque actual.** Se demostró mediante análisis de sensibilidad que el problema no es de calibración de umbral, sino estructural: predecir un frente de nubes súbito requiere información que la persistencia de condiciones recientes no contiene. Líneas de mejora futuras:
   - Incorporar imágenes satelitales de nubosidad (por ejemplo, productos de EUMETSAT).
   - Usar salidas de un modelo numérico de predicción meteorológica (NWP) en vez de solo lags.
   - Explorar un modelo de clasificación específico para ramp events, en vez de derivarlos de un modelo de regresión general.
3. **Solo 1 año de histórico**: limita la fiabilidad de la estacionalidad anual (afectó directamente a Prophet) y del percentil de la envolvente de cielo despejado en franjas con pocas muestras.
4. **Sistema no desplegado en continuo**: el Colector se ejecuta manualmente; las alertas se notifican por consola/log, no por email/Telegram real. Ambos son extensiones naturales una vez el sistema funcione de forma autónoma (scheduler horario).

## 8. Valor para el portfolio / TFB

Más allá del resultado final, el proceso documentado aquí —encontrar y corregir sistemáticamente más de 15 problemas reales mediante validación rigurosa contra datos y APIs reales, en vez de asumir que el código funciona porque compila— es en sí mismo la contribución más defendible del proyecto. La conclusión honesta sobre las limitaciones del Agente 5, respaldada con un análisis cuantitativo de sensibilidad, es exactamente el tipo de rigor metodológico que se espera en un trabajo académico o en un rol de analista de datos senior.
