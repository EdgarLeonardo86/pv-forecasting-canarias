# Agente 5 — Alertas (ramp events)

Detecta caídas bruscas de generación por nubosidad (ramp events),
usando el índice de cielo despejado (kt) en vez de la generación en
bruto -- la caída natural del atardecer no es un ramp event, una
caída repentina de kt durante el día sí lo es.

## Estructura

```
pv_alertas/
├── config.py         # umbral de caída de kt, niveles de confianza
├── deteccion.py         # lógica de detección (histórica y predictiva)
├── notificador.py         # consola + fichero de registro (extensible)
└── main.py                 # CLI (dos modos: historico y backtest)
```

## Dos modos de uso

**Modo histórico** — cuenta y lista los ramp events YA OCURRIDOS en un
periodo, comparando kt real hora a hora (útil para análisis, informes,
o para conectar con el TFB: "¿cuántos ramp events hubo en 2024?"):

```bash
python -m pv_alertas.main --modo historico --features data/features/2024-01-01_2024-12-31.parquet
```

**Modo backtest** — simula el AVISO TEMPRANO de verdad: recorre el
histórico como si fuera tiempo real, usando el modelo XGBoost de 1h ya
entrenado (Agente 3) para predecir la hora siguiente y alertar ANTES
de que ocurra la caída, no después:

```bash
python -m pv_alertas.main --modo backtest --features data/features/2024-01-01_2024-12-31.parquet
```

Requiere tener ya entrenado `data/models/xgboost_h1.json` (Agente 3).

## Notificaciones

Esta primera versión notifica por **consola/log y fichero**
(`data/alertas/alertas.jsonl`, una alerta por línea en JSON). Se
decidió así deliberadamente para esta fase del proyecto: no hay
todavía un proceso corriendo en continuo (el Colector se ejecuta a
mano), y añadir email/Telegram real implicaría manejar credenciales
sensibles sin necesidad todavía.

El diseño deja el punto de extensión preparado: `notificador.py` tiene
un docstring explicando cómo añadir un canal real (por ejemplo,
Telegram) más adelante sin tocar `deteccion.py`.

## Decisiones de diseño

- **Ramp event = caída de kt, no de generación bruta.** La generación
  cae cada atardecer de forma esperada; eso no es una anomalía. Una
  caída del índice de cielo despejado sí aísla el efecto real de las
  nubes.
- **La confianza es una heurística simple** (según cuánto se aleje la
  caída del umbral), no una probabilidad calibrada estadísticamente.
  Se declara así explícitamente para no sobrevender la fiabilidad.
- **La teórica de la hora objetivo es determinista** (PVGIS,
  climatología), así que se puede conocer de antemano sin necesitar
  ningún pronóstico -- solo la generación total predicha por el
  modelo depende de la incertidumbre meteorológica real.

## Limitación conocida: baja precisión/recall del aviso temprano

Se hizo un análisis explícito de sensibilidad al umbral sobre 2024
completo, cruzando el modo histórico (eventos reales confirmados)
contra el modo backtest (avisos emitidos con 1h de antelación):

| Umbral kt | Eventos reales (histórico) | Alertas emitidas (backtest) | Aciertos exactos |
|---|---|---|---|
| 0.40 (original) | 2 | 15 | 0 |
| 0.30 (más sensible) | 9 | 72 | 1 |

**Conclusión importante**: bajar el umbral multiplica las falsas
alarmas (15 → 72) sin mejorar apenas el recall (0 → 1 de 9 eventos
reales). Esto indica que el problema **no es de calibración del
umbral**, sino una limitación estructural del enfoque: el modelo de
1h se apoya en lags y en el estado meteorológico actual (persistencia
de condiciones recientes), y un ramp event genuino -- un frente de
nubes que llega de forma súbita -- por definición no se anuncia con
antelación en esas variables. Para un aviso temprano fiable haría
falta una fuente de datos que capture la dinámica atmosférica de
mayor alcance (imágenes satelitales de nubosidad, o un modelo numérico
de predicción meteorológica real), fuera del alcance de las APIs
gratuitas usadas en este proyecto.

Se mantiene el umbral original (0.4) por ser más conservador (menos
ruido de falsas alarmas), documentando esta limitación como línea de
trabajo futura en vez de perseguir una calibración que los datos ya
muestran que no va a resolver el problema de fondo.

## Próximo paso

Con los 5 agentes funcionando (Colector → Feature engineer → Modelos
ML → Validador → Alertas), el siguiente paso natural del proyecto
original es el **dashboard de visualización** (Streamlit o Power BI),
y en producción real, conectar el modo backtest a una ejecución en
continuo (scheduler horario) en vez de simularlo sobre histórico.
