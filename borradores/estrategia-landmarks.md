# Estrategia por hitos · puntos faciales (landmarks) y geometría

Idea de Vincent (2026-11-19): ir **por partes, iterando**. Cada hito se deja cerrado y medido antes de empezar el siguiente; si uno funciona, se intenta el próximo; si no, se documenta el resultado negativo y se para ahí. No sabemos si llegaremos al final, y eso está bien: lo importante es que lo que quede esté terminado.

**Reglas comunes**
- Cada hito es una rama `trabajo/<tema>` y un PR pequeño, con su línea en `bitacora.md` y su experimento en `experimentos.md` (pregunta, método, dato, muestra, semilla y métrica, **definidos antes de correr**; los resultados malos también se anotan).
- Solo visión clásica (D11). Se mide siempre contra un **piso trivial** (ver M1): un modelo que no lo supera no cuenta como logro.
- Cada módulo se puede apagar sin romper la cámara ni los filtros.
- Tiempos: son estimaciones mías, no compromisos; la fecha límite (≈ fin de noviembre) sigue sin confirmar con el profesor.
- Fotos de personas solo locales y con permiso; nada de capturas al repo.

| Hito | Qué se hace | Se mide con | Puerta para pasar al siguiente |
|---|---|---|---|
| **M0 · Entrada: cara principal sólida** | Detector y evaluación de D13 funcionando en la app; E7/E11 repetidos con la regla nueva | Recall de la cara principal y falsos positivos por imagen, con la meta que fije el equipo | El equipo declara el detector «estable». Sin esto no se empieza M1 (los puntos se predicen sobre la cara que entrega el detector) |
| **M1 · Datos y piso** (**hecho 2026-11-19**: `src/keypoints_datos.py`, E23; piso NME 0,148; división 70/15/15 en vez de 85/15 para guardar un `test`) | Descargar Facial Keypoints (pedir visto bueno; ficha completa y SHA-256), dividir 85/15 el `training`, decidir qué hacer con las filas con puntos faltantes (filtrar o imputar, justificado), y medir el **piso**: predecir siempre la forma media | NME del piso (normalizado por distancia entre ojos), RMSE en píxeles | Lector de datos reproducible y piso medido |
| **M2 · Pocos puntos** | 4 puntos (centro de cada ojo, nariz, centro de la boca) con HOG o LBP + regresor (Ridge, ExtraTrees o SVR). Comparar con el detector Haar de ojos y, si el profesor lo permite, con el predictor de 68 puntos de dlib (árboles de regresión) | NME por punto contra el piso y contra la línea base de Haar | El modelo supera claramente el piso en validación. Si no, se para aquí y se documenta |
| **M3 · Los 15 puntos** | Mismo método extendido a los 15 puntos; probar PCA sobre la forma y un segundo paso que refina cada punto | NME, RMSE por punto y % dentro de tolerancia; por tamaño de cara | Mejor que M2 en los 4 puntos comunes |
| **M4 · De 96×96 a la cámara** | Recortar la cara con el detector, pasar a 96×96 en gris, predecir y dibujar los puntos en la app (casilla apagable). Medir cuánto cambia el resultado de laboratorio a webcam | Cualitativo con capturas consentidas en local + FPS con y sin el módulo | La cámara sigue fluida (límite de FPS que fije el equipo) y el módulo se apaga sin romper nada |
| **M5 · Geometría 2D** | Medidas (distancia entre ojos, inclinación, ojos–nariz, apertura de boca) y orientación cualitativa (frontal, perfil izquierdo/derecho, arriba/abajo) con reglas sobre los puntos | Para las medidas: coherencia con los puntos. Para la orientación **no tenemos etiquetas propias**: hay que decidir si se usan fotos consentidas en local o el atributo de pose de WIDER (solo dice típica/atípica) | Definir antes la verdad de referencia; si no hay, se presenta como demostración, no como métrica |
| **M6 · Pose 3D (opcional)** | `solvePnP` con un modelo facial 3D genérico y cámara calibrada | Error contra una referencia conocida | Solo si M1–M5 quedaron sólidos y sobra tiempo |

**Qué no se hace dentro de esta estrategia:** atributos, expresiones y efectos son líneas aparte; no bloquean los hitos de arriba ni dependen de ellos (salvo que usan el mismo detector).

**Preguntas abiertas**
1. ¿El profesor acepta como «clásico» el predictor de 68 puntos de dlib (árboles de regresión)? Mismo trámite que el HOG de dlib.
2. ¿Qué meta de M0 y qué límite de FPS fija el equipo? Yo no pongo cifras sin base.
3. ¿Quién responde por cada hito? (la reunión del equipo).
