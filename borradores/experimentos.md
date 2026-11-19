# Experimentos · deteccion-rostros

E1, E2, … con configuración y resultado. (Los resultados E1, E3, E5 y E6 del proyecto anterior, cuyo repositorio se eliminó, están resumidos en `antecedentes-proyecto-anterior.md`.)


## E7 · 2026-11-14 · Detectores clásicos con el alcance de D1 (caras ≥ 48 px de alto)
- Pregunta: con el alcance acordado, ¿cuánto recall y precisión dan Haar, Haar con giros y perfiles, dlib HOG (ampliación x1 y x2) y la fusión dlib+Haar?
- Datos: 300 imágenes al azar (semilla 1) de WIDER en formato YOLO (`entradas/wider-yolo`, solo para desarrollo, nunca la prueba final). Ancho máximo 800 px, IoU ≥ 0,5.
- Regla de conteo (D1): una cara menor de 48 px no suma al recall ni su acierto cuenta como falso positivo; solo la detección que no acierta ninguna cara cuenta como falso positivo.
- Código: `src/evaluar_caras.py` (cambiado hoy: ruta nueva, regla de D1, detector `fusion`). Salida cruda: `borradores/e7-bruto.log`.
- Resultado parcial: Haar 0°: recall 0,452, precisión 0,747 (48–96 px: 0,426; ≥96 px: 0,484), 0,05 s/imagen.
- Haar con giros y perfiles: recall 0,476, precisión 0,257 (849 falsos positivos), 0,79 s/imagen: los giros casi no suben el recall.
- dlib HOG x1: recall 0,478, precisión 0,955 (14 falsos positivos), 0,14 s/imagen. dlib HOG x2: recall 0,463, precisión 0,919, 0,52 s/imagen (ampliar x2 ya no ayuda con caras ≥48 px).
- Fusión dlib+Haar con giros: recall 0,496, precisión 0,262 (857 falsos positivos), 1,12 s/imagen. Sube el recall solo 0,018 sobre dlib x1 y arrastra los falsos positivos de Haar.
- Conclusión: dlib x1 es el mejor equilibrio (0,478 / 0,955, 0,14 s). Ningún método probado pasa de ≈0,50 de recall en WIDER con caras ≥48 px; para subirlo hace falta algo nuevo (p. ej. HOG+SVM propio con negativos difíciles, filtro de segunda etapa para podar falsos positivos de Haar). Se completa esta entrada al terminar.
- Límite: una sola muestra y una semilla; sirve para comparar métodos, no como cifra final.

## E8 · 2026-11-14 · Por qué la app no veía las caras del fondo (prueba de aula, cámara del portátil)
- Observación (Vincent, captura de una clase): se escondían las 3 caras más cercanas; varias personas del fondo (≈ 55–100 px de alto en el video de 1920 px) no se detectaban.
- Causa: la app reducía el video a 480 px de ancho antes de detectar (`ANCHO_DET = 480`); una cara de 55 px queda de ≈ 14 px, por debajo del mínimo de Haar (24 px).
- Prueba sobre la propia captura (1908×1088, solo para diagnóstico): a 480 px, 0 caras nuevas; a 960 y 1280 px, Haar con giros encuentra las 2 personas con lentes del fondo; dlib HOG x1 no las encuentra a ninguna escala (miran hacia abajo, no son frontales). Tiempos Haar con giros: 0,53 s (480), 1,31 s (960), 2,03 s (1280), 3,05 s (1908).
- Cambio hecho: `ANCHO_DET = 1280` y la detección pasa a un hilo aparte para que el video no se trabe (`hilo_deteccion` en `src/app_blur.py`).
- Resultado de la prueba en vivo: la detección tarda 0,4 s al inicio y ≈ 3,3 s después, y el video cae a ≈ 2,9 cuadros/s. Inaceptable (y el equipo se calentó mucho).
- Medición del costo real (sobre la captura del aula, 960 px): una detección frontal+perfil cuesta ≈ 0,6 s de procesador (aunque solo ≈ 0,1 s de reloj, porque OpenCV la reparte entre todos los núcleos). A 5 detecciones por segundo eran ≈ 3 núcleos llenos; esa fue la causa del recalentamiento. Los giros (13 ángulos × 2 cascadas) son lo más caro.
- Cambios hechos: (1) frontal a 960 px en cada vuelta, sin giros; (2) giros solo a 480 px y cada 2 vueltas; (3) perfil cada 3 vueltas; (4) tope de ≈ 2,5 detecciones por segundo (`INTERVALO = 0.4`); (5) video limitado a 20 cuadros/s; (6) el desenfoque se calcula sobre la región reducida a 1/4.
- Resultado (la captura del aula como fuente, sin cámara): 2 caras del fondo detectadas y escondidas, video a 16 cuadros/s, detección 54 ms de reloj, procesador ≈ 150–220 % (antes 270–420 %). Falta probar con la cámara real y con movimiento. No se midió recall: es una sola imagen de diagnóstico.
- Revisión de una captura real con la versión liviana (2026-11-14, `salidas_demo/captura-20261114-090250.jpg`, solo local; pixelado 64×64; aula con ≈ 8 personas): se esconden la cara cercana, la del borde izquierdo y la de la derecha. Fallos: (1) dos caras del fondo izquierdo sin esconder (≈ 55 px, con lentes rojos, y ≈ 40 px); (2) cajas demasiado grandes: la de la cara cercana cubre ≈ 800 px de ancho para una cara de < 400 px (margen 25 % + suavizado); (3) un recuadro grande a la derecha cubre pared y mesa (posible falso positivo o caja inflada; falta confirmarlo con «Mostrar cajas»). Pendiente de mejoras: recall en caras de 48–60 px, tamaño de caja, falsos positivos.

## E9 · 2026-11-14 · Revisor v1 (parte B): dónde y qué tipo de alteración
- Pregunta: con solo visión clásica, ¿se puede localizar una cara alterada y decir si es desenfoque, pixelado o tapado?
- Método (`src/revisor.py`): etapa 1, parches de 32×32 px (paso 16) → «alterado / natural» con un bosque aleatorio (150 árboles) sobre 18 rasgos: log de la varianza del Laplaciano, gradiente medio, fracción de píxeles planos, desviación local, 9 «razones de cuadrícula» (diferencia entre vecinos dentro de bloques frente a la de sus bordes, para bloques 4–64 alineados al (0,0)), su mínimo, y 4 rasgos de contexto (cada rasgo menos el promedio de los vecinos a 32 px). Los parches «alterados» son los que están completamente dentro de una región; los «naturales», los que no tocan ninguna. Los dudosos se ignoran al entrenar. Las zonas alteradas se unen en cajas (cierre morfológico + componentes conexos; la caja se amplía 16 px). Etapa 2: cada caja → tipo con otro bosque (300 árboles) sobre rasgos de la región entera (nitidez dentro y fuera, color plano, colores distintos, razones de cuadrícula).
- Datos: `datos/alteradas/` (ver `procesamiento-dataset.md`); entrenamiento con 1.500 de las 3.000 imágenes; se evalúa en validación (600). Prueba: sin tocar.
- Métrica: una región real cuenta como localizada si una caja predicha tiene IoU ≥ 0,5; después se mira si el tipo es correcto. Falsas = cajas que no aciertan ninguna región.
- Resultado (validación, 300 imágenes, 505 regiones; entrenado con 1.500 imágenes; `borradores/e9-bruto.log`): localiza 326 (recall 0,646) pero con 1.155 cajas falsas (precisión 0,220); tipo correcto en 80,7 % de las localizadas; 55 de 84 imágenes sin edición disparan alarma; 0,42 s por imagen. Por tipo (localizadas / reales): desenfoque 87/154 (tipo correcto 85), pixelado 139/200 (127), tapado 100/151 (51: 46 tapados se leen como desenfoque).
- Lectura: el bosque entrenado ve más desenfoques que las reglas, pero llena de falsas alarmas y confunde el tapado. Sirve como comparación (D6).

## E10 · 2026-11-14 · Revisor por análisis directo de píxeles, sin entrenar (método principal, D6)
- Método (`src/analisis_pixeles.py`): (1) píxeles «planos» = rango en 3×3 ≤ 6 niveles; (2) componentes planos rectangulares (relleno ≥ 0,8, lado ≥ 4) = bloques candidatos; (3) pixelado = bloques cuadrados de lado b ∈ {4…64} alineados a la cuadrícula del (0,0) y vecinos entre sí, con ≥ 2 colores distintos, ≥ 6 bloques válidos si b < 24 y ≥ 1 si b ≥ 24; se informa b; (4) tapado = rectángulo plano de ≥ 40×40 que no pertenece a una cuadrícula, con contraste ≥ 20 niveles frente a sus 6 px de alrededor y sin tocar el borde de la imagen; (5) desenfoque = zona rectangular (relleno ≥ 0,8, lado ≥ 40) donde la nitidez (log de la energía del Laplaciano en 15×15) cae ≥ 1,2 respecto de su entorno (gaussiana de σ = 24).
- Ajuste honesto: los umbrales de las reglas 3–5 se ajustaron en las primeras 150 imágenes de validación y se comprobaron en las otras 150 (pixelado+tapado: precisión 0,32 → 0,60 y recall 0,65 → 0,59 en las 150 de comprobación; desenfoque: 253 → 31 falsas alarmas con 23 → 20 aciertos). Prueba: sin tocar.
- Resultado final en validación (300 imágenes, 505 regiones, `borradores/e10-bruto.log`): localiza 294 (recall 0,582), 165 falsas (precisión 0,641); tipo correcto en 89,1 % de las localizadas; recall con tipo correcto 0,519; 18 de 84 imágenes sin edición disparan alarma; 0,05 s por imagen. Por tipo (localizadas / reales): pixelado 137/200 (tipo correcto 135; tamaño de bloque correcto en 81,5 %), tapado 103/151 (90 con tipo correcto), desenfoque 54/154 (37).
- Comparación con E9 (mismas 300 imágenes): las reglas tienen mejor precisión (0,64 vs 0,22), mejor tipo (0,89 vs 0,81), 7× menos falsas, 8× más rápidas y se pueden explicar; el modelo entrenado gana solo en recall de desenfoque (0,56 vs 0,35) y de lo demás empata. Recall total con tipo correcto: 0,519 vs 0,521.
- Puntos débiles: el desenfoque (una caída de nitidez no distingue un desenfoque artificial de un fondo desenfocado por la cámara) y los pixelados/tapados que no se localizan (≈ 30 %: región pegada al borde, bloques de colores casi iguales, cara de piel lisa). Vincent aceptó sostener lo del desenfoque en la exposición.
- Límites: las imágenes son sintéticas y generadas con el mismo código que la app; una sola corrida; los umbrales se ajustaron en validación (no en prueba). Pendiente: analizar los no localizados, probar con una captura real de la app, y la evaluación final única en prueba.

## E10b · 2026-11-14 · Revisor por píxeles: correcciones tras revisar capturas reales de la app
- Hallazgos en 4 capturas reales (`salidas_demo/`, solo local): (1) falsos «tapado» en lámparas del techo (franjas de ≈ 12×200 px) y pantallas negras; (2) bloques pixelados unidos con zonas lisas vecinas (cajas demasiado grandes); (3) la cara de la izquierda aparece con dos pixelaciones superpuestas (la app tenía dos cajas casi iguales sobre la misma cara), lo que rompe la cuadrícula; (4) muchos «desenfoque» de 48×48 sobre bloques lisos de la misma pixelación cuando esta no se reconoce.
- Cambios: filtro de proporción ancho/alto 0,35–1,8 (los datos de caras: 0,42–1,4, percentiles 1–99); el pixelado solo une componentes que encajan en la misma cuadrícula (origen alineado, lados ≤ b o múltiplos de b, para bloques vecinos de color casi igual que se funden).
- Resultado en validación (300 imágenes; antes → después): falsas 165 → 133; precisión 0,641 → 0,692; recall de localización 0,582 → 0,592; imágenes sin edición con alarma 18 → 13; tipo correcto 0,891 → 0,890. Una variante intermedia (sin permitir múltiplos de b) empeoró: pixelados localizados 137 → 95.
- Pendiente: en las capturas reales el pixelado de 64 px sigue partido en pedazos (algunos componentes empiezan a ≈ 10 px de la línea de la cuadrícula); tras unir cajas duplicadas en la app hay que volver a capturar y medir. Los umbrales siguen ajustados en validación; prueba sin tocar.

## E8b · 2026-11-14 · Regresión que introduje en la app y su arreglo
- Síntoma (capturas de Vincent): su cara, cerca de la cámara con gorra y lentes, no se escondía; sí la de otras personas.
- Causa: al bajar la detección a ≈ 2,5 por segundo (E8) dejé la votación en «5 de las últimas 9 detecciones», es decir ≈ 2 s quieto para confirmar una cara.
- Arreglo: se esconde desde la primera detección (D2: falso negativo peor) y se retiene 4 detecciones (≈ 1,6 s) tras perderla; entre detecciones las cajas siguen a la cara con flujo óptico (Lucas-Kanade sobre 16 puntos por caja); y las cajas que se solapan ≥ 50 % se unen (antes se pixelaba dos veces la misma cara). Probado con una foto y con pruebas unitarias del seguimiento; falta repetir con la cámara.
- Revisión de 3 capturas nuevas (2026-11-14 09:31–09:32, desenfoque en óvalo, solo local): se esconden la cara cercana de Vincent (óvalo de ≈ 530–590 px para una cara de ≈ 300 px: caja inflada) y 3–5 caras de fondo. Fallos: (1) en cada captura quedan 4–6 caras visibles de ≈ 50–70 px, varias con lentes (izquierda del aula y filas del fondo); (2) un óvalo de ≈ 250 px sobre una zona sin cara (falso positivo o caja con retraso); (3) cajas mal dimensionadas. El revisor por píxeles no encuentra nada en estas capturas (0 regiones): esperado, porque el óvalo con desenfoque suave no cumple «rectángulo con caída brusca de nitidez» (relleno 0,79 < 0,8); es el punto débil ya declarado.

## E11 · 2026-11-14 · Segunda etapa para el detector de caras: candidatas permisivas + HOG + SVM propio
- Motivo: en capturas reales quedaban 4–6 caras visibles de 50–70 px (varias con lentes), y en WIDER el recall se estancaba en ≈ 0,5 (E7). Hipótesis: dejar los detectores permisivos (más candidatas, también falsas) y que una segunda etapa entrenada decida, sube el recall sin hundir la precisión.
- Candidatas (`src/verificador.py`): Haar frontal alt2 y alt (escala 1,05, vecinos 1), Haar de perfil y de perfil espejado (vecinos 2) y dlib HOG con umbral −0,4 (x1). Se guardan fuente y puntaje de cada una.
- Segunda etapa: HOG de 64×64 del recorte (margen 15 %; 1.764 rasgos) + fuente, puntaje de dlib, log del alto, y acuerdo (cuántos detectores distintos proponen la misma caja) → SVM lineal (C = 0,05, clases balanceadas, rasgos normalizados). Positivo = candidata con IoU ≥ 0,5 con una cara en alcance (≥ 48 px de alto); negativo = IoU < 0,2 con cualquier cara (negativos difíciles: las falsas alarmas de los propios detectores); el resto se ignora al entrenar.
- Datos: entrenamiento = 1.500 imágenes del conjunto train de `datos/division_wider.csv` (semilla 1); evaluación = 300 de val (semilla 2). Prueba: sin tocar. Misma regla de conteo que E7 (D1).
- Resultado (validación, 300 imágenes, 746 caras en alcance; `borradores/e11-bruto.log`; entrenamiento con 16.517 candidatas de 1.500 imágenes: 7.381 caras y 9.136 falsas):
  - Candidatas permisivas por separado (recall / precisión): Haar alt2 0,524 / 0,296; Haar alt 0,519 / 0,421; perfil 0,221 / 0,477; perfil espejado 0,241 / 0,499; **dlib con umbral −0,4: 0,552 / 0,808**. Todas juntas, sin filtro: recall 0,630 (techo de la segunda etapa) y precisión 0,238.
  - Con segunda etapa (umbral del SVM → recall / precisión): −1,0 → 0,591 / 0,666; −0,5 → 0,578 / 0,727; 0,0 → 0,564 / 0,770; 0,5 → 0,555 / 0,801.
  - Referencia E7 (300 imágenes distintas, semilla 1): dlib x1 con su umbral por defecto 0,478 / 0,955.
- Lectura: (1) la mayor ganancia viene de bajar el umbral de dlib (recall 0,478 → 0,552, precisión 0,955 → 0,808), no de la segunda etapa; (2) el SVM propio sube el recall unos 0,04 más a cambio de precisión (F1 ≈ 0,64–0,66, igual que dlib permisivo solo: 0,656), así que como filtro no aporta mucho más que «dlib permisivo»; (3) el techo es el recall de las candidatas (0,63): lo que ningún detector propone, no lo rescata el filtro. Para subirlo hay que mejorar el origen (más resolución, giros, perfiles) y no el filtro.
- Límites: una sola muestra de 300 imágenes (comparaciones con E7 son de muestras distintas); WIDER es difícil (poses extremas, oclusión); no se midió el efecto en vivo. Prueba: sin tocar.
- Siguiente paso propuesto: usar dlib con umbral bajo (y el resto de candidatas como respaldo) en la app y comprobarlo con capturas reales.

## E12 · 2026-11-19 · Detectores clásicos sobre la «cara principal» (D13), por dataset
**Pregunta:** ¿qué tan bien encuentran hoy los detectores que ya tenemos la cara principal (la más grande, ≥ 96 px), en imágenes con 1 a 3 caras, y a qué costo?
**Método:** `src/evaluar_principal.py val 400 1 …`: 400 imágenes de validación de cada dataset (semilla 1) del conjunto de `conjunto_principal.py`; imágenes reducidas a 800 px de ancho; IoU ≥ 0,5; las caras etiquetadas de < 48 px no cuentan como falso positivo. Detectores sin entrenar con nuestros datos. Salida cruda: `borradores/e12-bruto.log`. **Prueba distinta a E7/E11 (regla de D1): no se comparan las cifras.**
**Resultado (recall de la cara principal / falsos positivos por imagen / precisión / segundos por imagen):**

| Detector | WIDER | Open Images |
|---|---|---|
| Haar frontal | 0,583 / 0,36 / 0,69 / 0,34 s | 0,532 / 0,33 / 0,70 / 0,26 s |
| Haar con giros y perfil | 0,603 / 1,66 / 0,34 / 3,5 s | 0,647 / 1,40 / 0,39 / 2,7 s |
| dlib HOG ×0 | 0,660 / 0,03 / 0,97 / 0,05 s | 0,605 / 0,04 / 0,95 / 0,04 s |
| dlib HOG ×1 | 0,652 / 0,05 / 0,95 / 0,20 s | 0,605 / 0,07 / 0,92 / 0,16 s |
| dlib HOG ×2 | 0,655 / 0,10 / 0,90 / 0,81 s | 0,608 / 0,15 / 0,85 / 0,67 s |
| dlib HOG ×1, umbral −0,4 | **0,725** / 0,28 / 0,78 / 0,20 s | **0,675** / 0,21 / 0,82 / 0,16 s |
| dlib HOG ×2, umbral −0,4 | 0,720 / 0,49 / 0,67 / 0,80 s | 0,682 / 0,48 / 0,67 / 0,67 s |

**Qué se ve:**
- Ningún detector clásico encuentra la cara principal más de ~73 % (WIDER) y ~68 % (Open Images) de las veces: **faltan 3 de cada 10 caras grandes**. Ahí está el margen de mejora.
- Bajar el umbral de dlib (−0,4) sube el recall ≈ 6–7 puntos a cambio de más falsos positivos (0,2–0,3 por imagen). Subir la escala (×2) casi no ayuda y cuesta 4 veces más tiempo.
- dlib ×0 con umbral por defecto es el más limpio y rápido (precisión 0,95–0,97; 0,05 s), pero pierde más caras.
- Haar con giros y perfil es inusable: 1,4–1,7 falsos positivos por imagen y 3–3,5 s por imagen.
- Open Images sale algo peor que WIDER con dlib (caras más variadas); que las cifras sean parecidas indica que el conjunto nuevo no es trivialmente más fácil.
**Límites:** 400 imágenes por dataset: el error estadístico del recall es de unos ±2,5 puntos (1 desviación), así que diferencias de 1–2 puntos no significan nada. Un solo corte de validación y una semilla. No se probó con webcam.
**Siguiente:** entrenar un HOG+SVM propio sobre este conjunto con negativos difíciles y compararlo con `dlib ×1, umbral −0,4` (el mejor hasta ahora).

## E13 · 2026-11-19 · HOG+SVM propio con minado de negativos difíciles (cara principal)
**Pregunta:** ¿un detector HOG+SVM entrenado por nosotros con WIDER + Open Images supera a dlib?
**Método:** `src/hog_svm_cara.py entrenar 3000 1500 2 1`: ventana 64×64 (HOG 9 orientaciones, celdas de 8 px), positivos = caras ≥ 48 px con 12,5 % de margen y su espejo (7.580), negativos = ventanas sin cara (23.999 al inicio), SVM lineal (C = 0,01), 2 rondas de minado de falsos positivos sobre 1.500 imágenes de entrenamiento distintas. Umbral escogido en una muestra de validación (semilla 2, 300 por dataset). Logs: `borradores/e13-entrenamiento.log`, `borradores/e13-umbral-bruto.log`.
**Resultado:** el minado se agotó: ronda 1 → 29.944 falsos positivos nuevos; ronda 2 → solo 375 (54.318 negativos al final). En validación, el umbral 0 da recall de la cara principal 0,523 (WIDER) / 0,487 (Open Images) con precisión 0,84 / 0,90; el umbral −0,25 llega a 0,683 / 0,653 pero con 1,3–1,7 falsos positivos por imagen. **No supera a dlib** (0,725 / 0,675 con 0,2–0,3 falsos positivos).
**Lectura:** un solo filtro de 64×64 entrenado con ~3.000 imágenes no alcanza a un detector entrenado con muchos más datos y varias vistas. Sirve como fuente de candidatas del detector combinado (E14), no como detector final. No se probó con más datos, otro tamaño de ventana ni otro C.

## E14 · 2026-11-19 · Detector combinado: varias fuentes + árboles con parada temprana + tope de 3 cajas
**Pregunta:** ¿se puede subir el recall de la cara principal sin disparar los falsos positivos combinando fuentes clásicas?
**Método:** `src/detector_fusion.py`. Candidatas de 7 fuentes (dlib HOG con umbral −1 sobre la imagen normal, con CLAHE, espejada y rotada ±15°; Haar frontal; HOG+SVM de E13), agrupadas por solapamiento (IoU > 0,4). Segundo paso: `HistGradientBoostingClassifier` (árboles con boosting) con **parada temprana** (128 de 500 árboles), 14 rasgos por grupo (puntaje por fuente, número de fuentes, puntaje del HOG+SVM sobre el grupo, tamaño y posición relativos, proporción). Entrenado con 2.000 imágenes de entrenamiento que E13 no vio (47.416 grupos; 2.448 caras). Tope de 3 cajas por imagen (D13). Umbrales (0,2 y 0,1) escogidos en una muestra de validación (semilla 2, 300 por dataset); medición final en **toda** la validación (804 WIDER + 1.549 Open Images). Logs: `borradores/e14-*.log`.
**Resultado (recall de la cara principal / recall de secundarias / falsos positivos por imagen / precisión / s por imagen):**

| Detector | WIDER (804) | Open Images (1.549) |
|---|---|---|
| dlib HOG ×0 | 0,629 / 0,446 / 0,04 / 0,95 / 0,06 | 0,602 / 0,466 / 0,05 / 0,94 / 0,05 |
| dlib HOG ×1, umbral −0,4 (mejor de E12) | 0,698 / 0,607 / 0,35 / 0,73 / 0,25 | 0,675 / 0,589 / 0,23 / 0,81 / 0,21 |
| **Combinado, umbral 0,2** | **0,756** / 0,646 / 0,18 / 0,85 / 1,9 | **0,750** / 0,624 / 0,17 / 0,86 / 1,6 |
| **Combinado, umbral 0,1 (modo privacidad)** | **0,775** / 0,670 / 0,33 / 0,76 / 2,0 | **0,759** / 0,645 / 0,29 / 0,79 / 1,7 |

**Lectura:** el combinado con umbral 0,2 gana 6–7 puntos de recall de la cara principal contra el mejor dlib **y** tiene menos falsos positivos; con umbral 0,1 gana más recall al mismo costo en falsos positivos. El error estadístico es de ≈ ±1,5 puntos (WIDER) y ±1,1 (Open Images), así que la ganancia es real. Aun así faltan ~1 de cada 4 caras principales.
**Límites:** los umbrales se escogieron en una muestra que forma parte de la validación (optimismo pequeño: se eligieron 2 valores); **no se ha tocado `test`**. Es lento: 1,6–2 s por imagen en 10 procesos (la app necesita ≈ 0,4 s). No se midió qué fuente aporta cuánto. No se probó con webcam.
**Siguiente:** quitar las fuentes que menos aportan para acelerar, mirar las caras que siguen fallando (lentes, perfiles, tamaño) y probar rasgos nuevos en el segundo paso.

## E15 · 2026-11-19 · Ablación del detector combinado y versiones rápida y equilibrada
**Pregunta:** ¿qué fuente de candidatas aporta de verdad, y se puede acelerar el detector de E14 (1–2 s por imagen) sin perder recall?
**Método:** `src/ablacion_fusion.py`. Se guardan una vez las candidatas de 2.000 imágenes de entrenamiento (las que E13 no vio) y de toda la validación (2.353) en `datos/cache_fusion.pkl` (fuera de git). Para cada subconjunto de fuentes se reagrupa, se reentrena el segundo paso (árboles con parada temprana, igual que E14) y se mide en validación: recall de la cara principal y falsos positivos por imagen, con tope de 3 cajas. Después, dos versiones se miden de punta a punta con `evaluar_principal.py`. Logs: `borradores/e15-*.log`.
**Tiempo por fuente (s por imagen de 800 px):** cada pasada de dlib ≈ 0,25–0,27 (hay cinco: normal, CLAHE, espejo, rotación +15°, −15°); Haar 0,08; HOG+SVM 0,05.
**Ablación (umbral 0,2; recall de la cara principal / falsos positivos por imagen; WIDER · Open Images):**

| Fuentes | WIDER | Open Images |
|---|---|---|
| Todas (E14) | 0,755 / 0,17 | 0,749 / 0,17 |
| Sin una fuente cualquiera (siete pruebas) | 0,746–0,762 / 0,15–0,18 | 0,733–0,748 / 0,16–0,19 |
| dlib + espejo + HOG+SVM | 0,752 / 0,21 | 0,745 / 0,19 |
| dlib + Haar + HOG+SVM | 0,749 / 0,16 | 0,726 / 0,18 |
| dlib normal + HOG+SVM | 0,740 / 0,19 | 0,716 / 0,19 |
| **Solo dlib normal** (con árboles y tope de 3) | 0,724 / 0,17 | 0,704 / 0,18 |

**Versiones (medición de punta a punta en toda la validación; tiempo en un solo proceso):**

| Versión | Fuentes | WIDER | Open Images | Tiempo |
|---|---|---|---|---|
| Rápida (`modelos/fusion_rapido.pkl`) | dlib + Haar + HOG+SVM | 0,746 / 0,18 | 0,726 / 0,18 | **0,30 s** |
| Equilibrada (`modelos/fusion_equilibrado.pkl`) | dlib + espejo + HOG+SVM | 0,754 / 0,20 | 0,745 / 0,18 | 0,42 s |
| Completa (`modelos/fusion.pkl`) | las siete | 0,756 / 0,18 | 0,750 / 0,17 | 1,02 s |
| dlib −0,4 (referencia) | — | 0,698 / 0,35 | 0,675 / 0,23 | 0,19 s |

**Lectura:** (1) Ninguna fuente es imprescindible: quitar cualquiera cuesta menos de 1,5 puntos (dentro del error estadístico). (2) **Gran parte de la mejora viene del segundo paso y del tope de 3 cajas, no de las fuentes extra**: solo dlib normal con árboles ya supera a dlib −0,4 solo, con la mitad de falsos positivos. (3) La versión rápida cabe en el presupuesto de la app (≈ 0,4 s por detección) y gana 4–5 puntos de recall frente a dlib −0,4 con la mitad de falsos positivos.
**Límites:** los modelos y umbrales se eligieron con la misma validación (optimismo pequeño); `test` sin tocar. Con 10 procesos en paralelo los tiempos salen ~3 veces mayores (competencia por CPU). Los falsos positivos de la ablación usan una regla de caras pequeñas aproximada, pero las versiones finales usan el evaluador oficial. No se probó con webcam.

## E16 · 2026-11-19 · Rasgos nuevos para el segundo paso y más datos de entrenamiento
**Pregunta:** ¿el segundo paso (árboles con parada temprana) mejora con 11 rasgos más (nitidez, contraste, brillo y bordes dentro de la caja; lugar de cada candidata por tamaño y por puntaje frente a las demás de la imagen) o con 3 veces más imágenes de entrenamiento (2.000 → 6.834)?
**Método:** `src/ablacion_fusion.py medir …+extra`, con `CACHE_ENTRENAMIENTO=datos/cache_fusion_grande.pkl` para el entrenamiento grande. Misma validación completa (804 + 1.549). Una primera corrida mezcló por error todas las variantes en una sola (el shell no separó los argumentos); se repitió bien. Logs: `borradores/e16-*.log`.
**Resultado (versión rápida, umbral 0,2; recall de la cara principal WIDER · Open Images):** sin rasgos nuevos 0,749 · 0,726 (2.000 imágenes) y 0,750 · 0,728 (6.834); con rasgos nuevos 0,756 · 0,743 (2.000) y 0,756 · 0,740 (6.834). Las 7 fuentes con rasgos nuevos: 0,767 · 0,765 (2.000) y 0,772 · 0,770 (6.834), contra 0,755 · 0,749 sin ellos (E15).
**Lectura:** los rasgos nuevos suben ≈ 0,7–1,7 puntos (en el borde del error estadístico, ±1,5); **3 veces más datos casi no cambia nada** (+0,1 a +0,2): los datos no son el cuello de botella. Se conservan los rasgos nuevos (no cuestan casi tiempo) y se entrena con 2.000 imágenes.
**Análisis de fallos:** con las 7 fuentes, a la cara principal la encuentra alguna candidata en 0,873 (WIDER) y 0,852 (Open Images) de los casos (el «techo»); el detector acierta 0,75: ≈ 12 puntos se pierden en el segundo paso. De las 332 caras principales sin ninguna candidata (de 2.353), un mosaico de 24 al azar (visto solo en local) mostró: perfiles o caras de espaldas, cabezas giradas casi 90°, caras cortadas por el borde, caras tapadas (casco, gorra, manos) y **2 etiquetas que cubren una multitud entera** (ruido del dataset, no se puede arreglar). Eso llevó a E17.

## E17 · 2026-11-19 · Cuatro fuentes nuevas de candidatas (perfil, borde reflejado, giros de 90°)
**Pregunta:** ¿se pueden encontrar las caras que ninguna fuente ve (perfiles, cortadas, giradas)?
**Método:** `src/detector_fusion.py` (`NUEVAS`): `haar_perfil` (cascada de perfil de OpenCV, imagen y su espejo), `dlib_pad` (borde reflejado 20 %: una cara cortada queda «completa» al otro lado del espejo), `dlib_rot_90` y `dlib_rot_270` (imagen girada 90° a cada lado; el mapeo de coordenadas se verificó girando 43 fotos: 43/43 en cada sentido). `ablacion_fusion.py agregar` suma solo las fuentes nuevas al caché. Se reentrena el segundo paso (con rasgos nuevos) para cada combinación y se mide en la validación completa; las versiones finales se miden de punta a punta con `evaluar_principal.py`. Logs: `borradores/e17-*.log`.
**Techo de candidatas:** 0,873 → **0,897** (WIDER) y 0,852 → **0,873** (Open Images) con las 11 fuentes. Caras que solo encuentran las nuevas: 25 el Haar de perfil, 18 y 14 las rotaciones de 90°, 4 el borde reflejado.
**Aporte de cada fuente nueva a la versión rápida (umbral 0,2; recall WIDER · Open Images):** base (con rasgos nuevos) 0,756 · 0,743; **+ Haar de perfil 0,770 · 0,768**; + borde reflejado 0,757 · 0,744; + rotaciones de 90° 0,761 · 0,744 (cuestan ≈ 0,5 s); las 11 fuentes 0,781 · 0,790.
**Versiones finales (validación completa; umbral 0,2 salvo indicación; recall de la cara principal / falsos positivos por imagen; tiempo en un solo proceso):**

| Versión | Fuentes | WIDER | Open Images | Tiempo |
|---|---|---|---|---|
| Rápida (app) | dlib + Haar + HOG+SVM + Haar de perfil | 0,765 / 0,22 | 0,765 / 0,20 | **0,36 s** |
| Rápida, umbral 0,1 | igual | 0,785 / 0,38 | 0,776 / 0,34 | 0,36 s |
| Equilibrada | + dlib espejado | 0,781 / 0,18 | 0,771 / 0,18 | 0,53 s |
| Completa | las 11 | 0,782 / 0,19 | 0,783 / 0,19 | 1,87 s |
| (E15, versión rápida anterior) | dlib + Haar + HOG+SVM | 0,746 / 0,18 | 0,726 / 0,18 | 0,30 s |
| (dlib −0,4, referencia) | — | 0,698 / 0,35 | 0,675 / 0,23 | 0,19 s |

**Lectura:** la mejor mejora por lo que cuesta es el Haar de perfil (+1,4 y +2,5 puntos por ≈ 0,06 s). Frente a dlib −0,4, la versión rápida gana **+6,7 y +9 puntos** de recall con menos falsos positivos. El borde reflejado y las rotaciones de 90° casi no sirven en el conjunto de la cara principal (se dejan en la versión completa). Queda ≈ 1 de cada 4 caras principales sin encontrar (parte son etiquetas ruidosas).
**Límites:** modelos y umbrales escogidos con la misma validación (optimismo pequeño); error estadístico ≈ ±1,5 puntos; `test` sin tocar; no se probó con cámara real; las cifras de segundos de `evaluar_principal.py` salen ~3 veces mayores por la competencia entre procesos (los de la tabla son de un solo proceso). Durante la medición aparecieron dos errores de código míos (un choque de nombres y la compatibilidad con modelos viejos) que se corrigieron antes de las cifras finales.

## E18 · 2026-11-19 · Corrección de cajas aprendida por fuente
**Pregunta:** ¿hay caras principales que sí se encuentran pero con la caja mal ajustada (IoU < 0,5), y se puede corregir?
**Hallazgo previo (solo mirando el caché de validación):** el 6,6 % de las caras principales tiene como mejor candidata una con IoU entre 0,3 y 0,5 (encontrada pero mal ajustada) y otro 3,9 % entre 0,1 y 0,3. Las cajas tienen un **sesgo sistemático**: las de dlib salen ≈ 1,09× más anchas y ≈ 0,82× más bajas que la etiqueta y ≈ 0,05 alturas más abajo; las de Haar de perfil, ≈ 1,4× más anchas. Es igual en WIDER y Open Images y en entrenamiento y validación (las etiquetas tienen proporción ancho/alto de 0,74–0,78; los detectores dan cajas casi cuadradas).
**Método:** `src/ablacion_fusion.py correccion` mide, **solo en entrenamiento**, para cada fuente el desplazamiento del centro y el cociente caja/etiqueta de ancho y alto (candidatas con IoU ≥ 0,4 con la cara principal) y lo guarda en `datos/correccion_cajas.json` (versionado: son 11 filas de números, sin imágenes). `detector_fusion.corregir()` rehace cada caja antes de agrupar. La tabla se guarda dentro de cada modelo. Se reentrena el segundo paso con las cajas corregidas. Logs: `borradores/e18-*.log`.
**Resultado en caché (validación completa, umbral 0,2; recall de la cara principal / falsos positivos por imagen, WIDER · Open Images):** versión rápida sin corrección 0,770 / 0,20 · 0,768 / 0,20; con corrección 0,775 / 0,17 · 0,781 / 0,18. Las 11 fuentes: 0,781 / 0,19 · 0,790 / 0,20 → 0,802 / 0,16 · 0,798 / 0,17.
**Versiones finales (detector real, validación completa; tiempo en un solo proceso):**

| Versión | WIDER | Open Images | Tiempo |
|---|---|---|---|
| Rápida, umbral 0,2 | 0,771 / 0,18 | 0,776 / 0,18 | 0,34 s |
| **Rápida, umbral 0,1 (la de la app)** | 0,784 / 0,30 | 0,788 / 0,28 | 0,34 s |
| Equilibrada, umbral 0,2 | 0,784 / 0,18 | 0,789 / 0,18 | 0,54 s |
| Completa (11 fuentes), umbral 0,2 | **0,806** / 0,16 | **0,795** / 0,17 | 1,88 s |
| (E17, rápida sin corrección, umbral 0,2) | 0,765 / 0,22 | 0,765 / 0,20 | 0,36 s |
| (dlib −0,4, referencia) | 0,698 / 0,35 | 0,675 / 0,23 | 0,19 s |

**Lectura:** la corrección sube el recall unos +0,5 a +2 puntos y **baja los falsos positivos** (menos cajas «casi bien» contadas como error), sin costo de tiempo. Frente a dlib −0,4, la versión rápida gana **+7 y +11 puntos** de recall con la mitad de falsos positivos o menos. Ya se encuentran ≈ 4 de cada 5 caras principales; el techo de candidatas (0,897 / 0,873) deja poco margen más sin fuentes nuevas.
**Límites:** la tabla viene del conjunto de entrenamiento de 2.000 imágenes; modelos y umbrales escogidos con la misma validación; error estadístico ≈ ±1,5 puntos; `test` sin tocar; sin cámara real. Para que los equipos no lo olviden: **si cambias las fuentes, repite `correccion` y reentrena** (la tabla y el modelo van juntos).

## E19 · 2026-11-19 · Hiperparámetros del segundo paso (resultado negativo)
**Pregunta:** ¿el segundo paso (`HistGradientBoostingClassifier`) mejora cambiando sus hiperparámetros?
**Método:** `HGB="…" src/ablacion_fusion.py medir dlib+haar+hogsvm+haar_perfil+extra+corr` sobre el caché (2.000 imágenes de entrenamiento; validación completa). Cinco configuraciones frente a la de referencia (hojas por defecto, tasa 0,05, parada temprana). Log: `borradores/e19-bruto.log`.
**Resultado (umbral 0,2, recall de la cara principal / falsos positivos, WIDER · Open Images):** referencia 0,775 / 0,17 · 0,781 / 0,18; 15 hojas 0,777 · 0,780; 63 hojas 0,767 · 0,779; regularización L2 y hojas mínimas 0,777 · 0,779; tasa 0,1 y profundidad 6: 0,777 · 0,781; tasa 0,03 y más iteraciones: 0,776 · 0,782. Con umbral 0,1 las diferencias son iguales de pequeñas (0,785–0,795).
**Lectura:** ninguna configuración supera a la de referencia por más que el error estadístico (≈ ±1,5 puntos). **El segundo paso ya no depende de sus hiperparámetros**; lo que queda por ganar está en las candidatas, los rasgos o las etiquetas ruidosas. Se conserva la configuración por defecto.
**Límites:** una sola semilla; seis configuraciones comparadas en la misma validación; no se probó otro tipo de modelo (bosque aleatorio, regresión logística) ni calibración de probabilidades.

## E20 · 2026-11-19 · Ancho de trabajo del detector (velocidad contra recall)
**Pregunta:** ¿hace falta trabajar a 800 px de ancho, o se puede bajar para ganar velocidad sin perder la cara principal?
**Método:** `ANCHO_EVAL=<ancho> src/evaluar_principal.py val 5000 1 fusionrapido_t0.1` (versión rápida con corrección de cajas, validación completa: 804 WIDER + 1.549 Open Images). Tiempo medido aparte, en un solo proceso, con 30 imágenes. Log: `borradores/e20-bruto.log`. Los modelos se entrenaron a 800 px y no se reentrenaron.
**Resultado (recall de la cara principal WIDER · Open Images / falsos positivos por imagen / recall de secundarias WIDER · Open Images / tiempo):**

| Ancho | Cara principal | FP | Secundarias | Tiempo |
|---|---|---|---|---|
| 480 px | 0,775 · 0,793 | 0,28 · 0,26 | 0,607 · 0,623 | 0,13 s |
| 560 px | 0,782 · 0,795 | 0,29 · 0,28 | 0,667 · 0,642 | 0,18 s |
| **640 px** | **0,792 · 0,790** | 0,29 · 0,27 | 0,676 · 0,652 | **0,24 s** |
| 800 px (E18) | 0,784 · 0,788 | 0,30 · 0,28 | ≈ 0,688 · 0,676 | 0,37 s |
| 1000 px | 0,784 · 0,790 | 0,30 · 0,30 | 0,693 · 0,679 | 0,54 s |

**Lectura:** el recall de la **cara principal** es igual entre 480 y 1000 px (todo dentro del error estadístico, ±1,5 puntos). Lo que se pierde al bajar son las caras **secundarias** (más pequeñas), y empieza por debajo de 640 px. **Decisión: la app trabaja a 640 px** (`ANCHO_FUSION`): misma cara principal, casi las mismas secundarias y un 37 % menos de tiempo (0,24 s contra 0,37 s).
**Límites:** una sola semilla; imágenes de WIDER y Open Images, no webcam (una cara de webcam ocupa más del cuadro que las de estos datasets, lo que favorece a anchos bajos, pero no está medido); los modelos no se reentrenaron a cada ancho.

## E21 · 2026-11-19 · Revisor v2: 5 tipos y óvalos (ruido y reducción de colores)
- Motivo: la app ya ofrece 5 alteraciones (desenfoque, pixelado, tapado, ruido, reducción de colores) y óvalos, pero el generador y el revisor solo conocían 3 tipos y rectángulos (hueco más grande de la app).
- Datos: `generar_alteraciones.py … v2` → `datos/alteradas_v2/` (train 3.000, val 600; test sin generar): 5 tipos al azar, 40 % de las ediciones en óvalo, ruido 30–100 %, 2–6 niveles de color. Mismas divisiones de WIDER (sin fuga).
- Método: `REVISOR_V2=1 src/revisor.py` (datos y modelo propios `modelos/revisor_v2.pkl`). Primer intento con los mismos rasgos de la v1: recall de localización 0,30 con 1.737 falsas (300 imágenes). Dos correcciones: (1) en los óvalos las etiquetas de parche usan la elipse real, no la caja; (2) rasgos nuevos: residuo frente a la mediana (ruido), distancia a una rejilla de 2–6 niveles por canal (posterización), y para el tipo, rasgos medidos en el centro de la región. Etapa 1 con 1.500 imágenes de entrenamiento.
- Umbral de la etapa 1 (ajustado en las primeras 150 de validación): 0,5 → 0,7 → 0,85 → 0,95 → 0,98 = precisión 0,30 / 0,45 / 0,54 / 0,60 con recall 0,57 / 0,59 / 0,52 / 0,41. Se eligió 0,95 y se comprobó en las otras 150.
- Resultado (validación, 300 imágenes, 518 regiones; umbral 0,95; suma de las dos mitades): localiza 267 (recall 0,52), 205 falsas (precisión 0,57), tipo correcto ≈ 0,95 entre las localizadas; en comprobación (150–300): 139/271, 95 falsas, alarma en 9 de 45 imágenes sin edición. Por tipo en comprobación (localizadas/reales): desenfoque 16/47, pixelado 33/48, tapado 34/67, ruido 46/60, colores 10/49.
- Comparación (mismas 300 imágenes v2): las reglas de píxeles, que solo conocen 3 tipos, localizan 105 (recall 0,20), 140 falsas (precisión 0,43), ruido y colores 0/219; pixelado 48/90 (mejor que el bosque), alarma en 13 de 92 sin edición. El bosque v2 duplica el recall total; las reglas siguen siendo mejores y más rápidas (0,05 s vs 0,5 s) en pixelado.
- Lectura: el bosque entrenado ya cubre los 5 tipos y los óvalos; el cuello de botella es la localización (cajas), no el tipo. Colores y desenfoque son lo más flojo (≈ 0,2–0,35).
- Límites: datos sintéticos; umbral ajustado en validación; una corrida; `test` sin tocar. Pendiente: combinar reglas (pixelado) + bosque (resto), mejorar la agrupación de cajas, reentrenar con las 3.000 imágenes.

## E22 · 2026-11-19 · Revisor combinado: reglas (pixelado, tapado) + bosque v2 (resto)
- Método (`src/revisor_combinado.py`): las cajas de pixelado y tapado de las reglas de píxeles mandan; las del bosque v2 (umbral 0,95) se añaden si no se solapan (IoU < 0,3) con ellas.
- Resultado (validación, imágenes 150–300, 271 regiones; antes = bosque solo): localiza 153 (recall 0,513 → 0,565), 128 falsas (95 → 128; precisión 0,594 → 0,544), tipo correcto 0,954, alarma en 12 de 45 sin edición (9 → 12). Por tipo: pixelado 33 → 38 de 48, tapado 34 → 42 de 67, desenfoque 17/47, ruido 46/60, colores 10/49.
- Lectura: gana cobertura en pixelado y tapado a cambio de ≈ 5 puntos de precisión; no cambia el cuello de botella (desenfoque y colores). Una sola corrida, mitad de validación; `test` sin tocar. Queda como opción, no como mejora clara.
