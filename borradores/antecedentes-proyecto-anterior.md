# Antecedentes: lo que este proyecto hereda del proyecto anterior

El proyecto anterior (`desenfoque-privacidad`, 2026-11-04 a 2026-11-07) empezó con caras **y documentos de identidad**. El 2026-11-07 el profesor pidió dejar los documentos y concentrarse en rostros (D13 de aquel proyecto), y este proyecto nació de ahí. **Su repositorio de GitHub se eliminó el 2026-11-14** (decisión de Vincent: era ruido). Este archivo conserva lo que la documentación de aquí cita: los resultados E1, E3, E5 y E6 (detección de caras) y las decisiones que siguen vigentes. Todo lo demás de aquel proyecto trataba de cédulas, MIDV y datasets de documentos y se dejó fuera a propósito.

En este documento, «WIDER» sin más es el dataset `lylmsc/wider-face-for-yolo-training` (solo se usan sus cajas; no hay YOLO). Los nombres E1–E6 y D1–D14 de abajo son los **del proyecto anterior**; los de este proyecto (E7 en adelante, D1 en adelante) están en `experimentos.md` y `decisiones.md`.

## Experimentos que siguen siendo relevantes

### E1 · Línea base: Haar `frontalface_alt2` solo (2026-11-04)
- Datos: 300 imágenes al azar (semilla 0) de `alirezakay/wider-selected` (4.275 caras frontales «fáciles» de WIDER). `scaleFactor` 1,1, `minNeighbors` 3, cara mínima 24×24; acierto con IoU ≥ 0,5.
- Resultado: 399 aciertos, 246 falsas alarmas, 47 caras no detectadas → recall 0,895; precisión 0,619.
- Lectura: **optimista**, porque el dataset ya estaba filtrado a caras frontales; E5 lo corrigió.

### E3 · Cabeza inclinada (2026-11-04)
- Con la cámara, una cabeza inclinada unos 30° daba 0 caras: el detector era Haar frontal a 0° con **ecualización de histograma**.
- Hallazgos: ecualizar empeoraba (0 detecciones); sin ecualizar, Haar frontal (alt2 y alt) encuentra la cara al girar la imagen −30° (y −20°); con `minNeighbors` 2 aparecen cajas falsas y con 3 queda solo la buena.
- Cambio: `src/detector_caras.py` (dos cascadas frontales sobre la imagen girada −60°…+60° cada 15°, más perfil espejado, sin ecualizar, `minNeighbors` 3). Es un ejemplo, no una tasa de acierto.

### E5 · Línea base en WIDER con muchas caras por imagen (2026-11-07)
- 120 imágenes al azar (semilla 0), 1.475 caras, imagen reducida a 800 px de ancho, IoU ≥ 0,5.

| Detector | Recall | Precisión | Falsas alarmas | s/imagen |
|---|---|---|---|---|
| Haar a 0° | 0,105 | 0,783 | 43 | 0,05 |
| Giros + perfil | 0,100 | 0,279 | 380 | 0,83 |

- Recall por alto de la cara: menos de 24 px (906 caras, el 61 %): 0,000 con ambos; 24–48 px: 0,091 y 0,085; 48–96 px: 0,471 y 0,355; 96 px o más: 0,500 y 0,581.
- Lectura: ningún detector clásico ve las caras diminutas, que son la mayoría en WIDER; y girar la imagen no mejora el recall general y multiplica las falsas alarmas. De aquí salió la regla de alcance de **48 px** (D1 de este proyecto).

### E6 · dlib HOG + SVM en WIDER (2026-11-07)
- Mismas 120 imágenes y regla que E5. `x0, x1, x2` = veces que dlib agranda la imagen antes de buscar.

| Detector | Recall | Precisión | Falsas alarmas | s/imagen |
|---|---|---|---|---|
| dlib HOG x0 | 0,052 | 0,962 | 3 | 0,04 |
| dlib HOG x1 | 0,094 | 0,965 | 5 | 0,13 |
| dlib HOG x2 | 0,148 | 0,956 | 10 | 0,52 |

- Recall por alto con x2: menos de 24 px: 0,011; 24–48: 0,261; 48–96: 0,435; 96 o más: 0,548.
- Lectura: dlib casi no inventa caras (precisión ≈ 0,96 frente a 0,78 de Haar). Límite: 120 imágenes y una semilla.

## Decisiones que siguen vigentes

- **D1 · Solo visión clásica** (2026-11-04). Nada de YOLO ni redes neuronales. Razón de fondo (Vincent): la materia es visión artificial y las operaciones clásicas cumplen la función sin recurrir a algo tan avanzado como una red neuronal. Hoy está como D11 de este proyecto.
- **D2 · Falso negativo peor que falso positivo** (2026-11-04): umbrales hacia recall alto; en privacidad, dejar una cara sin esconder es peor que esconder de más.
- **D8 · Interfaz web local con servidor estándar** (2026-11-04): `http.server` de Python y video MJPEG, solo en `127.0.0.1`; margen de 25 % alrededor de la caja. (El suavizado «5 de 9 cuadros» de entonces se cambió: ver E8b de este proyecto.)
- **D13 · Cambio de alcance (2026-11-07):** el profesor pidió (1) dejar las cédulas («es pegarse un tiro en el pie»), (2) ser muy bueno detectando rostros, muchos por imagen o video, (3) seguir con la app con caja por rostro, (4) elegir entre blur y pixelado, (5) guardar la imagen y una segunda parte que revise las guardadas y diga **dónde** se alteró un rostro y **de qué tipo**, (6) otras formas de editar además de blur y pixelado. De ahí salen las partes A y B de este proyecto.
- **D14 · «Todo menos deep learning» (2026-11-07):** entran HOG+SVM (incluido el detector HOG de dlib), Haar, LBP, árboles, bosques, boosting, k-NN, Fourier, flujo óptico, Kalman, geometría; quedan fuera las redes neuronales aunque vengan preentrenadas (YOLO, SSD, `cv2.dnn`, YuNet, MediaPipe, MTCNN, dlib CNN, FaceNet). Zona gris por confirmar con el profesor: si un modelo clásico preentrenado por otros, como el HOG de dlib, cuenta como «propio»; la regla dicha no lo prohíbe y el informe declara de dónde viene cada modelo.
