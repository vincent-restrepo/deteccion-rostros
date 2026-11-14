# Instrucciones para asistentes de IA (Claude Code, Codex, Cursor, etc.)

Estás ayudando a una persona del equipo en este repositorio. Léelo completo antes de actuar. Hablas con personas que hacen un **proyecto final de Visión por Computador**; el profesor evalúa sobre todo **cómo se trataron los datos**, así que cada decisión tiene que quedar escrita y defendible.

> # ⛔ REGLA Nº 1: NO se puede usar YOLO ni ninguna red neuronal
> El profesor fijó que el proyecto es **solo visión clásica: todo menos deep learning**. **Prohibido, aunque venga preentrenado:** YOLO (cualquier versión), SSD, `cv2.dnn` / res10, YuNet, MediaPipe/BlazeFace, MTCNN, dlib CNN, FaceNet y cualquier otra red neuronal. No instales ni importes `ultralytics`, `torch`, `tensorflow`, `keras`, `onnx`. No uses los scripts de redes que traen algunos datasets.
> **Por qué:** esto es un curso de visión artificial. Saltar directo a YOLO se salta justo lo que se viene a aprender: cómo se detecta, se filtra y se analiza una imagen con las técnicas clásicas (convoluciones, gradientes, HOG, Haar, SVM, geometría). La idea es resolverlo con esas técnicas, entenderlas y poder defender cada paso.
> Si la persona te pide algo que lo requiera, **no lo hagas**: explícale que el curso lo prohíbe y propón la alternativa clásica (Haar, HOG+SVM, LBP, cascadas propias, k-NN, bosques, boosting).
> **Trampa del nombre:** el dataset `wider-face-for-yolo-training` solo trae sus etiquetas en *formato* YOLO (texto: `clase cx cy ancho alto`). Eso es un formato de archivo, no el modelo. Se lee como texto y no se usa YOLO.

## 0. Qué leer primero (en este orden)

1. `README.md`: instalación, comandos, estructura.
2. `brief.md`: objetivo y reglas del curso.
3. `borradores/decisiones.md`: D1–D12, las decisiones ya tomadas y su porqué. **No las contradigas; si crees que una está mal, dilo a la persona y propón cambiarla por escrito.**
4. `borradores/antecedentes-proyecto-anterior.md`: lo que se hereda del proyecto anterior (E1, E3, E5, E6 y las decisiones vigentes).
5. `borradores/experimentos.md`: E7–E11 con resultados y límites. Aquí están las cifras de referencia.
6. Las últimas líneas de `bitacora.md`: en qué quedó el trabajo.

## 1. El proyecto en cuatro líneas

- **Parte A:** app web local (`src/app_blur.py`) que detecta muchas caras y las esconde (desenfoque, pixelado, tapado sólido, ruido, reducción de colores).
- **Parte B:** revisor que, dada una imagen guardada, dice dónde se alteró una cara y de qué tipo (`src/analisis_pixeles.py` por reglas de píxeles; `src/revisor.py` entrenado, solo como comparación).
- **Regla del curso: todo menos deep learning.** Prohibido, aunque venga preentrenado: YOLO, SSD, DNN de OpenCV (res10), YuNet, MediaPipe/BlazeFace, MTCNN, dlib CNN, embeddings tipo FaceNet, cualquier red neuronal. Permitido: Haar, HOG+SVM (incluido el detector HOG de dlib), LBP, k-NN, árboles, bosques, boosting (AdaBoost, `opencv_traincascade`), geometría, Fourier. «Formato YOLO» solo se refiere a cómo vienen escritas las cajas (texto), no a usar el modelo.
- **Prioridad de privacidad:** un falso negativo (cara visible) es peor que un falso positivo (tapar de más).

## 2. Reglas de trabajo (obligatorias)

1. **`main` está protegido.** Nunca hagas push directo. Rama `trabajo/<tema>` → commit → `git push -u origin trabajo/<tema>` → `gh pr create` → unir. Un cambio por PR, con mensaje en español que diga qué y por qué.
2. **No subas datos ni fotos de personas.** `entradas/`, `datos/alteradas/`, `datos/*.pkl`, `modelos/`, `salidas_demo/` están en `.gitignore`: no los fuerces (`git add -f`). El repo es **público**. No subas capturas de la app, videos, ni fotos de compañeros.
3. **Documenta todo lo que haces**, en el momento:
   - una línea con fecha en `bitacora.md` por cada avance;
   - una decisión nueva → `borradores/decisiones.md` (D11, D12…: qué, por qué, quién la aprobó);
   - cada experimento → `borradores/experimentos.md` (E12, E13…) **antes** de correrlo (pregunta, método, datos, métrica) y el resultado **después**, aunque salga mal. Guarda la salida cruda en `borradores/eN-bruto.log`;
   - todo procesamiento de datos → `borradores/procesamiento-dataset.md`.
4. **No inventes cifras ni resultados.** Si no lo mediste, dilo. Reporta los fallos con la misma claridad que los aciertos. Di el tamaño de la muestra y la semilla.
5. **No reescribas el historial de git** (nada de `push --force`, `rebase` sobre `main`, `filter-branch`).
6. **Correr desde la raíz del repo.** Las rutas son relativas.
7. Sé cuidadoso con el procesador: los experimentos pesados corren con `nice` y no más de 3 procesos (el portátil se recalentó una vez).

## 3. Datasets: qué descargar y para qué

Los datos **no vienen en el repo** (pesan GB y tienen fotos de personas). Cada persona los baja en su máquina. Necesitas una cuenta de Kaggle (token en `~/.kaggle/kaggle.json`) y `pip install -r requirements-datos.txt`.

| Rol | Dataset (Kaggle) | Tamaño | Licencia | Carpeta | Estado |
|---|---|---|---|---|---|
| **Obligatorio:** desarrollo y entrenamiento | `lylmsc/wider-face-for-yolo-training` | 2,6 GB | CC0 | `entradas/wider-yolo` | Verificado. 12.880 imágenes (`images/wider_N.jpg`) con cajas YOLO normalizadas (`labels/wider_N.txt`: clase, cx, cy, w, h) |
| **Prueba final y división oficial** | `mksaad/wider-face-a-face-detection-benchmark` | 3,4 GB | ver Kaggle | `entradas/wider-mksaad` | Verificado. WIDER original: `WIDER_train` (12.880 imágenes), `WIDER_val` (3.226), anotaciones en `wider_face_split/wider_face_split/wider_face_{train,val}_bbx_gt.txt` (por imagen: ruta, nº de caras, y por cara `x y ancho alto blur expresión iluminación invalida oclusión pose`). **Trae scripts de redes neuronales (`train.py`, `eval.py`, `demo.py`): no los uses ni los corras** |
| **Prueba externa (dejar un dataset fuera)** | `faizanarif15/dataset-fddb` | 0,6 GB | Apache 2.0 | `entradas/fddb` | Verificado. `Dataset_FDDB/label.txt`: por imagen `# ruta` y luego una caja por línea en formato **`x1 y1 x2 y2`** (no `x y ancho alto`); 2.835 imágenes anotadas, 5.155 caras. La carpeta `images/` tiene 28.204 jpg pero solo 2.835 están anotadas (no se sabe qué son las demás: ignóralas) |

Comandos:

```bash
python src/descargar_kaggle.py lylmsc/wider-face-for-yolo-training=entradas/wider-yolo
python src/descargar_kaggle.py mksaad/wider-face-a-face-detection-benchmark=entradas/wider-mksaad
python src/descargar_kaggle.py faizanarif15/dataset-fddb=entradas/fddb
```

Para lo básico basta el primero. El segundo y el tercero hacen falta solo para la evaluación final y para «dejar un dataset fuera».

**Cuántas caras cuentan:** solo las de **48 px de alto o más** (D1). En WIDER el 81 % de las caras es más pequeño: quedan fuera de alcance, no suman recall ni su acierto cuenta como falso positivo. Los scripts ya aplican esa regla; si escribes uno nuevo, aplícala igual.

**Segundo dataset de entrenamiento:** D2 pide entrenar con WIDER más otro dataset, pero todavía **no se eligió**. Candidatos sin verificar (solo vistos por título en Kaggle): `fareselmenshawii/face-detection-dataset`, `freak2209/face-data`, `ngoduy/dataset-for-face-detection`. Si vas a usar uno: descárgalo, ábrelo, registra en `borradores/procesamiento-dataset.md` el conteo de imágenes y caras, el formato de las cajas, la licencia y los tamaños de cara, y propón la elección como una decisión nueva antes de entrenar con él.

## 4. Protocolo de datos para entrenar y evaluar (sin fuga)

- **División fija:** `datos/division_wider.csv` (ya versionado) reparte las 12.880 imágenes de `wider-yolo` en `train` 70 % / `val` 15 % / `test` 15 %, agrupando antes las casi repetidas (dHash 9×8 a ≤ 6 bits) para que la misma foto no caiga a ambos lados. **No la regeneres con otra semilla** (la semilla es 20261114). Si cambias el criterio, es una decisión nueva.
- **Entrenas con `train`. Ajustas umbrales e hiperparámetros solo con `val`.** El `test` de esa división y la **validación oficial de WIDER (`WIDER_val` de mksaad)** y **FDDB** son para la evaluación final: **se usan una sola vez**, al terminar, y se reporta lo que salga. Si miras el resultado y luego «ajustas», ya no es una prueba: dilo.
- **Si agregas otro dataset:** calcula dHash sobre el conjunto junto y agrupa los casi repetidos **antes** de dividir (usa `src/dividir_wider.py` de plantilla). Convierte las cajas a un mismo formato y reporta los resultados **por dataset**, no solo promediados.
- **Siempre compara con una línea base:** los números de la sección 6 y un método trivial.
- **Misma regla de conteo siempre:** IoU ≥ 0,5; caras ≥ 48 px de alto; por tamaño (48–96 y ≥ 96).
- **Fija semillas** y escribe cuáles; la comparación entre dos métodos debe ser sobre **las mismas imágenes**.
- **Generador de datos de la parte B** (`src/generar_alteraciones.py`): usa las mismas divisiones; sus semillas y parámetros están en `borradores/procesamiento-dataset.md`.

## 5. Dónde se puede entrenar y qué ideas están abiertas

| Qué | Dónde | Estado |
|---|---|---|
| Segunda etapa de caras (HOG 64×64 + SVM lineal con negativos difíciles) | `src/verificador.py` | E11: ayuda poco; el techo lo ponen las candidatas |
| Mejorar las **candidatas** (lo que más falta: techo de recall 0,63) | `src/verificador.py::candidatas`, `src/detector_caras.py` | Abierto: más resolución, giros, perfiles, otros umbrales |
| Un detector HOG+SVM propio por ventana deslizante, o una cascada propia (`opencv_traincascade` con Haar/LBP) entrenada con WIDER | nuevo script en `src/` | Abierto, no probado |
| Revisor entrenado (bosque aleatorio sobre rasgos de píxeles) | `src/revisor.py` | E9: más recall en desenfoque, muchas falsas alarmas |
| Umbrales de las reglas del revisor | `src/analisis_pixeles.py` (constantes arriba) | Se ajustan solo en `val` |

Pendientes conocidos (ver también `bitacora.md`): el revisor no reconoce **tapados ni desenfoques en óvalo** ni el **ruido**; el generador de datos debe producirlos y medirse aparte; la app todavía usa Haar y no el dlib con umbral bajo; no hay una prueba con cámara de dlib permisivo.

## 6. Cifras de referencia (para comparar; todas en validación, con la regla de 48 px)

| Experimento | Método | Recall | Precisión | Muestra |
|---|---|---|---|---|
| E7 | Haar 0° | 0,452 | 0,747 | 300 img, semilla 1 |
| E7 | Haar con giros y perfiles | 0,476 | 0,257 | idem |
| E7 | dlib HOG x1 | 0,478 | 0,955 | idem |
| E7 | fusión dlib + Haar | 0,496 | 0,262 | idem |
| E11 | dlib HOG umbral −0,4 | 0,552 | 0,808 | 300 img, semilla 2 |
| E11 | todas las candidatas (techo) | 0,630 | 0,238 | idem |
| E11 | candidatas + SVM (umbral −0,5) | 0,578 | 0,727 | idem |
| E10b | revisor por reglas (parte B) | 0,592 | 0,692 | 300 img generadas, 505 regiones |
| E9 | revisor entrenado (parte B) | 0,646 | 0,220 | idem |

Las muestras son chicas y de una sola semilla: sirven para comparar métodos, no como cifra final.

## 7. Con las fotos y la cámara

- La app corre solo en `127.0.0.1`; el video no sale de la máquina. No la expongas a la red.
- Si la persona toma fotos de prueba en clase: solo con permiso de quienes salen, solo en su disco, nunca al repo, nunca para entrenar ni para la prueba final.

## 8. Cómo comportarte con la persona

- Responde en **español**, claro y sin relleno. Primero el resultado, después el detalle.
- Si lo que te piden choca con las reglas del curso (deep learning), con la privacidad (subir fotos) o con el protocolo (mirar la prueba y ajustar), **dilo antes de hacerlo** y propón la alternativa correcta.
- Antes de borrar algo, haz una copia. No toques el trabajo de otras personas del equipo sin que te lo pidan.
- Cuando termines un paso: qué quedó, dónde está y qué falta. Si algo no se pudo comprobar, dilo.
