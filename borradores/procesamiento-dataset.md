# Procesamiento del dataset · deteccion-rostros

Cada paso con datos: qué se hizo, filtros, por qué y cifras.

## 2026-11-07 · Lista de candidatos (solo metadatos de la API de Kaggle; nada descargado)

**Estado a 2026-11-19 (la tabla es documentación; no se presenta en la exposición):** USADOS: `lylmsc/wider-face-for-yolo-training`, `mksaad/...` y `faizanarif15/dataset-fddb`. DESCARTADOS: `alirezakay/wider-selected` (subconjunto fácil), `tanmaypawar/...` y `cormacwc/...` (FDDB repetido; ya tenemos el completo con licencia clara), `ngoduy/dataset-for-face-detection` (contiene WIDER y FDDB; ver abajo). SIN USAR: `vin1234/...` y `andrewmvd/face-mask-detection` (no son de detección de caras en general). Segundo dataset de entrenamiento: ver la sección del 2026-11-19.

| Dataset | Tamaño | Licencia (Kaggle) | Papel propuesto |
|---|---|---|---|
| `lylmsc/wider-face-for-yolo-training` | 2,63 GB | CC0 | Base: WIDER, 12.880 imágenes con cajas (solo se usa el formato de cajas) |
| `alirezakay/wider-selected` | 0,49 GB | CC0 | Subconjunto fácil frontal, agrupa por evento; no sirve como única evaluación (E1 fue optimista) |
| `mksaad/wider-face-a-face-detection-benchmark` | 3,66 GB | Desconocida en Kaggle; el profesor dijo (2026-11-19, según Vincent) que se puede usar por ser un entorno académico | **USADO**: WIDER oficial; `WIDER_val` es la prueba final |
| `tanmaypawar/fddb-subset-for-face-detection-yolo-format` | 0,06 GB | CC0 | FDDB (subconjunto): segundo dataset |
| `faizanarif15/dataset-fddb` | 0,55 GB | Apache 2.0 | FDDB completo con licencia clara |
| `cormacwc/fddb-dataset` | 0,55 GB | Desconocida | FDDB, alternativa |

Sin verificar tamaño/licencia todavía: `fareselmenshawii/face-detection-dataset`, `freak2209/face-data`, `ngoduy/dataset-for-face-detection`, `vin1234/count-the-number-of-faces-present-in-an-image`, `andrewmvd/face-mask-detection`, `jessicali9530/celeba-dataset`.

## Parte B · datos de imágenes alteradas (2026-11-14)
1. **División sin fuga de WIDER** (`src/dividir_wider.py` → `datos/division_wider.csv`, `division_wider_resumen.txt`): 12.880 imágenes de `entradas/wider-yolo`; dHash 9×8, casi repetidas a ≤ 6 bits en el mismo grupo (transitivo). Resultado: 12.848 grupos, el mayor de 11; solo 48 imágenes están en grupos de 2 o más. Grupos completos repartidos 70/15/15 con semilla 20261114: entrenamiento 9.016, validación 1.932, prueba 1.932.
2. **Generación** (`src/generar_alteraciones.py`, ediciones en `src/ediciones.py`, las mismas de la app): 3.000 imágenes de entrenamiento (semilla 1), 600 de validación (semilla 2), 600 de prueba (semilla 3; no se toca hasta la evaluación final). Por imagen: 15 % sin editar; de las demás, cada cara de ≥ 48 px de alto se edita con probabilidad 0,6. Tipo uniforme entre desenfoque (fuerza 0,4–2,0), pixelado (bloque 4, 6, 8, 12, 16, 24, 32, 48, 64) y tapado (negro, blanco, color medio de la cara o color al azar); margen 0–0,35; se descartan ediciones que se solapan > 0,3 IoU; JPEG con calidad al azar 70–98. Se guarda la región exacta modificada, el tipo y los parámetros (`etiquetas.jsonl`) y las caras sin editar (negativos).
3. Una primera versión usaba bloques 2 y 3; se descartó porque casi no cambian la cara (no esconden) y la app no los ofrece. Se regeneró todo.
4. Las imágenes generadas viven en `datos/alteradas/` (fuera de git; se regeneran con los comandos de arriba).

## Descargas completadas y verificadas (2026-11-14)
Las descargas relanzadas el 2026-11-14 (la anterior llevaba 7 días en 0 B) terminaron. Se abrieron y se contó lo siguiente:
1. **WIDER oficial** (`mksaad/wider-face-a-face-detection-benchmark`, `entradas/wider-mksaad`, 3,4 GB descargados): `WIDER_train` 12.880 imágenes y 159.424 caras; `WIDER_val` 3.226 imágenes y 39.708 caras; todas las imágenes tienen al menos una cara. Cajas `x y ancho alto` + atributos (blur, expresión, iluminación, invalida, oclusión, pose) en `wider_face_split/wider_face_split/wider_face_{train,val}_bbx_gt.txt` (y `.mat`). Caras de menos de 48 px de alto: 129.545 de 159.424 en train (81 %) y 31.966 de 39.708 en val (80,5 %). Trae scripts de redes neuronales (`train.py`, `eval.py`, `demo.py`, `download_model.py`): no se usan (regla del curso). Papel: `WIDER_val` queda **reservada como prueba final** (se usa una sola vez, D2); aún no se ha usado para nada.
2. **FDDB** (`faizanarif15/dataset-fddb`, `entradas/fddb`, 581 MB): `Dataset_FDDB/label.txt` con 2.835 imágenes anotadas y 5.155 caras. Formato por imagen: `# ruta` y una caja por línea **`x1 y1 x2 y2`** (comprobado: como `x y ancho alto` la primera caja se saldría de la imagen de 431×450). Alto de cara: percentil 10 = 57 px, mediana = 125 px, percentil 90 = 251 px; solo el 8 % mide menos de 48 px (casi todas están en alcance, a diferencia de WIDER). La carpeta `images/` tiene 28.204 jpg pero solo 2.835 están anotadas; no se sabe qué son las demás y no se usan. Papel: prueba externa («dejar un dataset fuera», D2); aún no se ha usado.
3. Las imágenes de `wider-yolo` (lylmsc) se llaman `wider_N.jpg` y las de `mksaad` conservan los nombres originales de WIDER (por ejemplo `36_Football_Football_36_63.jpg`): no hay una tabla que las una. **Pendiente:** comprobar con dHash cuántas imágenes de `wider-yolo` coinciden con `WIDER_train` de mksaad y, sobre todo, que `WIDER_val` no comparta imágenes casi idénticas con `wider-yolo` (si lo hiciera, no sería una prueba limpia).

## 2026-11-19 · Comprobación de WIDER oficial contra wider-yolo, hashes y segundo dataset
### 1. ¿`WIDER_val` repite imágenes de `wider-yolo`? (`src/comprobar_wider_oficial.py`)
dHash 9×8, «casi igual» = ≤ 6 bits, como en la división. Resultado (`datos/comprobacion_wider_oficial.txt` y `.csv`, versionados):
- **`wider-yolo` es el `WIDER_train` oficial:** las 12.880 imágenes de `WIDER_train` tienen su par en `wider-yolo` (12.509 con distancia 0; el resto ≤ 6 bits, por recompresión/tamaño). No son datasets distintos: no se puede «sumar» uno al otro para entrenar.
- **`WIDER_val`:** 19 de 3.226 imágenes son casi iguales a alguna de `wider-yolo` (solo 2 con distancia 0: `35_Basketball_playingbasketball_35_476` y `48_Parachutist_Paratrooper_..._48_321`). Caen 10 en la parte de entrenamiento de nuestra división, 5 en prueba y 4 en validación. **No las miré una por una**: con ≤ 6 bits puede haber parecidos que no sean la misma foto.
- **Decisión (D14):** la prueba final con `WIDER_val` **excluye esas 19 imágenes** (quedan 3.207). La lista está en el CSV. Con eso la prueba final es limpia frente a todo lo de `wider-yolo`.
### 2. Hashes SHA-256 de los datos crudos (`src/hashes_datos.py`, `datos/hashes_datos.json`)
Por dataset: número de archivos, bytes y un hash de conjunto (SHA-256 de las líneas «ruta hash» ordenadas); más el hash de los archivos de anotaciones. `python src/hashes_datos.py verificar` recalcula y compara (probado el 2026-11-19: los tres dan OK).

| Dataset | Archivos | Bytes | SHA-256 del conjunto |
|---|---|---|---|
| wider-yolo | 25.760 (12.880 imágenes + 12.880 etiquetas) | 2.759.633.019 | `170e10dc…42f492c7` |
| wider-mksaad | 16.122 | 1.848.206.357 | `269d8f35…a6fbd25d` |
| fddb | 28.205 (28.204 jpg + `label.txt`) | 550.576.110 | `40918c99…efc5fb46` |
| openimages | 50.199 | 4.765.826.603 | `eeaa4438…1fc9960a` |
| keypoints-fkd | 2 (`face_images.npz` + `facial_keypoints.csv`) | 521.234.420 | `f91de60a…404eaaea` |

Los hashes completos y los de las anotaciones están en el JSON. Si tu verificación da «DIFIERE», tu descarga no es idéntica: no entrenes ni mides con ella hasta entender por qué.
### 3. Segundo dataset de entrenamiento: lo que dice Kaggle (sin descargar nada; los conteos reales exigen bajarlo)
| Candidato | Tamaño | Licencia declarada | Qué es (según su ficha) | Veredicto |
|---|---|---|---|---|
| `fareselmenshawii/face-detection-dataset` | 4,76 GB | CC0-1.0 | 16,7 mil imágenes de **Google Open Images** (extraídas con OIDv4), solo caras; cajas en píxeles y en formato YOLO | **Recomendado**: fuente distinta de WIDER y probable mayoría de caras grandes, útil para el enfoque de cara principal (D13). Pendiente de verificar al descargar |
| `freak2209/face-data` | 3,51 GB | CC-BY-NC-SA-4.0 | Más de 1.000 imágenes PNG grandes con cajas YOLO; procedencia no explicada | Segunda opción: pocas imágenes, licencia no comercial con compartir igual y origen incierto |
| `ngoduy/dataset-for-face-detection` | 3,68 GB | desconocida | «WIDERFACE para train y test, FDDB para test, LFPW para landmarks» | **Descartado para detección**: repite WIDER (ya lo tenemos) y FDDB (nuestra prueba externa: sería fuga). LFPW podría servir luego para landmarks, pero la licencia es desconocida |

Cautela con `fareselmenshawii`: la licencia CC0 es la que declara quien subió el mirror; las imágenes de Open Images tienen sus propias licencias (las anotaciones CC BY 4.0 según Google; las fotos, CC BY o similares). Para uso académico local no hay problema, pero no se redistribuyen. Antes de entrenar: descargarlo, contar imágenes y caras, ver tamaños de cara, calcular dHash contra WIDER y FDDB y registrar todo aquí; la elección es una decisión nueva.

## 2026-11-19 · Segundo dataset elegido: Open Images (`fareselmenshawii/face-detection-dataset`) — D16
Descargado en `entradas/faces-openimages` (4,6 GB). Inspección con `src/inspeccionar_openimages.py` (`datos/inspeccion_openimages.txt`; hashes en `datos/openimages_dhash.csv`):
- **Contenido:** 16.733 imágenes (su división original: 13.386 train y 3.347 val; no la usamos), 51.846 caras, ninguna imagen sin caras. Dos formas de cajas: `labels/` (YOLO normalizado, por división original) y `labels2/` (píxeles `x1 y1 x2 y2` con el texto «Human face»; una carpeta para todo). El cargador usa `labels2/`.
- **Caras por imagen:** 1: 7.260; 2: 3.564; 3: 1.753; 4: 1.168; 5: 808; 6 o más: 2.180.
- **Tamaño de cara (alto en px):** mediana 85 (p10 27, p90 259); el 26,2 % mide menos de 48 px. La cara más grande por imagen: mediana 174 (p10 74, p90 415). Imágenes de ancho mediano 1.024 y alto 768.
- **Para D13:** con ≤ 3 caras y la mayor ≥ 96 px hay 10.741 imágenes (≥ 48 px: 12.173); con exactamente 1 cara y ≥ 96 px, 6.291.
- **Repetidos (dHash ≤ 6 bits):** 44 imágenes con otra casi igual dentro del propio dataset; 56 casi iguales a `wider-yolo`; 21 a `WIDER_val`; 64 a FDDB. Tratamiento en `src/dividir_openimages.py` (D16): 74 excluidas, 34 heredan división de WIDER, el resto agrupado y repartido: train 11.667, val 2.498, test 2.494 (`datos/division_openimages_resumen.txt`).
- **Conjunto de la cara principal** (`src/conjunto_principal.py`, `datos/conjunto_principal_resumen.txt`): train 11.334 (WIDER 3.809 + Open Images 7.525), val 2.353, test 2.407.
- SHA-256 de este dataset: hecho el 2026-11-19 (tabla de arriba). **Pendiente:** ver a ojo unas cuantas cajas para confirmar que son caras de frente y que no hay etiquetas raras.

## 2026-11-19 · Facial Keypoints (landmarks, hito M1) — ficha
- Qué: espejo de Kaggle `drgilermo/face-images-with-marked-landmark-points` (la competencia «Facial Keypoints Detection», datos de Yoshua Bengio, Univ. de Montreal). 7.049 caras de 96×96 en gris (`face_images.npz`, matriz 96×96×7049, valores 0–255) y `facial_keypoints.csv` (30 columnas = 15 puntos × (x, y); vacías cuando falta).
- Licencia: «desconocida» en Kaggle; uso académico, no se redistribuye, nada al repo. (La competencia original exige aceptar sus reglas en el navegador; el espejo es público.)
- Qué trae de verdad: los 15 puntos completos solo en 2.140 caras; centros de ojos ≈ 7.040; nariz 7.049; labio inferior 7.016; el resto ≈ 2.270 (ver E23). El `.npz` se guardó con Python 2 (aviso de numpy al leerlo; funciona).
- Duplicados: ≈ 16 % de las imágenes tienen un gemelo exacto por dHash; por eso la división es por grupos (umbral 1 bit; el de 6 bits de WIDER encadena caras parecidas). `datos/division_keypoints.csv`.
- Uso: `src/keypoints_datos.py` (`cargar`, `indices`, `dividir`, `piso`); resultados en E23; plan en `estrategia-landmarks.md`.
