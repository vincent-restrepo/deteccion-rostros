# Procesamiento del dataset · deteccion-rostros

Cada paso con datos: qué se hizo, filtros, por qué y cifras.

## 2026-11-07 · Lista de candidatos (solo metadatos de la API de Kaggle; nada descargado)

| Dataset | Tamaño | Licencia (Kaggle) | Papel propuesto |
|---|---|---|---|
| `lylmsc/wider-face-for-yolo-training` | 2,63 GB | CC0 | Base: WIDER, 12.880 imágenes con cajas (solo se usa el formato de cajas) |
| `alirezakay/wider-selected` | 0,49 GB | CC0 | Subconjunto fácil frontal, agrupa por evento; no sirve como única evaluación (E1 fue optimista) |
| `mksaad/wider-face-a-face-detection-benchmark` | 3,66 GB | Desconocida | Por abrir: ¿trae divisiones oficiales train/val/test? Si sí, mejor para comparar con la literatura |
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
