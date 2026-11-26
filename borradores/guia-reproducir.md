# Guía para reproducir el proyecto desde cero (con los datasets correctos)

Para quien se une al equipo: sigue este orden y tendrás los **mismos datos, las mismas divisiones y las mismas cifras** que Vincent. Todo lo que se puede regenerar no está en git (pesa GB o son fotos de personas); lo que decide la comparación justa (las divisiones y los hashes) **sí** está versionado: **no lo regeneres**.

## 0. Antes de empezar
- Python 3 con el entorno del proyecto: `python3 -m venv .venv && .venv/bin/pip install -r requirements-clasico.txt -r requirements-datos.txt`.
- Cuenta de Kaggle: crea un token en <https://www.kaggle.com/settings> y guárdalo en `~/.kaggle/kaggle.json` (`chmod 600`). La descarga usa el `python3` del sistema con `kaggle` instalado, no el `.venv`.
- Espacio: ≈ 12 GB para los datos crudos y ≈ 1,2 GB para lo generado.
- Reglas: solo visión clásica (nada de YOLO ni redes neuronales); nada de fotos ni datos al repo; todo cambio por rama y pull request; la **prueba (`test`) no se usa** hasta la evaluación final única.

## 1. Los datasets (qué es cada uno y para qué)

| Dataset | Para qué | Dónde queda | Descarga |
|---|---|---|---|
| **WIDER** (`lylmsc/wider-face-for-yolo-training`, CC0; es el `WIDER_train` oficial) | 1.º dataset de la cara principal; base del generador de la parte B | `entradas/wider-yolo` | `python3 src/descargar_kaggle.py lylmsc/wider-face-for-yolo-training=entradas/wider-yolo` |
| **Open Images** (`fareselmenshawii/face-detection-dataset`) | 2.º dataset de la cara principal (D16) | `entradas/faces-openimages` | `python3 src/descargar_kaggle.py fareselmenshawii/face-detection-dataset=entradas/faces-openimages` |
| **FDDB** (`faizanarif15/dataset-fddb`) | Solo prueba final (junto con `WIDER_val`); no se entrena con él | `entradas/fddb` | `python3 src/descargar_kaggle.py faizanarif15/dataset-fddb=entradas/fddb` |
| **WIDER mksaad** (`mksaad/wider-face-a-face-detection-benchmark`; licencia aprobada por el profesor, ambiente académico) | Solo para comprobar que `WIDER_val` no repite imágenes de entrenamiento | `entradas/wider-mksaad` | `python3 src/descargar_kaggle.py mksaad/wider-face-a-face-detection-benchmark=entradas/wider-mksaad` |
| **Facial Keypoints** (`drgilermo/face-images-with-marked-landmark-points`, espejo de la competencia de Bengio; licencia «desconocida» en Kaggle, uso académico, no se redistribuye) | Puntos faciales (hitos M1–M6 de `estrategia-landmarks.md`) | `entradas/keypoints-fkd` | `python3 src/descargar_kaggle.py drgilermo/face-images-with-marked-landmark-points=entradas/keypoints-fkd` |

Para la **cara principal** necesitas WIDER y Open Images (y FDDB solo al final). Para **landmarks**, Facial Keypoints. mksaad es opcional.

### Verifica tu descarga (SHA-256)
`.venv/bin/python src/hashes_datos.py verificar` debe dar **OK** en los cinco. Si dice «DIFIERE», tu descarga no es idéntica: no entrenes ni midas hasta entender por qué.

| Dataset | Archivos | Bytes | SHA-256 del conjunto |
|---|---|---|---|
| wider-yolo | 25.760 | 2.759.633.019 | `170e10dc…42f492c7` |
| wider-mksaad | 16.122 | 1.848.206.357 | `269d8f35…a6fbd25d` |
| fddb | 28.205 | 550.576.110 | `40918c99…efc5fb46` |
| openimages | 50.199 | 4.765.826.603 | `eeaa4438…1fc9960a` |
| keypoints-fkd | 2 | 521.234.420 | `f91de60a…404eaaea` |

Hashes completos: `datos/hashes_datos.json`. (Nota: el archivo `.npz` de keypoints es de 497 MB y no los «110 MB» que anuncia Kaggle.)

## 2. Divisiones (ya vienen en el repo; no las regeneres)
Sin fuga: las casi repetidas (dHash) siempre caen en la misma división.

| Dataset | Archivo | Reparto | Cómo |
|---|---|---|---|
| WIDER | `datos/division_wider.csv` | train / val / test por grupos | `dividir_wider.py` (dHash ≤ 6 bits, semilla 20261114) |
| Open Images | `datos/division_openimages.csv` | train 11.667 · val 2.498 · test 2.494 · excluidas 74 | `dividir_openimages.py` (se ignora la división original; 34 imágenes heredan la de WIDER; 74 repetidas de las pruebas finales se excluyen) |
| Keypoints | `datos/division_keypoints.csv` | train 4.935 · val 1.057 · test 1.057 | `keypoints_datos.py dividir` (dHash ≤ **1** bit, no 6: son caras recortadas parecidas y a 3+ bits se encadenan en grupos de cientos; semilla 20261114) |

La prueba final de la cara principal es: `WIDER_val` sin 19 imágenes repetidas (`datos/comprobacion_wider_oficial.csv`), FDDB y los `test` de las divisiones anteriores. **Una sola vez.**

Para cargar los datos en tu código: cara principal → `from conjunto_principal import cargar` (1–3 caras, la mayor ≥ 96 px de alto; entrenamiento 11.334 · validación 2.353 · prueba 2.407). Landmarks → `from keypoints_datos import cargar, indices`.

## 3. Regenerar lo derivado (en este orden)

```bash
# Imágenes alteradas de la parte B (revisor). v1 = 3 tipos; v2 = 5 tipos + óvalos
.venv/bin/python src/generar_alteraciones.py train 3000 1 && .venv/bin/python src/generar_alteraciones.py val 600 2 && .venv/bin/python src/generar_alteraciones.py test 600 3
.venv/bin/python src/generar_alteraciones.py train 3000 1 v2 && .venv/bin/python src/generar_alteraciones.py val 600 2 v2    # (test v2 aún sin generar)

# Detector de la cara principal (≈ 15 min; receta completa en README.md §3b)
.venv/bin/python src/hog_svm_cara.py entrenar 3000 1500 2 1
.venv/bin/python src/ablacion_fusion.py cache 1000 0
.venv/bin/python src/ablacion_fusion.py modelo dlib+haar+hogsvm+haar_perfil+extra+corr modelos/fusion_rapido.pkl

# Revisor (parte B)
REVISOR_V2=1 .venv/bin/python src/revisor.py entrenar 1500          # revisor entrenado v2 → modelos/revisor_v2.pkl (≈ 3 min)
REVISOR_V2=1 .venv/bin/python src/revisor_combinado.py val 300 150 0.95

# Reconocer transformación en la app (necesita modelos/revisor_v2.pkl, ver arriba): captura → «Siguiente» → «Reconocer transformación»
.venv/bin/python src/app_blur.py
.venv/bin/python src/reconocer_transformacion.py foto.jpg salida.jpg     # por línea de comandos

# Landmarks (M1)
.venv/bin/python src/keypoints_datos.py piso                          # el piso: la forma media
```

## 4. Cifras que debes obtener (validación)
Si te salen muy distintas, algo está distinto en tu entorno. El error estadístico de una medición es ≈ ±1,5 puntos.

| Qué | Cifra | Fuente |
|---|---|---|
| Cara principal, versión rápida (umbral 0,1), WIDER / Open Images | recall 0,784 / 0,788; 0,3 falsos por imagen; 0,34 s | E15–E20 |
| Cara principal, versión completa (umbral 0,2) | 0,806 / 0,795; 1,9 s | E18 |
| Revisor por reglas de píxeles (3 tipos, v1) | recall 0,59; precisión 0,69 | E10b |
| Revisor entrenado v2 (5 tipos, umbral 0,95; imágenes 150–300) | recall 0,51; precisión 0,59; tipo correcto 0,95 | E21 |
| Revisor combinado (reglas + bosque) | recall 0,57; precisión 0,54 | E22 |
| **Piso de landmarks** (forma media, 4 puntos básicos) | **NME 0,148**; RMSE 3,91 px (cara de 96×96; distancia entre ojos ≈ 36 px) | E23 |

## 5. Lo que sigue y cómo aportar
- Hitos de landmarks M2–M6: `borradores/estrategia-landmarks.md` (piso y «puerta» de cada hito: si no se supera el piso, se para y se documenta).
- Reparto del equipo, calendario y metas: `borradores/plan-trabajo-equipo.md`.
- Cada experimento nuevo se registra en `borradores/experimentos.md` con método, resultado, lectura y límites (también los que salen mal), y una línea en `bitacora.md`.
- Decisiones y su porqué: `borradores/decisiones.md`.
