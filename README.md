# deteccion-rostros

Proyecto final de Visión por Computador (Universidad Nacional). Solo visión clásica: **todo menos deep learning**.

Dos partes:

- **A · App que esconde caras.** Abre la cámara, un video o una foto, detecta **muchas caras** y las esconde con desenfoque, pixelado, tapado sólido o ruido, con un slider de intensidad por filtro. Corre en tu computador y solo escucha en `127.0.0.1`: el video nunca sale de tu máquina.
- **B · Revisor de imágenes alteradas.** Dada una imagen guardada, dice **dónde** se alteró una cara y **de qué tipo** (desenfoque, pixelado, tapado), analizando los píxeles.

Antes de tocar nada, lee `brief.md` (objetivo y reglas) y `borradores/decisiones.md` (qué se decidió y por qué). **Si trabajas con un asistente de IA (Claude Code, Codex, Cursor…), él lee `AGENTS.md`**: ahí están las reglas, los datasets que hay que descargar y el protocolo para entrenar y evaluar.

> ## Regla del curso: NO se puede usar YOLO (ni ninguna red neuronal)
> El profesor fijó que el proyecto es **solo visión clásica: todo menos deep learning**. Eso incluye **YOLO** (en cualquier versión), SSD, las redes de OpenCV (`cv2.dnn`, res10), YuNet, MediaPipe, MTCNN, dlib CNN y los embeddings tipo FaceNet, **aunque vengan preentrenados**. No instales `ultralytics`, `torch`, `tensorflow` ni `keras`.
>
> **Por qué:** esto es un curso de visión artificial. Saltar directo a YOLO se salta justo lo que se viene a aprender: cómo se detecta, se filtra y se analiza una imagen con las técnicas clásicas (convoluciones, gradientes, HOG, Haar, SVM, geometría). La idea es resolverlo con esas técnicas, entenderlas y poder defender cada paso.
>
> **Ojo con el nombre de un dataset:** `lylmsc/wider-face-for-yolo-training` (carpeta `entradas/wider-yolo`) dice «yolo» solo porque sus etiquetas están escritas en el *formato* que usa YOLO (un archivo de texto por imagen con `clase cx cy ancho alto`). **No usamos el modelo YOLO**: solo leemos esas cajas como texto.

## 1. Instalación

Probado en macOS con Python 3.13. En Linux y Windows no se ha probado.

```bash
git clone https://github.com/vincent-restrepo/deteccion-rostros.git
cd deteccion-rostros
python3 -m venv .venv
source .venv/bin/activate            # en Windows: .venv\Scripts\activate
pip install -r requirements-clasico.txt
```

`dlib` se compila al instalarse (tarda unos minutos y necesita un compilador de C++: en macOS, `xcode-select --install`; en Windows, Visual Studio Build Tools). `cmake` ya viene en los requisitos.

**Todos los comandos se corren desde la raíz del repositorio** (las rutas son relativas).

## 2. Correr la app (parte A)

```bash
python src/app_blur.py                              # cámara 0
python src/app_blur.py --fuente video.mp4           # un video
python src/app_blur.py --fuente foto.jpg            # una foto
python src/app_blur.py --puerto 8765                # puerto (por defecto 8765)
```

Abre <http://localhost:8765>. A la derecha eliges el filtro en un menú (desenfoque, pixelado, tapado sólido, ruido) y mueves su slider. «Guardar captura» deja la imagen en `salidas_demo/` (esa carpeta **no** se sube a git). En macOS, la primera vez el sistema te pide permiso para usar la cámara.

## 3. Datos (no vienen en el repo)

Los datasets pesan GB y las fotos de personas no se publican. Todo lo derivado se regenera con los scripts de `src/`.

```bash
pip install -r requirements-datos.txt
# Credenciales de Kaggle: crea un token en https://www.kaggle.com/settings y guárdalo en ~/.kaggle/kaggle.json
python src/descargar_kaggle.py lylmsc/wider-face-for-yolo-training=entradas/wider-yolo   # WIDER (2,6 GB, CC0)
python src/dividir_wider.py                    # división sin fuga → datos/division_wider.csv (ya viene en el repo)
python src/generar_alteraciones.py train 3000 1 && python src/generar_alteraciones.py val 600 2 && python src/generar_alteraciones.py test 600 3
```

`datos/division_wider.csv` ya está versionado (la misma división para todos). Las 4.200 imágenes alteradas se generan en `datos/alteradas/` (≈ 600 MB, fuera de git).

## 4. Experimentos y evaluación

Todos miden con la regla de alcance (caras de 48 px de alto o más) y el protocolo sin fuga de datos; la prueba se usa una sola vez al final.

| Qué | Comando |
|---|---|
| Detectores de caras (E7) | `python src/evaluar_caras.py 300 1 haar_0grados,giros_y_perfil,dlib_hog_x1,dlib_hog_x2,fusion` |
| Segunda etapa HOG+SVM (E11) | `python src/verificador.py extraer val 300 2 && python src/verificador.py extraer train 1500 1 && python src/verificador.py evaluar` |
| Revisor por reglas de píxeles (E10) | `python src/analisis_pixeles.py evaluar val 300` y `python src/analisis_pixeles.py ver imagen.jpg` |
| Revisor entrenado, comparación (E9) | `python src/revisor.py entrenar 1500 && python src/revisor.py evaluar val 300` |

Los resultados y su interpretación están en `borradores/experimentos.md`. Los modelos entrenados (`modelos/*.pkl`) no se versionan: se regeneran con los comandos de arriba.

## 5. Estructura

```
src/app_blur.py              app web local (parte A)
src/ediciones.py             los filtros; los usan la app y el generador de datos
src/detector_caras.py        detector clásico (Haar con giros y perfiles)
src/analisis_pixeles.py      revisor por reglas de píxeles (parte B, método principal)
src/revisor.py               revisor entrenado (comparación)
src/verificador.py           candidatas permisivas + HOG+SVM propio
src/dividir_wider.py         división sin fuga (dHash)
src/generar_alteraciones.py  imágenes con caras alteradas y su verdad exacta
src/evaluar_caras.py         evaluación de detectores
brief.md, bitacora.md        objetivo, reglas y registro de cada paso
borradores/                  decisiones, experimentos, procesamiento de datos
entrega/                     material de la presentación (página interactiva del desenfoque)
```

## 6. Reglas del equipo

- **Privacidad:** no subas fotos ni videos de personas, ni capturas de la app (`salidas_demo/`, `datos/`, `entradas/` y `modelos/` están en `.gitignore`). Las fotos de prueba propias se usan solo en tu disco, con permiso de quienes salen.
- **`main` está protegido:** nadie sube directo. Crea una rama (`git checkout -b trabajo/lo-que-sea`), haz commit, `git push -u origin trabajo/lo-que-sea` y abre un Pull Request.
- **Documenta lo que haces:** una línea con fecha en `bitacora.md` por avance; las decisiones en `borradores/decisiones.md` y los experimentos (con su resultado, aunque salga mal) en `borradores/experimentos.md`.
- **Solo visión clásica:** sin redes neuronales (YOLO, SSD, DNN de OpenCV, MediaPipe, etc.). Haar, HOG+SVM (incluido el detector de dlib), LBP, k-NN, árboles y bosques, geometría y Fourier sí.
