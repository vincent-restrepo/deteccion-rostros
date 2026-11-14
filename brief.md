# deteccion-rostros

## Objetivo
Proyecto final de Visión por Computador (Universidad Nacional). Dos partes:
- **A · detector y editor:** programa web local (solo `127.0.0.1`, video MJPEG) que abre cámara del Mac, video o foto, detecta **muchos rostros por imagen o video**, dibuja una caja (*bounding box*) por rostro y permite esconderlos (blur, pixelado, caja negra, color plano promedio, ruido, bloques revueltos, oscurecer, cubrir con forma). Controles: «Esconder: sí / no», tipo de edición, cajas de depuración, guardar captura, contador de rostros.
- **B · revisor:** abre imágenes guardadas y marca **dónde** se alteró un rostro y **de qué tipo** (blur, pixelado u otro). Rasgos clásicos por ventana + SVM o k-NN.

## Para quién
Profesor de Visión por Computador. Le importa sobre todo **cómo se trataron los datos**; Vincent tiene que poder sustentar cada decisión.

## Decisión o entrega
Código, informe y demo (cámara del Mac). Formato exacto: pendiente.

## Fecha límite
Pendiente (aprox. fin de noviembre de 2026; confirmar con el profesor).

## Reglas del curso
**NO se puede usar YOLO ni ninguna red neuronal** (aunque venga preentrenada). El dataset `wider-face-for-yolo-training` solo trae las cajas escritas en formato YOLO (texto); el modelo YOLO no se usa. **Por qué:** es un curso de visión artificial; saltar a YOLO se salta lo que se viene a aprender, así que se resuelve con las técnicas clásicas.

Todo menos deep learning. Entran HOG+SVM (incluido el detector de dlib), Haar, LBP, k-NN, árboles, bosques, boosting, geometría, Fourier. Quedan fuera aunque vengan preentrenados: YOLO, SSD, DNN de OpenCV, YuNet, MediaPipe, MTCNN, detector CNN de dlib, embeddings tipo FaceNet. Si se usa un modelo preentrenado clásico (HOG de dlib), el informe declara de dónde viene. Pendiente confirmar con el profesor si cuenta como «propio».

## Principios
- Falso negativo peor que falso positivo: umbrales hacia recall alto; el blur se mantiene unos cuadros tras perder la cara.
- Todo el procesamiento de datos queda registrado (`borradores/procesamiento-dataset.md`, `experimentos.md`, `decisiones.md`), con SHA-256 de los datos crudos.
- Protocolo: división sin fuga (casi repetidas con dHash ≤ 6 bits, dividir por grupo); la prueba se usa una sola vez; resultados por dataset y dejando un dataset fuera; línea base trivial; fallos reportados con honestidad. Detección: precisión, recall y AP con IoU ≥ 0,5, por tamaño de cara y por número de caras por imagen.

## Fuentes conocidas
Ver `borradores/procesamiento-dataset.md`. Antecedente: el proyecto anterior (repo eliminado el 2026-11-14); lo que se hereda está en `borradores/antecedentes-proyecto-anterior.md`.

## Pendiente
- Tamaño mínimo de cara en alcance (recomendación: 48 px de alto).
- Datasets de desarrollo, prueba final y «dejar uno fuera».

## Material para la presentación
- `entrega/transformaciones-blur-pixelado.html`: página interactiva (funciona sin internet; se abre con doble clic) para explicar cómo se hace el desenfoque y el pixelado con matrices.
