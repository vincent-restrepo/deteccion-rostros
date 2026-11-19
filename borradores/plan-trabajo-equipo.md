# Plan de trabajo del equipo · ampliación por frentes (borrador del 2026-11-19)

Base: la propuesta del equipo (PDF del 18 de noviembre), el enfoque de la cara principal (D13), los datos (D14, D16) y los hitos de landmarks (D15, `estrategia-landmarks.md`). Los tiempos son estimaciones mías; la fecha límite (≈ fin de noviembre) sigue sin confirmar con el profesor.

## Principio
Primero **detectar muy bien la cara principal** y medirlo; con esa base segura se intentan las ampliaciones, una por una, cada una cerrada y medida antes de la siguiente. Lo que no llegue se presenta como prototipo, no como terminado.

## Calendario propuesto
| Cuándo | Qué |
|---|---|
| Jue 19 (hoy) | Datos y reglas listos (esta rama); reunión del equipo: confirmar datasets, umbral de 96 px, reparto y metas |
| Vie 20 – Mar 24 | Los cinco frentes trabajan en paralelo, con PRs pequeños |
| Mié 25 | **Congelar alcance**: lo que no está medido pasa a «prototipo» o se pospone |
| Jue 26 – Sáb 28 | Integración en la app, prueba con cámara (capturas consentidas, solo en local), informe |
| Dom 29 – Lun 30 | Prueba final **una sola vez** (`WIDER_val` sin las 19 repetidas y FDDB), presentación |

## Frentes (cada uno en su rama `trabajo/<tema>` y sus archivos)
| # | Frente | Entradas que ya están listas | Primer paso concreto | Entregable y criterio de salida |
|---|---|---|---|---|
| 1 | **Detector y app** | `detector_caras.py`, resultados E7/E11, dlib con umbral −0,4 | Medir los detectores actuales con la regla de D13 y meter dlib (umbral bajo) en `app_blur.py` | App con el mejor detector del frente 2 y cajas más justas; mejora medida en el conjunto de la cara principal |
| 2 | **Entrenamiento del detector** | `conjunto_principal.cargar()` (cara principal de WIDER + Open Images, D16, ya con división sin fuga), `verificador.py` | Entrenar HOG+SVM propio con caras principales y **negativos difíciles** (los falsos positivos que ya comete el detector) | Modelo comparado contra dlib y Haar por dataset; si no gana, se reporta |
| 3 | **Revisor por reglas** | `analisis_pixeles.py`, generador y su ficha | Agregar al generador óvalos, ruido, reducción de colores y mosaico; ajustar las reglas | Recall/precisión por tipo de alteración, con el desenfoque (56/154 hoy) como punto a mejorar |
| 4 | **Revisor entrenado** | `revisor.py` (precisión 0,22), mismos datos | Rasgos nuevos (periodicidad por Fourier, gradientes por bloque, forma de la máscara) y negativos difíciles; probar «las reglas proponen, el bosque confirma» | Precisión sube sin perder recall; comparación honesta con las reglas |
| 5 | **Mosaico, calidad y medición** | `ediciones.py`, guía de calidad de la propuesta (nitidez, brillo, contraste, ruido, tamaño) | Mosaico con semilla fija en `ediciones.py`; módulo `calidad.py`; mantener `experimentos.md` y la bitácora ordenados | Filtro reproducible, panel de calidad y evaluación del revisor frente al mosaico |

**Después de M0** (detector estable, declarado por el equipo): landmarks por hitos M1→M6 (`estrategia-landmarks.md`) y, con tiempo, los dos efectos baratos (caricatura y contornos). **Fuera por ahora:** expresiones, atributos más allá de gafas y `solvePnP`.

## Reglas para todos
- Todo por PR; una línea de bitácora por avance; experimentos definidos antes de correrlos y con sus fallos anotados.
- Entrenar solo con `train`; escoger umbrales con `val`; la prueba final se toca una sola vez.
- Sin redes neuronales (D11). Sin fotos de personas en el repo.
- Cada módulo se puede apagar sin romper la cámara ni los filtros.
- Las metas numéricas las fija el equipo antes de empezar cada frente; no se prometen cifras sin medirlas.
