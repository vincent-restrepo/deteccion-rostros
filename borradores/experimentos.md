# Experimentos · deteccion-rostros

E1, E2, … con configuración y resultado. (Los E1–E6 del proyecto anterior están en el repo desenfoque-privacidad.)


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
