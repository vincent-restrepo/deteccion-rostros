# Decisiones · deteccion-rostros

D1, D2, … con el porqué.

## D1 · 2026-11-07 · Tamaño mínimo de cara en alcance: 48 px de alto
Con WIDER, el 61 % de las caras mide menos de 24 px y el recall clásico ahí es ≈ 0 (E5, E6 del proyecto anterior). 48 px equivale a una cara a unos 2 m de una cámara normal. Las caras menores quedan fuera de alcance y el informe lo dice. Aprobado por Vincent.

## D2 · 2026-11-07 · Varios datasets: entrenar con WIDER + un segundo; FDDB solo de prueba externa
Mezclar para entrenar (más variedad), agrupando repetidos con dHash ≤ 6 bits sobre el conjunto junto antes de dividir; convertir las cajas a un mismo criterio; resultados siempre por dataset. FDDB no entra al entrenamiento, para que sirva en «dejar un dataset fuera». Prueba final: partición oficial de WIDER, una sola vez. Aprobado por Vincent.

## D3 · 2026-11-14 · La demostración corre local, en el portátil de Vincent (no en la nube)
La cámara en vivo no se sube a ningún servidor: sería mandar caras sin proteger por internet, añade demora y exige HTTPS. Los métodos son clásicos (sin tarjeta gráfica), así que el cuello de botella es el procesador del portátil y se arregla bajando la carga, no con un servidor. GCP queda como plan B (máquina con muchos núcleos para las evaluaciones pesadas) y, al final, posible publicación de la parte de revisión con imágenes de prueba. Decidido por Vincent.

## D4 · 2026-11-14 · Fotos de prueba propias: compañeros del curso, con permiso, solo en el disco local
Vincent toma fotos de prueba en el aula con compañeros que también están en el proyecto y que se toman fotos de prueba entre sí para los proyectos de cada uno (permiso de ellos, confirmado por Vincent). Se guardan en `salidas_demo/`, que está fuera de git; no se suben al repo ni se usan para entrenar ni para la prueba final (para eso están WIDER y FDDB). Sirven para diagnóstico y para la demostración.

## D5 · 2026-11-14 · Diseño de la parte B (revisor de alteraciones)
- Entrada: una imagen guardada. Salida: lista de regiones alteradas, cada una con su tipo: desenfoque, pixelado o tapado (color plano). Más tipos (ruido, mosaico revuelto, oscurecer) se agregan después, de a uno, con su medición.
- Datos: no hay un dataset de caras alteradas; se genera. Se toman caras de WIDER (≥ 48 px, D1), se aplica una edición al azar (parámetros al azar; pixelado con bloque 4–64 y cuadrícula alineada al (0,0), como la app) y se guarda la región y el tipo reales. Algunas imágenes se dejan sin editar (negativos) y se recomprimen en JPEG con calidad al azar (70–95), porque las capturas reales son JPEG.
- Sin fuga: antes de generar, las imágenes de WIDER se agrupan por dHash ≤ 6 y se dividen por grupo en entrenamiento 70 % / validación 15 % / prueba 15 % (semilla fija). La prueba se usa una sola vez, al final.
- Método: rasgos clásicos por ventana (varianza del Laplaciano, energía de gradiente en la cuadrícula del pixelado, fracción de píxeles de color plano, contraste con el borde de la región) + clasificador clásico (SVM, bosque o k-NN, el mejor en validación) + unión de ventanas en cajas. Línea base trivial: «nunca hay alteración» y «todo es desenfoque».
- Honestidad: las ediciones sintéticas son las mismas que aplica la app; el revisor podría fallar con ediciones hechas en otros programas. Se declara como límite y, si hay tiempo, se prueba con una captura real de la app.

## D6 · 2026-11-14 · El revisor principal analiza los píxeles con reglas; el modelo entrenado es solo comparación
Vincent: para la clase pesa más el análisis de píxeles que entrenar un modelo. Entonces la parte B se reorganiza: (1) método principal: reglas medibles sobre los píxeles (bloques de color plano alineados a una cuadrícula → pixelado, con su tamaño de bloque; una región plana rectangular → tapado; caída brusca de nitidez en un borde rectangular → desenfoque). Los pocos umbrales se ajustan en validación y se declara cuáles son. (2) El revisor entrenado (E9: bosque aleatorio sobre rasgos) queda como línea de comparación: «¿qué ganamos entrenando?». Los datos sintéticos siguen sirviendo para medir (ya no para entrenar el método principal).

## D7 · 2026-11-14 · El desenfoque va en óvalo; el pixelado y el tapado, en rectángulo
Vincent: el desenfoque se hacía en un cuadrado y prefiere un óvalo que siga la cara (pidió solo el desenfoque). `editar(..., forma="ovalo")` usa la elipse inscrita en la región; la app lo aplica solo al desenfoque. (Una primera versión lo aplicó también al pixelado y tuvo una opción «Forma» en el panel; Vincent corrigió: solo el desenfoque. Se quitó la opción.) Consecuencia para la parte B: un desenfoque en óvalo ya no es un rectángulo completo; el generador de datos debe producir desenfoques ovalados y medirse aparte (pendiente).

## D8 · 2026-11-14 · Sliders de intensidad en la app (bloque 4–64 y desenfoque 0,2–3,0)
Vincent: el pixelado y el desenfoque deben tener cada uno un slider para mover la intensidad en vivo (le gustó el slider de la página de explicación más que los botones 4/8/16/32/64). Pixelado: bloque entero de 4 a 64 px. Desenfoque: `fuerza` de 0,2 a 3,0 (1,0 = valor anterior), relativa al tamaño de la cara. El panel muestra solo el slider del modo elegido, y al abrir la página se sincroniza el servidor con los controles.

## D9 · 2026-11-14 · Menú desplegable de filtros: desenfoque, pixelado, tapado sólido y ruido
Vincent pidió agregar ruido y tapado sólido y cambiar los botones de modo por un menú desplegable porque ya son varios. La app ofrece: Desenfoque (óvalo, slider de intensidad), Pixelado (slider de bloque 4–64), Tapado sólido (color negro, blanco o promedio de la cara) y Ruido (gaussiano, desviación 10–150; con poco ruido la cara se puede recuperar). Cada modo muestra solo sus controles. El tapado ya lo conocía el revisor (parte B); el ruido todavía no: queda pendiente agregarlo al generador de datos y medirlo (el revisor sin entrenar buscaría una varianza alta y sin estructura).

## D10 · 2026-11-14 · Tapado y ruido en óvalo; el ruido pasa a ser una mezcla; se quita «Mostrar cajas»
- Vincent: el tapado sólido y el ruido deben ir en óvalo como el desenfoque (el pixelado sigue en rectángulo, porque sus bloques son cuadros). Actualiza D7.
- El ruido gaussiano de σ hasta 150 dejaba ver mucho de la cara. Ahora el ruido mezcla la región con ruido uniforme: (1 − a)·imagen + a·ruido, con a = intensidad / 100 (slider 10–100 %). Al 100 % es ruido puro: la correlación con la cara original medida es 0,01 (al 80 %: 0,07).
- Se elimina «Mostrar cajas» (casilla, dibujo y estado del servidor): aportaba poco, según Vincent.
- Consecuencia para la parte B: los tapados ovalados ya no son rectángulos planos completos (relleno 0,785, y la regla de E10 exige 0,8); hay que ajustar la regla y medirla con óvalos (pendiente, junto con el ruido).

## D11 · 2026-11-14 · La prohibición de YOLO y del deep learning queda explícita en la documentación
Vincent pidió dejar dicho que no se puede usar YOLO, porque ahora trabaja más gente (y sus asistentes de IA) y el dataset `wider-face-for-yolo-training` puede confundir por el nombre. Se agregó un aviso destacado al inicio de `README.md` y de `AGENTS.md`, y una línea al principio de «Reglas del curso» en `brief.md`: no se usa YOLO ni ninguna red neuronal, aunque venga preentrenada; el nombre del dataset solo se refiere al formato de sus etiquetas. La regla viene del profesor (D14 del proyecto anterior).
