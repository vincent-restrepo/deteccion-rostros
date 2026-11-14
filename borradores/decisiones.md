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

## D7 · 2026-11-14 · Forma de la zona escondida: óvalo (por defecto) o rectángulo
Vincent: el desenfoque se hacía en un cuadrado y prefiere un óvalo que siga la cara. `editar(..., forma="ovalo")` usa la elipse inscrita en la región; el panel tiene la opción «Forma» (óvalo por defecto, rectángulo). Consecuencia para la parte B: las regiones con óvalo ya no son rectángulos completos; el revisor (E10) asume bloques y regiones rectangulares, así que hay que medirlo con ambas formas (pendiente: el generador de datos debe producir las dos y reportar por separado).
