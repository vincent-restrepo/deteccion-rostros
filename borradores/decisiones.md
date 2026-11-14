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
