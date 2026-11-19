# Fichas preliminares · datasets de la propuesta del equipo (2026-11-18)

**Nada de esto está descargado ni verificado.** Las cifras vienen del PDF de propuesta del equipo (`Propuesta_ampliacion_proyecto_vision_computador.pdf`) y de lo que se sabe de cada dataset; antes de usar uno hay que descargarlo, contarlo y llenar su ficha completa en `procesamiento-dataset.md` (conteos, formato, licencia, duplicados). Solo entran si el equipo aprueba la fase correspondiente (ver `estrategia-landmarks.md`).

| Dataset | Fase | Qué trae (según la propuesta) | Licencia / condiciones | Cautelas | Estado |
|---|---|---|---|---|---|
| Facial Keypoints Detection (Kaggle, competencia) | 2 | 7.049 imágenes de entrenamiento y 1.783 de prueba, 96×96 en gris, 15 puntos (ojos, cejas, nariz, boca) | Reglas de la competencia de Kaggle | La prueba no trae etiquetas: hay que dividir `training`; faltan puntos en muchas filas; los rostros están recortados y alineados, no son de webcam (cambio de dominio) | Por descargar solo en el hito M1 |
| CelebA (`jessicali9530/celeba-dataset`) | 2 y 3 | 202.599 imágenes, 10.177 identidades, 40 atributos binarios, 5 puntos | Investigación no comercial; no se redistribuye | Fuga por identidad: separar por persona. Famosos: no subir nada al repo | No aprobado |
| FER2013 (Kaggle, competencia) | 3 | 28.709 de entrenamiento + 3.589 + 3.589, 48×48 en gris, 7 expresiones | Reglas de la competencia | Desbalance fuerte; los notebooks suelen usar redes: solo los datos. Inferir emociones es delicado: se presenta como «expresión visible», no estado emocional | No aprobado; el equipo lo posterga |
| CK+ (`davilsena/ckdataset`) | 3 | 920 imágenes 48×48, 8 clases, muy desbalanceado | Declara CC0 el publicador del mirror; el original puede tener otras condiciones | Verificar procedencia | No aprobado |
| AFLW2000-3D / LFPW | 2 (extra) | 2.000 imágenes con 68 puntos 3D / puntos de referencia | Por revisar (LFPW viene mezclado en `ngoduy/...` con licencia desconocida) | Solo si hace falta pose 3D | Fuera por ahora |
