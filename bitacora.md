# Bitácora · deteccion-rostros

- 2026-11-07: Proyecto creado con brief, registros vacíos, código base copiado de desenfoque-privacidad y repo git local. Siguiente paso: lista de datasets de rostros (tamaños y licencias) y definir el tamaño mínimo de cara.
- 2026-11-07: Lista de candidatos con tamaños y licencias (consulta a la API de Kaggle, nada descargado) en borradores/procesamiento-dataset.md. Siguiente paso: Vincent confirma el tamaño mínimo de cara (propuesta: 48 px) y los datasets.
- 2026-11-07: Entorno .venv probado (OpenCV 4.10, dlib 20, numpy 2.5.4), repo subido a vincent-restrepo/deteccion-rostros y prioridades.md actualizado. Propuesta de datasets: desarrollo WIDER, prueba final partición oficial, dejar uno fuera FDDB; sin decidir hasta abrir los datos. Siguiente paso: respuesta de Vincent sobre el tamaño mínimo de cara (48 px).
- 2026-11-07: Vincent aprueba D1 (tamaño mínimo 48 px de alto) y D2 (entrenar con varios datasets, FDDB solo de prueba externa). Descarga en segundo plano de WIDER (lylmsc), WIDER con divisiones (mksaad) y FDDB (faizanarif15) a entradas/. Siguiente paso: abrirlos, contar imágenes, caras, tamaños y repetidos.
- 2026-11-07: Descarga de los tres datasets en curso (log en entradas/descarga.log); sin cambios de decisión. Siguiente paso: al terminar, abrirlos y registrar conteos, tamaños de cara y repetidos.
