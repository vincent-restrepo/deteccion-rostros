# Ficha técnica · generador de imágenes alteradas (`src/generar_alteraciones.py`)

Datos de la parte B (revisor). Crea imágenes de WIDER con caras alteradas y guarda la **verdad exacta** de cada alteración. Las ediciones son las mismas de la app (`src/ediciones.py`).

| Campo | Valor |
|---|---|
| Comando | `python src/generar_alteraciones.py <train\|val\|test> <N> [semilla]` |
| Entrada | `entradas/wider-yolo` (imágenes y etiquetas) y `datos/division_wider.csv` (la división sin fuga; solo toma imágenes de la división pedida) |
| Salida | `datos/alteradas/<división>/imgs/*.jpg` y `etiquetas.jsonl` (una línea por imagen). Fuera de git |
| Generadas | `train` 3.000 (semilla 1), `val` 600 (semilla 2), `test` 600 (semilla 3; no se toca hasta la evaluación final) = 4.200 |
| Reproducible | Sí: mismas semilla, división y versión de `ediciones.py` dan las mismas imágenes |
| Peso | ≈ 600 MB en total |

## Qué se altera
- Solo caras de **48 px de alto o más** y ancho ≥ 8 (regla D1). Las más pequeñas no se tocan ni se cuentan.
- Por imagen: el 15 % queda **sin editar** (negativos limpios). En las demás, **cada cara** se edita con probabilidad 0,6; las caras no editadas se guardan en `caras_sin_editar`.
- Tipo, uniforme entre **tres**: `desenfoque`, `pixelado` y `tapado`.
- Parámetros al azar por región: margen alrededor de la cara 0–0,35; pixelado con bloque ∈ {4, 6, 8, 12, 16, 24, 32, 48, 64}; desenfoque con fuerza 0,4–2,0; tapado con color negro, blanco, promedio de la cara o uno al azar.
- Si la región editada se solapa más de 0,3 (IoU) con otra ya hecha, se descarta.
- Después de editar, cada imagen se guarda como **JPEG con calidad 70–98** (la verdad no cambia, pero los bordes de los bloques se ensucian como en una imagen real).

## Formato de `etiquetas.jsonl`
Por línea: `archivo`, `origen`, `ancho`, `alto`, `jpeg_q`, `regiones` (lista de `{region: [x1,y1,x2,y2], tipo, params, cara: [x,y,ancho,alto]}`) y `caras_sin_editar`.

## Límites conocidos (lo que NO genera todavía)
- **Todas las regiones son rectángulos.** La app edita desenfoque, tapado y ruido en óvalo (D7, D10): el revisor no se mide con óvalos (pendiente).
- **No produce ruido ni reducción de colores** (D9, D12) ni mosaico (propuesta del equipo): el revisor no se mide con ellos (pendiente).
- Las caras vienen de WIDER, con pocos rostros grandes y de frente: sirven menos para el enfoque de cara principal (D13), donde las caras son mayores.
- No hay ediciones superpuestas ni recompresión doble.
- Las cifras del revisor sobre estas imágenes (E9, E10, E10b) están en `experimentos.md`.
