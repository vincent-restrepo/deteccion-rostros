"""Inspecciona el candidato `fareselmenshawii/face-detection-dataset` (entradas/faces-openimages: imágenes de Open Images con cajas de caras).
Cuenta imágenes y caras, tamaños de cara y caras por imagen, y compara con dHash (<= 6 bits) contra wider-yolo (= WIDER_train), WIDER_val oficial y FDDB
para detectar repetidos (fuga) y consigo mismo (duplicados internos). No usa redes neuronales.
Salida: datos/inspeccion_openimages.txt (resumen) y datos/openimages_dhash.csv (ruta, dhash, caras, alto_mayor, division_original).
Uso:  .venv/bin/python src/inspeccionar_openimages.py"""
import csv, glob, os, sys, collections
import numpy as np, cv2
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dividir_wider import dhash, popcount, UMBRAL
from rutas import exigir
RAIZ = "entradas/faces-openimages"

def info(ruta):
    im = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    if im is None: return None
    H, W = im.shape; h = dhash(ruta)
    txt = ruta.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"; hs = []
    if os.path.exists(txt):
        for l in open(txt).read().split("\n"):
            v = l.split()
            if len(v) >= 5: hs.append(float(v[4]) * H)
    return ruta, h, W, H, sorted(hs, reverse=True)

def hashes(patron):
    rutas = sorted(glob.glob(patron, recursive=True))
    with ProcessPoolExecutor() as ex: hs = list(ex.map(dhash, rutas, chunksize=64))
    ok = [(r, x) for r, x in zip(rutas, hs) if x is not None]
    return [r for r, _ in ok], np.array([x for _, x in ok], dtype=np.uint64)

def cruza(a, b, nombre, out):
    """cuántas imágenes de a (hashes) tienen alguna casi igual en b"""
    n = 0
    for x in a:
        if (popcount(b ^ x) <= UMBRAL).any(): n += 1
    out.append(f"  contra {nombre}: {n} de {len(a)} imágenes casi iguales (<= {UMBRAL} bits)"); return n

if __name__ == "__main__":
    exigir(RAIZ, "Descarga: python3 src/descargar_kaggle.py fareselmenshawii/face-detection-dataset=entradas/faces-openimages")
    rutas = sorted(glob.glob(f"{RAIZ}/images/*/*.jpg"))
    with ProcessPoolExecutor() as ex: datos = [d for d in ex.map(info, rutas, chunksize=64) if d]
    out = ["# Inspección de fareselmenshawii/face-detection-dataset (Open Images)"]
    out.append(f"Imágenes leídas: {len(datos)} de {len(rutas)}")
    div = collections.Counter(d[0].split("/images/")[1].split("/")[0] for d in datos); out.append(f"Por división original: {dict(div)}")
    n = np.array([len(d[4]) for d in datos]); mayor = np.array([d[4][0] if d[4] else 0 for d in datos]); todas = np.array([h for d in datos for h in d[4]])
    out.append(f"Caras: {len(todas)} en total; imágenes sin caras: {(n == 0).sum()}")
    out.append("Caras por imagen: " + ", ".join(f"{k}:{(n == k).sum()}" for k in range(1, 6)) + f", 6 o más:{(n >= 6).sum()}")
    q = lambda a: ", ".join(f"p{p}={np.percentile(a, p):.0f}" for p in (10, 25, 50, 75, 90))
    out.append(f"Alto de cualquier cara (px): {q(todas)}; menores de 48 px: {(todas < 48).mean():.1%}")
    out.append(f"Alto de la cara más grande por imagen (px): {q(mayor[n > 0])}")
    for k in (1, 3):
        for mn in (48, 96):
            out.append(f"  imágenes con <= {k} caras y la mayor >= {mn} px: {((n <= k) & (n > 0) & (mayor >= mn)).sum()}")
    W = np.array([d[2] for d in datos]); H = np.array([d[3] for d in datos])
    out.append(f"Tamaño de imagen: ancho mediano {np.median(W):.0f}, alto mediano {np.median(H):.0f}")
    h = np.array([d[1] for d in datos], dtype=np.uint64)
    out.append("Duplicados internos (dHash): " + str(sum(int((popcount(h[i + 1:] ^ h[i]) <= UMBRAL).any()) for i in range(len(h) - 1))) + " imágenes con otra casi igual dentro del mismo dataset")
    out.append("Repetidos contra otros datasets:")
    base = list(csv.DictReader(open("datos/division_wider.csv")))
    bh = np.array([int(r["dhash"], 16) for r in base], dtype=np.uint64); cruza(h, bh, "wider-yolo (WIDER_train)", out)
    for nombre, pat in (("WIDER_val oficial", "entradas/wider-mksaad/WIDER_val/**/*.jpg"), ("FDDB", "entradas/fddb/**/*.jpg")):
        _, hh = hashes(pat); cruza(h, hh, nombre, out)
    open("datos/inspeccion_openimages.txt", "w").write("\n".join(out) + "\n")
    with open("datos/openimages_dhash.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["ruta", "dhash", "caras", "alto_mayor_px"])
        for d in datos: w.writerow([d[0], f"{d[1]:016x}", len(d[4]), round(d[4][0], 1) if d[4] else 0])
    print("\n".join(out))
