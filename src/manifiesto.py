"""Manifiesto: SHA-256 (archivo crudo) + dHash 64 bits + tamaño de cada imagen en entradas/.
Salida: datos/manifiesto.csv (dataset, ruta relativa, bytes, sha256, dhash_hex, ancho, alto)."""
import csv, hashlib, os, sys
from multiprocessing import Pool
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
RAIZ = "entradas"

def dhash(img):
    g = img.convert("L").resize((9, 8), Image.BILINEAR)
    px = list(g.getdata())
    bits = 0
    for r in range(8):
        for c in range(8):
            bits = (bits << 1) | (px[r*9+c] > px[r*9+c+1])
    return f"{bits:016x}"

def proc(ruta):
    rel = os.path.relpath(ruta, RAIZ)
    ds = rel.split(os.sep)[0]
    try:
        h = hashlib.sha256()
        with open(ruta, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        with Image.open(ruta) as im:
            w, hh = im.size
            if im.format == "JPEG":
                im.draft("L", (64, 64))
            d = dhash(im)
        return (ds, rel, os.path.getsize(ruta), h.hexdigest(), d, w, hh)
    except Exception as e:
        return (ds, rel, os.path.getsize(ruta), "", "", 0, 0)

if __name__ == "__main__":
    rutas = []
    for dp, _, fs in os.walk(RAIZ):
        for f in fs:
            if os.path.splitext(f)[1].lower() in EXT:
                rutas.append(os.path.join(dp, f))
    rutas.sort()
    print("imágenes:", len(rutas), flush=True)
    with Pool(max(2, os.cpu_count() - 2)) as p, open("datos/manifiesto.csv", "w", newline="") as out:
        w = csv.writer(out)
        w.writerow(["dataset", "ruta", "bytes", "sha256", "dhash", "ancho", "alto"])
        for i, fila in enumerate(p.imap(proc, rutas, chunksize=64)):
            w.writerow(fila)
            if i % 10000 == 0:
                print(i, flush=True)
    print("FIN", flush=True)
