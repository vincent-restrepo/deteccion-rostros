"""División sin fuga del segundo dataset (entradas/faces-openimages, D16) a partir de datos/openimages_dhash.csv (src/inspeccionar_openimages.py).
Reglas (dHash <= 6 bits = «casi igual»):
  1. Si una imagen es casi igual a una de las pruebas finales (WIDER_val oficial o FDDB) -> división «excluida»: no se usa para nada.
  2. Si es casi igual a una de wider-yolo (= WIDER_train) -> hereda la división de esa imagen en datos/division_wider.csv (no se toca esa división).
  3. El resto se agrupa (unión transitiva de casi repetidas) y los grupos completos se reparten 70/15/15 con semilla 20261114.
Se ignora la división original del dataset (train/val): usamos la nuestra, por grupos.
Salida: datos/division_openimages.csv (ruta, dhash, grupo, division, motivo) y datos/division_openimages_resumen.txt.
Uso:  .venv/bin/python src/dividir_openimages.py"""
import csv, glob, os, sys, random, collections
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dividir_wider import dhash, popcount, UMBRAL, SEMILLA
from rutas import exigir

def hashes(patron):
    rutas = sorted(glob.glob(patron, recursive=True))
    with ProcessPoolExecutor() as ex: hs = list(ex.map(dhash, rutas, chunksize=64))
    return np.array([x for x in hs if x is not None], dtype=np.uint64)

def cerca(h, base):
    d = popcount(base ^ h); j = int(np.argmin(d)); return (j if d[j] <= UMBRAL else None)

if __name__ == "__main__":
    exigir("datos/openimages_dhash.csv", "Primero: python src/inspeccionar_openimages.py"); exigir("datos/division_wider.csv", "Primero: python src/dividir_wider.py")
    filas = list(csv.DictReader(open("datos/openimages_dhash.csv"))); h = np.array([int(r["dhash"], 16) for r in filas], dtype=np.uint64)
    wider = list(csv.DictReader(open("datos/division_wider.csv"))); wh = np.array([int(r["dhash"], 16) for r in wider], dtype=np.uint64)
    pruebas = np.concatenate([hashes("entradas/wider-mksaad/WIDER_val/**/*.jpg"), hashes("entradas/fddb/**/*.jpg")])
    n = len(filas); div = [None] * n; motivo = [""] * n
    for i in range(n):
        if cerca(h[i], pruebas) is not None: div[i], motivo[i] = "excluida", "casi igual a WIDER_val o FDDB (pruebas finales)"
        else:
            j = cerca(h[i], wh)
            if j is not None: div[i], motivo[i] = wider[j]["division"], "hereda la división de una imagen casi igual de wider-yolo"
    resto = [i for i in range(n) if div[i] is None]; padre = {i: i for i in resto}
    def f(a):
        while padre[a] != a: padre[a] = padre[padre[a]]; a = padre[a]
        return a
    hr = h[resto]
    for k, i in enumerate(resto[:-1]):
        for j in np.flatnonzero(popcount(hr[k + 1:] ^ h[i]) <= UMBRAL):
            ra, rb = f(i), f(resto[k + 1 + int(j)])
            if ra != rb: padre[ra] = rb
    por = collections.defaultdict(list)
    for i in resto: por[f(i)].append(i)
    ids = sorted(por); random.Random(SEMILLA).shuffle(ids); cuenta = {"train": 0, "val": 0, "test": 0}
    for g in ids:
        c = sum(cuenta.values()); d = "train" if cuenta["train"] < 0.70 * (c + len(por[g])) else ("val" if cuenta["val"] < 0.15 * (c + len(por[g])) else "test")
        for i in por[g]: div[i], motivo[i] = d, "grupo repartido 70/15/15"
        cuenta[d] += len(por[g])
    gid = {g: k for k, g in enumerate(ids)}
    with open("datos/division_openimages.csv", "w", newline="") as fo:
        w = csv.writer(fo); w.writerow(["ruta", "dhash", "grupo", "division", "motivo"])
        for i, r in enumerate(filas): w.writerow([r["ruta"], r["dhash"], f"g{gid[f(i)]}" if i in padre else "", div[i], motivo[i]])
    c = collections.Counter(div); m = collections.Counter(motivo)
    res = [f"# División de faces-openimages ({n} imágenes), semilla {SEMILLA}, dHash <= {UMBRAL} bits", f"Por división: {dict(c)}", "Por motivo: " + str(dict(m))]
    open("datos/division_openimages_resumen.txt", "w").write("\n".join(res) + "\n"); print("\n".join(res))
