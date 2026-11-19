"""Comprueba si WIDER oficial (entradas/wider-mksaad: WIDER_train y WIDER_val) comparte imágenes casi idénticas con
wider-yolo (entradas/wider-yolo, ya divididas en datos/division_wider.csv). dHash 9x8; «casi igual» = distancia <= 6 bits (como en la división).
Importa para la prueba final: si WIDER_val repite imágenes con las de entrenamiento de wider-yolo, no sería una prueba limpia.
Salida: datos/comprobacion_wider_oficial.txt (resumen; versionado) y datos/comprobacion_wider_oficial.csv (pares; versionado).
Uso:  .venv/bin/python src/comprobar_wider_oficial.py"""
import csv, glob, os, sys, collections
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dividir_wider import dhash, popcount, UMBRAL
from rutas import exigir
RAIZ = "entradas/wider-mksaad"

def h(r): return dhash(r)

def hashes(carpeta):
    rutas = sorted(glob.glob(f"{RAIZ}/{carpeta}/**/*.jpg", recursive=True))
    with ProcessPoolExecutor() as ex: hs = list(ex.map(h, rutas, chunksize=64))
    ok = [(r, x) for r, x in zip(rutas, hs) if x is not None]
    return [r for r, _ in ok], np.array([x for _, x in ok], dtype=np.uint64)

if __name__ == "__main__":
    exigir("datos/division_wider.csv", "Primero: python src/dividir_wider.py."); exigir(RAIZ, "Descarga WIDER oficial: ver AGENTS.md.")
    base = list(csv.DictReader(open("datos/division_wider.csv")))
    bh = np.array([int(r["dhash"], 16) if not r["dhash"].isdigit() or len(r["dhash"]) == 16 else int(r["dhash"]) for r in base], dtype=np.uint64)
    out = ["# Comprobación WIDER oficial (mksaad) contra wider-yolo (dHash 9x8, casi igual = <= %d bits)" % UMBRAL]
    filas = []
    for nombre, carpeta in (("WIDER_val", "WIDER_val"), ("WIDER_train", "WIDER_train")):
        rutas, hs = hashes(carpeta); por_div = collections.Counter(); exactas = 0; coinciden = 0
        for r, x in zip(rutas, hs):
            d = popcount(bh ^ x); j = int(np.argmin(d))
            if d[j] <= UMBRAL:
                coinciden += 1; por_div[base[j]["division"]] += 1; exactas += int(d[j] == 0)
                filas.append((nombre, os.path.relpath(r, RAIZ), base[j]["ruta"], int(d[j]), base[j]["division"]))
        out.append(f"{nombre}: {len(rutas)} imágenes; {coinciden} casi iguales a alguna de wider-yolo ({exactas} con distancia 0); por división de wider-yolo: {dict(por_div)}")
    open("datos/comprobacion_wider_oficial.txt", "w").write("\n".join(out) + "\n")
    with open("datos/comprobacion_wider_oficial.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["oficial", "imagen_oficial", "imagen_wider_yolo", "bits", "division_wider_yolo"]); w.writerows(filas)
    print("\n".join(out))
