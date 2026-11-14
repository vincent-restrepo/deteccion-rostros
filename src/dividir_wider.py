"""Divide WIDER (entradas/wider-yolo) sin fuga: dHash 9×8; dos imágenes a ≤ 6 bits quedan en el mismo grupo (unión transitiva);
los grupos completos se reparten 70 % entrenamiento / 15 % validación / 15 % prueba con semilla fija.
Salida: datos/division_wider.csv (ruta, dhash, grupo, division) y datos/division_wider_resumen.txt.
Uso:  .venv/bin/python src/dividir_wider.py"""
import csv, glob, os, random, collections, hashlib
import cv2, numpy as np
RAIZ = "entradas/wider-yolo/images"; UMBRAL = 6; SEMILLA = 20261114
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def dhash(ruta):
    g = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    if g is None: return None
    p = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (p[:, 1:] > p[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)

def popcount(x):
    t = np.zeros(x.shape, dtype=np.uint8)
    for s in range(0, 64, 8): t += POP[((x >> np.uint64(s)) & np.uint64(255)).astype(np.intp)]
    return t

if __name__ == "__main__":
    rutas = sorted(glob.glob(f"{RAIZ}/*.jpg")); hs = []; ok = []
    for r in rutas:
        h = dhash(r)
        if h is not None: hs.append(h); ok.append(r)
    h = np.array(hs, dtype=np.uint64); n = len(h); padre = list(range(n))
    def f(a):
        while padre[a] != a: padre[a] = padre[padre[a]]; a = padre[a]
        return a
    for i in range(n - 1):
        d = popcount(h[i] ^ h[i + 1:])
        for j in np.flatnonzero(d <= UMBRAL):
            ra, rb = f(i), f(i + 1 + int(j))
            if ra != rb: padre[ra] = rb
    grupo = [f(i) for i in range(n)]; por = collections.defaultdict(list)
    for i, g in enumerate(grupo): por[g].append(i)
    ids = sorted(por); random.Random(SEMILLA).shuffle(ids)
    div = {}; cuenta = {"train": 0, "val": 0, "test": 0}
    for g in ids:   # se asigna al conjunto que va más atrás de su cuota (70/15/15), grupo completo
        total = sum(cuenta.values()) or 1
        falta = {"train": 0.70 - cuenta["train"] / total, "val": 0.15 - cuenta["val"] / total, "test": 0.15 - cuenta["test"] / total}
        k = max(falta, key=falta.get); cuenta[k] += len(por[g])
        for i in por[g]: div[i] = k
    with open("datos/division_wider.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["ruta", "dhash", "grupo", "division"])
        for i in range(n): w.writerow([ok[i], f"{int(h[i]):016x}", f"g{grupo[i]}", div[i]])
    tam = collections.Counter(len(v) for v in por.values())
    txt = (f"imágenes: {n}\ngrupos: {len(por)}\ngrupo más grande: {max(len(v) for v in por.values())}\n"
           f"imágenes en grupos de 2 o más: {sum(len(v) for v in por.values() if len(v) > 1)}\n"
           f"división: {cuenta}\numbral dHash: {UMBRAL} bits; semilla: {SEMILLA}\n")
    open("datos/division_wider_resumen.txt", "w").write(txt); print(txt)
