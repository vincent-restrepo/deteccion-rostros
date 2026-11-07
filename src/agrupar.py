"""Agrupa casi repetidos: dentro de cada dataset, dos imágenes con dHash a <= 6 bits quedan en el mismo grupo
(unión transitiva). Búsqueda por 7 trozos de 9 bits (principio del palomar: si difieren <=6 bits, algún trozo es igual).
Salida: datos/grupos.csv (dataset, ruta, grupo) y datos/grupos_resumen.csv."""
import csv, collections
import numpy as np
UMBRAL = 6
filas = [r for r in csv.DictReader(open("datos/manifiesto.csv")) if r["dhash"]]
por_ds = collections.defaultdict(list)
for i, r in enumerate(filas):
    por_ds[r["dataset"]].append(i)
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
def popcount(x):
    x = x.astype(np.uint64); t = np.zeros(x.shape, dtype=np.uint8)
    for s in range(0, 64, 8):
        t += POP[((x >> np.uint64(s)) & np.uint64(255)).astype(np.intp)]
    return t
salida = open("datos/grupos.csv", "w", newline=""); w = csv.writer(salida); w.writerow(["dataset", "ruta", "grupo"])
res = open("datos/grupos_resumen.csv", "w", newline=""); wr = csv.writer(res)
wr.writerow(["dataset", "imagenes", "grupos", "grupo_mas_grande", "imagenes_en_grupos_de_2_o_mas"])
for ds, idx in sorted(por_ds.items()):
    h = np.array([int(filas[i]["dhash"], 16) for i in idx], dtype=np.uint64)
    n = len(h); padre = list(range(n))
    def f(a):
        while padre[a] != a:
            padre[a] = padre[padre[a]]; a = padre[a]
        return a
    cortes = [(0, 9), (9, 18), (18, 27), (27, 36), (36, 45), (45, 54), (54, 64)]
    for a, b in cortes:
        clave = (h >> np.uint64(a)) & np.uint64((1 << (b - a)) - 1)
        orden = np.argsort(clave, kind="stable"); ks = clave[orden]
        ini = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]]); fin = np.r_[ini[1:], len(ks)]
        for s, e in zip(ini, fin):
            if e - s < 2: continue
            miembros = orden[s:e]
            if len(miembros) > 3000: miembros = miembros[:3000]  # cubo gigante (fondos lisos): se acota y se anota
            hh = h[miembros]
            for k in range(len(miembros) - 1):
                d = popcount(hh[k] ^ hh[k+1:])
                for j in np.flatnonzero(d <= UMBRAL):
                    ra, rb = f(int(miembros[k])), f(int(miembros[k+1+j]))
                    if ra != rb: padre[ra] = rb
    g = collections.Counter(f(k) for k in range(n))
    for k, i in enumerate(idx):
        w.writerow([ds, filas[i]["ruta"], f"{ds}:{f(k)}"])
    mult = sum(v for v in g.values() if v > 1)
    wr.writerow([ds, n, len(g), max(g.values()), mult])
    print(ds, n, len(g), max(g.values()), flush=True)
print("FIN", flush=True)
