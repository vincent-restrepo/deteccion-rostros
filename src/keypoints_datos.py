"""Datos de landmarks (M1 de borradores/estrategia-landmarks.md): Facial Keypoints (Kaggle, Bengio), espejo drgilermo/face-images-with-marked-landmark-points.
7.049 caras de 96×96 en gris; 15 puntos (x, y) por cara, pero solo ≈ 2.140 los tienen todos: los 4 «básicos» (centro de cada ojo, punta de la nariz, centro del labio inferior) están en ≈ 7.000.
Comandos:
  .venv/bin/python src/keypoints_datos.py dividir   → datos/division_keypoints.csv (indice, grupo, division) por grupos de repetidas (dHash ≤ 1 bit), 70/15/15, semilla 20261114
  .venv/bin/python src/keypoints_datos.py piso      → NME y RMSE del piso (la forma media de entrenamiento), medido en validación; datos/piso_keypoints.txt
Como biblioteca:  from keypoints_datos import cargar;  X, Y, nombres = cargar();  idx = indices("train")."""
import sys, os, csv, random, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RAIZ = "entradas/keypoints-fkd"; DIV = "datos/division_keypoints.csv"; UMBRAL = 1; SEMILLA = 20261114   # 1 bit, no 6: son caras recortadas parecidas y a 3+ bits el dHash las encadena en grupos de cientos (medido: a 6 bits un grupo de 1.980)
BASICOS = ("left_eye_center", "right_eye_center", "nose_tip", "mouth_center_bottom_lip")

def cargar():
    """X: (N, 96, 96) uint8; Y: (N, 15, 2) float con NaN donde falta el punto; nombres: los 15 nombres en orden."""
    from rutas import exigir; exigir(f"{RAIZ}/face_images.npz", "Descarga: python3 src/descargar_kaggle.py drgilermo/face-images-with-marked-landmark-points=entradas/keypoints-fkd")
    X = np.load(f"{RAIZ}/face_images.npz")["face_images"].transpose(2, 0, 1).round().astype(np.uint8)
    r = list(csv.reader(open(f"{RAIZ}/facial_keypoints.csv"))); cab = r[0]
    M = np.array([[float(v) if v else np.nan for v in f] for f in r[1:]], np.float32)
    return X, M.reshape(len(M), -1, 2), [c[:-2] for c in cab[::2]]

def indices(division):
    from rutas import exigir; exigir(DIV, "Primero: python src/keypoints_datos.py dividir (el CSV ya viene en el repo)")
    return np.array([int(r["indice"]) for r in csv.DictReader(open(DIV)) if r["division"] == division])

def dividir():
    from dividir_wider import dhash, popcount
    X, Y, _ = cargar(); n = len(X); os.makedirs("datos", exist_ok=True)
    import cv2
    def h(a):
        p = cv2.resize(a, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16); return int("".join("1" if b else "0" for b in (p[:, 1:] > p[:, :-1]).flatten()), 2)
    H = np.array([h(a) for a in X], dtype=np.uint64); padre = list(range(n))
    def raiz(i):
        while padre[i] != i: padre[i] = padre[padre[i]]; i = padre[i]
        return i
    for i in range(n):
        for j in np.flatnonzero(popcount(H[i + 1:] ^ H[i]) <= UMBRAL): padre[raiz(i)] = raiz(i + 1 + int(j))
    grupos = collections.defaultdict(list)
    for i in range(n): grupos[raiz(i)].append(i)
    ids = sorted(grupos); random.Random(SEMILLA).shuffle(ids); div = {}; acum = 0
    for g in ids:
        d = "train" if acum < 0.70 * n else ("val" if acum < 0.85 * n else "test"); acum += len(grupos[g])
        for i in grupos[g]: div[i] = (g, d)
    with open(DIV, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["indice", "grupo", "division"])
        for i in range(n): w.writerow([i, div[i][0], div[i][1]])
    cuenta = collections.Counter(d for _, d in div.values()); mult = sum(1 for g in grupos.values() if len(g) > 1)
    print("imágenes:", n, "| grupos:", len(grupos), "| grupos con casi repetidas:", mult, "| división:", dict(cuenta))

def nme(P, V, ojos=(0, 1)):
    """Error medio de los puntos básicos normalizado por la distancia entre ojos de cada imagen (promedio sobre puntos y caras con el punto presente)."""
    d = np.linalg.norm(V[:, ojos[0]] - V[:, ojos[1]], axis=1); e = np.linalg.norm(P - V, axis=2) / d[:, None]
    return float(np.nanmean(e))

def piso():
    X, Y, nombres = cargar(); tr, va = indices("train"), indices("val"); media = np.nanmean(Y[tr], axis=0)
    idx = [nombres.index(b) for b in BASICOS]; ok = ~np.isnan(Y[va][:, idx]).any(axis=(1, 2))
    V = Y[va][ok][:, idx]; P = np.broadcast_to(media[idx], V.shape)
    e_px = np.sqrt(np.nanmean(((np.broadcast_to(media, Y[va].shape) - Y[va]) ** 2)))
    # normalizado por la distancia entre ojos: sobre los 4 puntos básicos y, por separado, por punto
    por = {BASICOS[k]: round(float(np.nanmean(np.linalg.norm(P[:, k] - V[:, k], axis=1) / np.linalg.norm(V[:, 0] - V[:, 1], axis=1))), 3) for k in range(4)}
    txt = (f"Piso (forma media de train) medido en validación\nimágenes train {len(tr)}, val {len(va)}; caras de val con los 4 puntos básicos: {int(ok.sum())}\n"
           f"NME (4 puntos básicos, distancia entre ojos): {nme(P, V):.3f}\nNME por punto: {por}\nRMSE en píxeles, todos los puntos disponibles: {e_px:.2f}\n"
           f"Distancia media entre ojos en val: {float(np.mean(np.linalg.norm(V[:, 0] - V[:, 1], axis=1))):.1f} px (cara de 96×96)\n")
    print(txt); open("datos/piso_keypoints.txt", "w").write(txt)

if __name__ == "__main__":
    {"dividir": dividir, "piso": piso}[sys.argv[1]]()
