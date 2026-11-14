"""Genera imágenes con caras alteradas (datos de la parte B). Parte de WIDER según datos/division_wider.csv (sin fuga),
aplica ediciones al azar a caras de >= 48 px de alto (D1) y guarda la verdad exacta: región, tipo y parámetros.
Salida: datos/alteradas/<division>/imgs/*.jpg y datos/alteradas/<division>/etiquetas.jsonl (las imágenes no van a git).
Uso:  .venv/bin/python src/generar_alteraciones.py <train|val|test> <N> [semilla]"""
import sys, os, csv, json, random
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ediciones import editar, TIPOS
MIN_ALTO = 48; BLOQUES = (4, 6, 8, 12, 16, 24, 32, 48, 64); P_SIN_EDITAR = 0.15; P_CARA = 0.6

def caras(ruta_img, W, H):
    txt = ruta_img.replace("/images/", "/labels/").replace(".jpg", ".txt"); out = []
    if not os.path.exists(txt): return out
    for l in open(txt).read().split("\n"):
        v = l.split()
        if len(v) >= 5:
            cx, cy, w, h = (float(x) for x in v[1:5]); out.append((int((cx - w / 2) * W), int((cy - h / 2) * H), int(w * W), int(h * H)))
    return [c for c in out if c[3] >= MIN_ALTO and c[2] >= 8]

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1); u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i / u if u else 0

if __name__ == "__main__":
    div, n = sys.argv[1], int(sys.argv[2]); sem = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    from rutas import exigir; exigir("datos/division_wider.csv", "Primero: python src/dividir_wider.py (ver README.md).")
    rng = random.Random(sem)
    rutas = [r["ruta"] for r in csv.DictReader(open("datos/division_wider.csv")) if r["division"] == div]
    rng.shuffle(rutas); carpeta = f"datos/alteradas/{div}"; os.makedirs(f"{carpeta}/imgs", exist_ok=True)
    sal = open(f"{carpeta}/etiquetas.jsonl", "w"); hechas = 0
    for ruta in rutas:
        if hechas >= n: break
        im = cv2.imread(ruta)
        if im is None: continue
        H, W = im.shape[:2]; cs = caras(ruta, W, H)
        if not cs: continue
        hechas += 1; salida = im.copy(); regiones = []; sin = []
        editar_img = rng.random() > P_SIN_EDITAR
        for c in cs:
            if not editar_img or rng.random() > P_CARA: sin.append(list(c)); continue
            tipo = rng.choice(TIPOS); prm = {"margen": round(rng.uniform(0.0, 0.35), 3)}
            if tipo == "pixelado": prm["bloque"] = rng.choice(BLOQUES)
            elif tipo == "desenfoque": prm["fuerza"] = round(rng.uniform(0.4, 2.0), 3)
            else: prm["color"] = rng.choice([(0, 0, 0), (255, 255, 255), tuple(int(v) for v in im[max(0,c[1]):c[1]+c[3], max(0,c[0]):c[0]+c[2]].reshape(-1, 3).mean(0)), tuple(rng.randint(0, 255) for _ in range(3))])
            reg = editar(salida, c, tipo, **prm)
            if reg is None or any(iou(reg, r["region"]) > 0.3 for r in regiones): continue
            prm = {k: (list(v) if isinstance(v, tuple) else v) for k, v in prm.items()}
            regiones.append({"region": list(reg), "tipo": tipo, "params": prm, "cara": list(c)})
        q = rng.randint(70, 98); nombre = os.path.basename(ruta)
        cv2.imwrite(f"{carpeta}/imgs/{nombre}", salida, [cv2.IMWRITE_JPEG_QUALITY, q])
        sal.write(json.dumps({"archivo": nombre, "origen": ruta, "ancho": W, "alto": H, "jpeg_q": q, "regiones": regiones, "caras_sin_editar": sin}) + "\n")
    sal.close(); print(div, "imágenes generadas:", hechas)
