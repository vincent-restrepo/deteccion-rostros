"""Conjunto de trabajo «cara principal» (D13): imágenes con 1 a MAX_CARAS caras cuya cara más grande mide MIN_ALTO px de alto o más,
de los dos datasets (WIDER = wider-yolo y Open Images = faces-openimages), con las cajas en un solo formato (x, y, ancho, alto en píxeles).
Usa las divisiones sin fuga: datos/division_wider.csv y datos/division_openimages.csv (las imágenes «excluida» no se devuelven nunca).
Desde otro script:   from conjunto_principal import cargar
                     for m in cargar("train"): m["ruta"], m["cajas"], m["principal"], m["dataset"]
Por consola:         .venv/bin/python src/conjunto_principal.py   (escribe datos/conjunto_principal_resumen.txt)
Los umbrales (96 px, 3 caras) son los de D13; si se cambian, es una decisión nueva."""
import csv, os, sys, collections
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rutas import exigir
MIN_ALTO = 96; MAX_CARAS = 3

def _cajas_wider(ruta):
    txt = ruta.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    if not os.path.exists(txt): return []
    W, H = Image.open(ruta).size; out = []
    for l in open(txt).read().split("\n"):
        v = l.split()
        if len(v) >= 5:
            cx, cy, w, h = (float(x) for x in v[1:5]); out.append((int((cx - w / 2) * W), int((cy - h / 2) * H), int(w * W), int(h * H)))
    return out

def _cajas_openimages(ruta):
    txt = ruta.replace("/images/train/", "/labels2/").replace("/images/val/", "/labels2/").rsplit(".", 1)[0] + ".txt"
    if not os.path.exists(txt): return []
    out = []
    for l in open(txt).read().split("\n"):
        v = l.split()
        if len(v) >= 4:
            x1, y1, x2, y2 = (float(x) for x in v[-4:]); out.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
    return out

DATASETS = {"wider": ("datos/division_wider.csv", _cajas_wider), "openimages": ("datos/division_openimages.csv", _cajas_openimages)}

def cargar(division, datasets=("wider", "openimages"), min_alto=MIN_ALTO, max_caras=MAX_CARAS):
    """Lista de dicts {dataset, ruta, cajas, principal} de la división pedida («train», «val» o «test») que cumplen la regla de la cara principal."""
    out = []
    for d in datasets:
        csv_ruta, leer = DATASETS[d]; exigir(csv_ruta, "Genera la división: ver README.md / AGENTS.md.")
        for r in csv.DictReader(open(csv_ruta)):
            if r["division"] != division: continue
            cajas = leer(r["ruta"])
            if 1 <= len(cajas) <= max_caras and max(c[3] for c in cajas) >= min_alto:
                out.append({"dataset": d, "ruta": r["ruta"], "cajas": cajas, "principal": max(cajas, key=lambda c: c[3])})
    return out

if __name__ == "__main__":
    res = [f"# Conjunto «cara principal» (D13): 1 a {MAX_CARAS} caras y la mayor >= {MIN_ALTO} px de alto"]
    for div in ("train", "val", "test"):
        c = cargar(div); por = collections.Counter(m["dataset"] for m in c); caras = collections.Counter(len(m["cajas"]) for m in c)
        res.append(f"{div}: {len(c)} imágenes  {dict(por)}  caras por imagen {dict(sorted(caras.items()))}")
    open("datos/conjunto_principal_resumen.txt", "w").write("\n".join(res) + "\n"); print("\n".join(res))
