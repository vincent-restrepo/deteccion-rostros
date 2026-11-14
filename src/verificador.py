"""Segunda etapa para el detector de caras (E11): candidatas permisivas (Haar frontal ×2, Haar de perfil ×2, dlib HOG con umbral bajo)
+ un HOG + SVM propio que decide cuáles son caras. Negativos = las falsas alarmas de los propios detectores (negativos difíciles).
Uso:  .venv/bin/python src/verificador.py extraer <train|val> <N> [semilla]   (guarda datos/cand_<div>.pkl)
      .venv/bin/python src/verificador.py evaluar <N_train>                    (entrena con train, mide en val)"""
import sys, os, csv, pickle, random, time, json
import cv2, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluar_caras import etiquetas, iou as iou_xywh
ANCHO = 800; MIN_ALCANCE = 48
_cas = {}
def cascadas():
    if not _cas:
        for k, n in (("alt2", "haarcascade_frontalface_alt2.xml"), ("alt", "haarcascade_frontalface_alt.xml"), ("perfil", "haarcascade_profileface.xml")):
            _cas[k] = cv2.CascadeClassifier(cv2.data.haarcascades + n)
        import dlib; _cas["dlib"] = dlib.get_frontal_face_detector()
    return _cas
FUENTES = ("alt2", "alt", "perfil", "perfil_esp", "dlib")
_hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)

def candidatas(g):
    """Lista de (x, y, w, h, fuente, puntaje). Permisivo a propósito: la segunda etapa filtra."""
    c = cascadas(); out = []
    for k in ("alt2", "alt"):
        for (x, y, w, h) in c[k].detectMultiScale(g, 1.05, 1, minSize=(24, 24)): out.append((x, y, w, h, k, 0.0))
    for esp in (False, True):
        im = cv2.flip(g, 1) if esp else g
        for (x, y, w, h) in c["perfil"].detectMultiScale(im, 1.08, 2, minSize=(24, 24)):
            out.append((g.shape[1] - x - w if esp else x, y, w, h, "perfil_esp" if esp else "perfil", 0.0))
    rects, scores, _ = c["dlib"].run(g, 1, -0.4)
    for r, s in zip(rects, scores): out.append((max(0, r.left()), max(0, r.top()), r.width(), r.height(), "dlib", float(s)))
    return out

def rasgos(g, cands):
    """HOG 64×64 del recorte (con 15 % de margen) + fuente, tamaño y acuerdo entre detectores."""
    H, W = g.shape; X = []
    for i, (x, y, w, h, f, s) in enumerate(cands):
        mx, my = int(w * .15), int(h * .15); x0, y0, x1, y1 = max(0, x - mx), max(0, y - my), min(W, x + w + mx), min(H, y + h + my)
        r = cv2.resize(g[y0:y1, x0:x1], (64, 64), interpolation=cv2.INTER_AREA) if x1 > x0 and y1 > y0 else np.zeros((64, 64), np.uint8)
        hog = _hog.compute(r).ravel()
        sol = [c for c in cands if c is not cands[i] and iou_xywh(c[:4], (x, y, w, h)) > 0.3]
        acuerdo = len({c[4] for c in sol}) + 1
        extra = [1.0 if f == k else 0.0 for k in FUENTES] + [s, np.log(h), acuerdo, len(sol)]
        X.append(np.concatenate([hog, np.array(extra, np.float32)]))
    return np.array(X, np.float32) if X else np.zeros((0, 1764 + len(FUENTES) + 4), np.float32)

def una(args):
    ruta = args
    im = cv2.imread(ruta); H, W = im.shape[:2]; esc = min(1.0, ANCHO / W)
    g = cv2.cvtColor(cv2.resize(im, (int(W * esc), int(H * esc))) if esc < 1 else im, cv2.COLOR_BGR2GRAY)
    gt = etiquetas(ruta.replace("/images/", "/labels/").replace(".jpg", ".txt"), W, H)
    gt_s = [tuple(v * esc for v in b) for b in gt]; dentro = [b[3] >= MIN_ALCANCE for b in gt]
    cs = candidatas(g)
    return {"ruta": ruta, "esc": esc, "gt": gt_s, "dentro": dentro, "cands": cs, "X": rasgos(g, cs)}

def extraer(div, n, sem=1):
    rutas = [r["ruta"] for r in csv.DictReader(open("datos/division_wider.csv")) if r["division"] == div]
    random.Random(sem).shuffle(rutas); rutas = [r for r in rutas if os.path.exists(r.replace("/images/", "/labels/").replace(".jpg", ".txt"))][:n]
    t0 = time.time()
    with Pool(3) as p: res = p.map(una, rutas, chunksize=10)
    pickle.dump(res, open(f"datos/cand_{div}.pkl", "wb")); print(div, len(res), "imágenes;", sum(len(r["cands"]) for r in res), "candidatas;", round(time.time() - t0), "s", flush=True)

def etiquetar(r):
    """1 = candidata que acierta una cara en alcance; 0 = no toca ninguna cara; -1 = ambigua (toca una cara fuera de alcance o IoU intermedio)."""
    y = []
    for c in r["cands"]:
        mi = max([iou_xywh(c[:4], g) for g, d in zip(r["gt"], r["dentro"]) if d] or [0]); mo = max([iou_xywh(c[:4], g) for g in r["gt"]] or [0])
        y.append(1 if mi >= 0.5 else (0 if mo < 0.2 else -1))
    return np.array(y)

def nms(items, umbral=0.3):   # items: (caja, puntaje); se queda con el mayor puntaje
    items = sorted(items, key=lambda t: -t[1]); fin = []
    for it in items:
        if all(iou_xywh(it[0], f[0]) < umbral for f in fin): fin.append(it)
    return fin

def medir(res, puntajes, t):
    tp = fp = fn = 0
    for r, sc in zip(res, puntajes):
        pred = nms([(c[:4], s) for c, s in zip(r["cands"], sc) if s >= t]); usado = set(); dentro = [j for j, d in enumerate(r["dentro"]) if d]
        for box, _ in pred:
            m = [(iou_xywh(box, r["gt"][j]), j) for j in dentro if j not in usado]
            if m and max(m)[0] >= 0.5: usado.add(max(m)[1]); tp += 1; continue
            if any(iou_xywh(box, g) >= 0.3 for j, g in enumerate(r["gt"]) if j not in dentro): continue
            fp += 1
        fn += len(dentro) - len(usado)
    return tp, fp, fn

def evaluar(n_train):
    from sklearn.svm import LinearSVC
    tr = pickle.load(open("datos/cand_train.pkl", "rb"))[:n_train]; va = pickle.load(open("datos/cand_val.pkl", "rb"))
    X = np.concatenate([r["X"] for r in tr if len(r["X"])]); y = np.concatenate([etiquetar(r) for r in tr if len(r["X"])]); k = y >= 0
    print(f"entrenamiento: {k.sum()} candidatas ({(y[k]==1).sum()} caras, {(y[k]==0).sum()} falsas) de {len(tr)} imágenes", flush=True)
    mu, sd = X[k].mean(0), X[k].std(0) + 1e-6; svm = LinearSVC(C=0.05, class_weight="balanced", max_iter=5000).fit((X[k] - mu) / sd, y[k])
    punt = [svm.decision_function((r["X"] - mu) / sd) if len(r["X"]) else np.array([]) for r in va]
    cota = [np.zeros(len(r["cands"])) for r in va]   # sin filtro: todas las candidatas (techo de recall)
    tp, fp, fn = medir(va, [np.ones(len(r["cands"])) for r in va], 0.5); print(f"SIN segunda etapa (todas las candidatas): recall {tp/max(1,tp+fn):.3f}, precisión {tp/max(1,tp+fp):.3f}, {tp+fn} caras", flush=True)
    for fuente in FUENTES:
        sc = [np.array([1.0 if c[4] == fuente else -1.0 for c in r["cands"]]) for r in va]; tp, fp, fn = medir(va, sc, 0.5)
        print(f"solo {fuente:<10} (permisivo): recall {tp/max(1,tp+fn):.3f}, precisión {tp/max(1,tp+fp):.3f}", flush=True)
    sal = []
    for t in (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5):
        tp, fp, fn = medir(va, punt, t); sal.append({"umbral": t, "recall": round(tp / max(1, tp + fn), 3), "precision": round(tp / max(1, tp + fp), 3), "tp": tp, "fp": fp, "fn": fn})
        print("con segunda etapa", sal[-1], flush=True)
    pickle.dump({"svm": svm, "mu": mu, "sd": sd}, open("modelos/verificador.pkl", "wb")); json.dump(sal, open("borradores/e11-curva.json", "w"))

if __name__ == "__main__":
    c = sys.argv[1]
    from rutas import exigir; exigir("datos/division_wider.csv", "Primero: python src/dividir_wider.py (ver README.md)."); exigir("entradas/wider-yolo/images", "Descarga WIDER: ver README.md, sección «Datos».")
    if c == "extraer": extraer(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 1)
    elif c == "evaluar": evaluar(int(sys.argv[2]) if len(sys.argv) > 2 else 10**9)
