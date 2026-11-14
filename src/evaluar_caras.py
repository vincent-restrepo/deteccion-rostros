"""Evalúa detectores de caras clásicos sobre WIDER (formato YOLO: images/*.jpg + labels/*.txt) con IoU>=0,5.
Reporta recall y precisión en total y por tamaño de cara (alto en píxeles de la imagen original) y por número de caras.
Uso: python src/evaluar_caras.py [N] [semilla] [detectores separados por coma: haar_0grados,giros_y_perfil,dlib_hog_x0,dlib_hog_x1,dlib_hog_x2]"""
import sys, os, glob, random, json, time
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector_caras import Detector
RAIZ = "entradas/wider-yolo"
MIN_ALCANCE = 48   # D1: las caras de menos de 48 px de alto quedan fuera de alcance: ni suman recall ni penalizan precisión
ANCHO = 800
haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

def etiquetas(ruta_txt, W, H):
    out = []
    for l in open(ruta_txt).read().split("\n"):
        v = l.split()
        if len(v) >= 5:
            cx, cy, w, h = (float(x) for x in v[1:5]); out.append((int((cx - w / 2) * W), int((cy - h / 2) * H), int(w * W), int(h * H)))
    return out

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    i = max(0, x2-x1) * max(0, y2-y1); u = a[2]*a[3] + b[2]*b[3] - i
    return i / u if u else 0

def haar_solo(g): return [tuple(int(v) for v in c) for c in haar.detectMultiScale(g, 1.1, 3, minSize=(24, 24))]
def nuevo(g): return Detector()(g)

def fusion(g):
    """Unión de dlib HOG (x1) y Haar con giros/perfiles; se queda con la caja más grande cuando se solapan."""
    from detector_caras import _nms
    a = dlib_hog(1)(g); b = Detector()(g)
    return [c for c, _ in _nms([(c, 0) for c in a + b])]

def dlib_hog(sube):
    import dlib
    det = dlib.get_frontal_face_detector()
    def f(g):
        return [(max(0, r.left()), max(0, r.top()), r.width(), r.height()) for r in det(g, sube)]
    return f

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120; sem = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    from rutas import exigir; exigir(RAIZ, "Descarga WIDER: ver README.md, sección «Datos».")
    imgs = sorted(glob.glob(f"{RAIZ}/images/*.jpg")); random.Random(sem).shuffle(imgs)
    bins = [(0, 24), (24, 48), (48, 96), (96, 10**6)]
    res = {}
    elegidos = sys.argv[3].split(",") if len(sys.argv) > 3 else ["haar_0grados", "giros_y_perfil"]
    todos = {"haar_0grados": haar_solo, "giros_y_perfil": nuevo, "dlib_hog_x0": None, "dlib_hog_x1": None, "dlib_hog_x2": None, "fusion": fusion}
    for nombre in elegidos:
        fn = todos[nombre]
        if nombre.startswith("dlib_hog_x"): fn = dlib_hog(int(nombre[-1]))
        tp = fp = fn_ = 0; por = {b: [0, 0] for b in bins}; t0 = time.time(); usadas = 0
        for f in imgs:
            if usadas >= n: break
            im = cv2.imread(f); txt = f.replace("/images/", "/labels/").replace(".jpg", ".txt")
            if im is None or not os.path.exists(txt): continue
            H, W = im.shape[:2]; gt = etiquetas(txt, W, H)
            if not gt: continue
            usadas += 1; esc = min(1.0, ANCHO / W)
            g = cv2.cvtColor(cv2.resize(im, (int(W * esc), int(H * esc))) if esc < 1 else im, cv2.COLOR_BGR2GRAY)
            pred = [tuple(int(v / esc) for v in p) for p in fn(g)]
            usado = set(); dentro = [j for j, g_ in enumerate(gt) if g_[3] >= MIN_ALCANCE]
            for p in pred:
                m = [(iou(p, gt[j]), j) for j in dentro if j not in usado]
                if m and max(m)[0] >= 0.5: usado.add(max(m)[1]); tp += 1; continue
                if any(iou(p, g_) >= 0.3 for j, g_ in enumerate(gt) if j not in dentro): continue   # acierta una cara fuera de alcance: se ignora
                fp += 1
            for j in dentro:
                g_ = gt[j]; b = next(b for b in bins if b[0] <= g_[3] < b[1]); por[b][1] += 1
                if j in usado: por[b][0] += 1
            fn_ += len(dentro) - len(usado)
        rec = tp / max(1, tp + fn_); pre = tp / (tp + fp) if tp + fp else 0
        res[nombre] = {"imagenes": usadas, "tp": tp, "fp": fp, "fn": fn_, "recall": round(rec, 3), "precision": round(pre, 3),
                       "recall_por_alto_px": {f"{a}-{b if b < 10**6 else 'mas'}": [por[(a, b)][0], por[(a, b)][1], round(por[(a, b)][0] / max(1, por[(a, b)][1]), 3)] for a, b in bins},
                       "segundos_por_imagen": round((time.time() - t0) / max(1, usadas), 2)}
        print(nombre, json.dumps(res[nombre], ensure_ascii=False), flush=True)
