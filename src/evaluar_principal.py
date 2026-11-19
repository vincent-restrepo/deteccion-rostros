"""Evalúa detectores clásicos sobre el conjunto «cara principal» (D13): imágenes de WIDER y Open Images con 1 a 3 caras y la mayor >= 96 px.
Reporta, POR DATASET: recall de la cara principal (IoU >= 0,5), recall de las caras secundarias (>= 48 px), falsos positivos por imagen,
precisión y segundos por imagen. Una predicción que acierta una cara etiquetada de < 48 px se ignora (no es falso positivo), igual que en D1.
Es otra prueba que la de D1 (src/evaluar_caras.py): no compares las cifras entre una y otra.
Uso:  .venv/bin/python src/evaluar_principal.py <train|val> <N por dataset> <semilla> [detectores separados por coma]
Detectores: haar_0grados, giros_y_perfil, dlib_x0, dlib_x1, dlib_x2, dlib_x1_t-0.4, dlib_x2_t-0.4, hogsvm_t<umbral> (modelo propio de src/hog_svm_cara.py), fusion_t<umbral> (src/detector_fusion.py).   (la división «test» no se usa aquí: es la prueba final)"""
import sys, os, json, time, random, collections
import cv2, numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conjunto_principal import cargar
ANCHO = int(os.environ.get("ANCHO_EVAL", 800)); MIN_SEC = 48   # ANCHO_EVAL: ancho de trabajo en píxeles (E20)
_det = {}

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    i = max(0, x2-x1) * max(0, y2-y1); u = a[2]*a[3] + b[2]*b[3] - i
    return i / u if u else 0

def detector(nombre):
    if nombre in _det: return _det[nombre]
    if nombre == "haar_0grados":
        c = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
        f = lambda g: [tuple(int(v) for v in r) for r in c.detectMultiScale(g, 1.1, 3, minSize=(24, 24))]
    elif nombre == "giros_y_perfil":
        from detector_caras import Detector
        d = Detector(); f = lambda g: d(g)
    elif nombre.startswith("dlib_x"):
        import dlib
        d = dlib.get_frontal_face_detector(); sube = int(nombre[6]); umbral = float(nombre.split("_t")[1]) if "_t" in nombre else 0.0
        f = lambda g: [(max(0, r.left()), max(0, r.top()), r.width(), r.height()) for r in d.run(g, sube, umbral)[0]]
    elif nombre.startswith("hogsvm_t"):
        from hog_svm_cara import cargar_detector
        f = cargar_detector(float(nombre[8:]))
    elif nombre.startswith("fusion"):
        from detector_fusion import cargar_detector as cf
        base, u = nombre.split("_t"); f = cf(float(u), {"fusion": "modelos/fusion.pkl", "fusionrapido": "modelos/fusion_rapido.pkl", "fusioneq": "modelos/fusion_equilibrado.pkl", "fusioncompleto": "modelos/fusion_completo.pkl"}[base])
    else: raise SystemExit("Detector desconocido: " + nombre)
    _det[nombre] = f; return f

def una(args):
    nombre, m = args; im = cv2.imread(m["ruta"])
    if im is None: return None
    H, W = im.shape[:2]; esc = min(1.0, ANCHO / W)
    g = cv2.cvtColor(cv2.resize(im, (int(W * esc), int(H * esc))) if esc < 1 else im, cv2.COLOR_BGR2GRAY)
    t0 = time.time(); pred = [tuple(int(v / esc) for v in p[:4]) for p in detector(nombre)(g)]; seg = time.time() - t0
    cajas = m["cajas"]; pr = m["principal"]; usado = set(); tp = fp = 0
    sec = [c for c in cajas if c is not pr and c[3] >= MIN_SEC]
    for p in pred:
        mejor = max(((iou(p, c), k) for k, c in enumerate(cajas) if k not in usado), default=(0, None))
        if mejor[0] >= 0.5: usado.add(mejor[1]); tp += 1; continue
        if any(iou(p, c) >= 0.3 for c in cajas if c[3] < MIN_SEC): continue
        fp += 1
    ip = next(k for k, c in enumerate(cajas) if c is pr)
    return {"dataset": m["dataset"], "principal": int(ip in usado), "alto": pr[3], "sec": sum(1 for k, c in enumerate(cajas) if k in usado and c in sec), "n_sec": len(sec), "tp": tp, "fp": fp, "seg": seg}

if __name__ == "__main__":
    div, n, sem = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); nombres = sys.argv[4].split(",") if len(sys.argv) > 4 else ["haar_0grados", "giros_y_perfil", "dlib_x1", "dlib_x1_t-0.4"]
    if div == "test": sys.exit("«test» es la prueba final: no se usa en esta evaluación.")
    todo = cargar(div); elegidos = []
    for d in ("wider", "openimages"):
        xs = [m for m in todo if m["dataset"] == d]; random.Random(sem).shuffle(xs); elegidos += xs[:n]
    with ProcessPoolExecutor() as ex:
        for nombre in nombres:
            t0 = time.time(); rs = [r for r in ex.map(una, [(nombre, m) for m in elegidos], chunksize=8) if r]; salida = {}
            for d in ("wider", "openimages", "ambos"):
                x = [r for r in rs if d == "ambos" or r["dataset"] == d]
                if not x: continue
                tp = sum(r["tp"] for r in x); fp = sum(r["fp"] for r in x); ns = sum(r["n_sec"] for r in x)
                salida[d] = {"imagenes": len(x), "recall_principal": round(sum(r["principal"] for r in x) / len(x), 3),
                             "recall_secundarias": round(sum(r["sec"] for r in x) / ns, 3) if ns else None,
                             "fp_por_imagen": round(fp / len(x), 3), "precision": round(tp / (tp + fp), 3) if tp + fp else 0,
                             "seg_por_imagen_deteccion": round(float(np.mean([r["seg"] for r in x])), 3)}
            print(nombre, json.dumps(salida, ensure_ascii=False), f"({time.time() - t0:.0f} s)", flush=True)
