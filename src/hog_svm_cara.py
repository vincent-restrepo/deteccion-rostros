"""Detector de cara propio: HOG + SVM lineal, con ventana deslizante en pirámide y «hard negative mining» (técnica clásica, sin redes neuronales).
Entrena con el conjunto «cara principal» (D13) de las DOS divisiones de entrenamiento (WIDER + Open Images). No toca val ni test.
Pasos: (1) positivos = caras >= 48 px (con 12,5 % de margen) y su espejo; negativos = ventanas al azar sin caras; (2) SVM; (3) N rondas: se corre el
detector sobre imágenes de entrenamiento y los falsos positivos se agregan como negativos; (4) se guarda modelos/hogsvm_cara.npz (w, b).
Uso:  .venv/bin/python src/hog_svm_cara.py entrenar [imgs_positivos=3000] [imgs_minado=1500] [rondas=2] [semilla=1]
Como módulo:  from hog_svm_cara import cargar_detector; det = cargar_detector(umbral); cajas = det(gris)  # cajas (x, y, ancho, alto)"""
import sys, os, random, time
import numpy as np, cv2
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RUTA_MODELO = "modelos/hogsvm_cara.npz"; VENT = 64; MARGEN = 0.125; ANCHO = 800; MIN_CARA = 48
HOG = lambda: cv2.HOGDescriptor((VENT, VENT), (16, 16), (8, 8), (8, 8), 9)

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    i = max(0, x2-x1) * max(0, y2-y1); u = a[2]*a[3] + b[2]*b[3] - i
    return i / u if u else 0

def ventana_de_cara(c):   # caja de la cara -> ventana cuadrada con margen
    s = max(c[2], c[3]) * (1 + 2 * MARGEN); cx, cy = c[0] + c[2] / 2, c[1] + c[3] / 2
    return (cx - s / 2, cy - s / 2, s, s)

def cara_de_ventana(v):   # inverso
    s = v[2] / (1 + 2 * MARGEN); cx, cy = v[0] + v[2] / 2, v[1] + v[3] / 2
    return (int(cx - s / 2), int(cy - s / 2), int(s), int(s))

def recorte(g, v):
    x, y, s = int(round(v[0])), int(round(v[1])), int(round(v[2])); H, W = g.shape
    if s < 16 or x < 0 or y < 0 or x + s > W or y + s > H: return None
    return cv2.resize(g[y:y + s, x:x + s], (VENT, VENT), interpolation=cv2.INTER_AREA)

def cargar_img(m):
    im = cv2.imread(m["ruta"], cv2.IMREAD_GRAYSCALE)
    if im is None: return None, None, 1.0
    H, W = im.shape; esc = min(1.0, ANCHO / W)
    g = cv2.resize(im, (int(W * esc), int(H * esc)), interpolation=cv2.INTER_AREA) if esc < 1 else im
    return g, [tuple(v * esc for v in c) for c in m["cajas"]], esc

def _positivos_negativos(args):
    m, rng_sem, n_neg = args; g, cajas, _ = cargar_img(m); rng = random.Random(rng_sem)
    if g is None: return [], []
    hog = HOG(); pos = []; neg = []
    for c in cajas:
        if c[3] < MIN_CARA * 0.8: continue
        r = recorte(g, ventana_de_cara(c))
        if r is not None: pos.append(hog.compute(r).ravel()); pos.append(hog.compute(r[:, ::-1].copy()).ravel())
    H, W = g.shape; vent_cajas = [ventana_de_cara(c) for c in cajas]; intentos = 0
    while len(neg) < n_neg and intentos < 60:
        intentos += 1; s = rng.uniform(48, max(49, min(H, W) / 1.2)); v = (rng.uniform(0, max(1, W - s)), rng.uniform(0, max(1, H - s)), s, s)
        if any(iou(v, w) > 0.1 for w in vent_cajas): continue
        r = recorte(g, v)
        if r is not None: neg.append(hog.compute(r).ravel())
    return pos, neg

def predecir(g, w, b, umbral=0.0, escala=1.05):
    hog = HOG(); hog.setSVMDetector(np.append(w, b).astype(np.float32))
    rects, pesos = hog.detectMultiScale(g, hitThreshold=umbral, winStride=(8, 8), padding=(0, 0), scale=escala)
    items = sorted(zip([tuple(int(v) for v in r) for r in rects], [float(p) for p in np.ravel(pesos)]), key=lambda t: -t[1]); out = []
    for r, p in items:
        if all(iou(r, o[0]) < 0.3 for o in out): out.append((r, p))
    return out   # [(ventana, puntaje)]

def _falsos_positivos(args):
    m, w, b, maximo = args; g, cajas, _ = cargar_img(m)
    if g is None: return []
    hog = HOG(); vent = [ventana_de_cara(c) for c in cajas]; res = []
    for r, p in predecir(g, w, b, 0.0):
        if any(iou(r, v) > 0.2 for v in vent): continue
        x = recorte(g, r)
        if x is not None: res.append(hog.compute(x).ravel())
        if len(res) >= maximo: break
    return res

def partir_train(n_pos=3000, n_min=1500, sem=1):
    """Reparte las imágenes de entrenamiento (conjunto cara principal) en: positivos/negativos, minado y «resto» (no vistas por este modelo)."""
    from conjunto_principal import cargar
    todo = cargar("train"); rng = random.Random(sem); por = {d: [m for m in todo if m["dataset"] == d] for d in ("wider", "openimages")}
    pos_img, min_img, resto = [], [], []
    for d, xs in por.items():
        rng.shuffle(xs); pos_img += xs[:n_pos // 2]; min_img += xs[n_pos // 2: n_pos // 2 + n_min // 2]; resto += xs[n_pos // 2 + n_min // 2:]
    return pos_img, min_img, resto

def cargar_detector(umbral=0.0, ruta=RUTA_MODELO):
    z = np.load(ruta); w, b = z["w"], float(z["b"])
    def det(g): return [cara_de_ventana(r) for r, _ in predecir(g, w, b, umbral)]
    return det

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "entrenar": sys.exit(__doc__)
    from sklearn.svm import LinearSVC
    n_pos = int(sys.argv[2]) if len(sys.argv) > 2 else 3000; n_min = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
    rondas = int(sys.argv[4]) if len(sys.argv) > 4 else 2; sem = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    pos_img, min_img, _ = partir_train(n_pos, n_min, sem)
    print(f"imágenes para positivos/negativos: {len(pos_img)}; para minar: {len(min_img)}", flush=True); t0 = time.time()
    with ProcessPoolExecutor() as ex:
        r = list(ex.map(_positivos_negativos, [(m, sem * 100003 + i, 8) for i, m in enumerate(pos_img)], chunksize=16))
        P = [x for p, _ in r for x in p]; N = [x for _, n in r for x in n]
        print(f"positivos {len(P)}, negativos iniciales {len(N)} ({time.time() - t0:.0f} s)", flush=True)
        for ronda in range(rondas + 1):
            X = np.array(P + N, dtype=np.float32); y = np.array([1] * len(P) + [0] * len(N))
            clf = LinearSVC(C=0.01, max_iter=5000, random_state=sem).fit(X, y); w, b = clf.coef_.ravel(), float(clf.intercept_[0])
            print(f"ronda {ronda}: {len(P)} positivos, {len(N)} negativos; exactitud de entrenamiento {clf.score(X, y):.3f} ({time.time() - t0:.0f} s)", flush=True)
            if ronda == rondas: break
            fps = [x for lst in ex.map(_falsos_positivos, [(m, w, b, 25) for m in min_img], chunksize=8) for x in lst]
            print(f"  falsos positivos minados: {len(fps)}", flush=True); N += fps
    os.makedirs("modelos", exist_ok=True); np.savez(RUTA_MODELO, w=w, b=b); print("guardado", RUTA_MODELO)
