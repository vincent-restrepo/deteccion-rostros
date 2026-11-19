"""Detector combinado de cara principal (solo técnicas clásicas): varias fuentes de candidatas -> agrupar -> un segundo paso (árboles con boosting,
parada temprana) que decide cuáles son caras -> tope de MAX_CARAS cajas por imagen (D13).
Fuentes: dlib HOG (umbral bajo) sobre la imagen normal, con contraste igualado (CLAHE), espejada y rotada ±15°; Haar frontal; y el HOG+SVM propio.
El segundo paso se entrena con imágenes de entrenamiento que el HOG+SVM no vio (partir_train(...)[2]); val y test no se tocan.
Uso:  .venv/bin/python src/detector_fusion.py entrenar [imgs_por_dataset=1000] [semilla=1]
Como módulo:  from detector_fusion import cargar_detector; det = cargar_detector(umbral=0.5); cajas = det(gris)  # (x, y, ancho, alto)"""
import sys, os, random, time
import numpy as np, cv2
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hog_svm_cara as hs
RUTA = "modelos/fusion.pkl"; MAX_CARAS = 3; UMBRAL_DLIB = -1.0; AUSENTE = -3.0
FUENTES = ("dlib", "dlib_clahe", "dlib_espejo", "dlib_rot_mas", "dlib_rot_menos", "haar", "hogsvm")   # las 7 de E14/E15 (modelos viejos)
NUEVAS = ("haar_perfil", "dlib_pad", "dlib_rot_90", "dlib_rot_270")   # E17: perfiles, caras cortadas por el borde, cabezas giradas 90°
ORDEN = FUENTES + NUEVAS
_c = {}

def _recursos():
    if not _c:
        import dlib
        _c["dlib"] = dlib.get_frontal_face_detector(); _c["haar"] = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
        _c["clahe"] = cv2.createCLAHE(2.0, (8, 8)); _c["perfil"] = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml"); z = np.load(hs.RUTA_MODELO); _c["w"], _c["b"] = z["w"], float(z["b"])
    return _c

def _dlib(g):
    r, s, idx = _recursos()["dlib"].run(g, 1, UMBRAL_DLIB)
    return [((max(0, a.left()), max(0, a.top()), a.width(), a.height()), float(sc)) for a, sc in zip(r, s)]

def candidatas(g, usar=None):
    """Lista de (caja, fuente, puntaje) en coordenadas de g. `usar`: conjunto de fuentes a calcular (None = todas); las demás no se calculan."""
    c = _recursos(); H, W = g.shape; ok = lambda n: usar is None or n in usar; out = []
    if ok("dlib"): out += [(b, "dlib", s) for b, s in _dlib(g)]
    if ok("dlib_clahe"): out += [(b, "dlib_clahe", s) for b, s in _dlib(c["clahe"].apply(g))]
    if ok("dlib_espejo"): out += [((W - b[0] - b[2], b[1], b[2], b[3]), "dlib_espejo", s) for b, s in _dlib(cv2.flip(g, 1))]
    for ang, nombre in ((15, "dlib_rot_mas"), (-15, "dlib_rot_menos")):
        if not ok(nombre): continue
        M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0); inv = cv2.invertAffineTransform(M)
        for b, s in _dlib(cv2.warpAffine(g, M, (W, H))):
            cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2; ox, oy = inv @ np.array([cx, cy, 1.0])
            out.append(((int(ox - b[2] / 2), int(oy - b[3] / 2), b[2], b[3]), nombre, s))
    if ok("haar_perfil"):
        for r in c["perfil"].detectMultiScale(g, 1.1, 3, minSize=(40, 40)): out.append((tuple(int(v) for v in r), "haar_perfil", 1.0))
        for r in c["perfil"].detectMultiScale(cv2.flip(g, 1), 1.1, 3, minSize=(40, 40)): out.append(((W - int(r[0]) - int(r[2]), int(r[1]), int(r[2]), int(r[3])), "haar_perfil", 1.0))
    if ok("dlib_pad"):   # el borde se refleja: una cara cortada por el borde queda «completa» al otro lado del espejo
        p_ = int(0.2 * max(H, W))
        for b, s in _dlib(cv2.copyMakeBorder(g, p_, p_, p_, p_, cv2.BORDER_REFLECT_101)): out.append(((b[0] - p_, b[1] - p_, b[2], b[3]), "dlib_pad", s))
    if ok("dlib_rot_90"):
        for b, s in _dlib(cv2.rotate(g, cv2.ROTATE_90_CLOCKWISE)): out.append(((b[1], H - 1 - (b[0] + b[2]), b[3], b[2]), "dlib_rot_90", s))
    if ok("dlib_rot_270"):
        for b, s in _dlib(cv2.rotate(g, cv2.ROTATE_90_COUNTERCLOCKWISE)): out.append(((W - 1 - (b[1] + b[3]), b[0], b[3], b[2]), "dlib_rot_270", s))
    if ok("haar"):
        for r in c["haar"].detectMultiScale(g, 1.1, 3, minSize=(40, 40)): out.append((tuple(int(v) for v in r), "haar", 1.0))
    if ok("hogsvm"):
        for v, p in hs.predecir(g, c["w"], c["b"], -0.5)[:15]: out.append((hs.cara_de_ventana(v), "hogsvm", p))
    return out

def agrupar(cands, g, extra=False, fuentes=FUENTES):
    """Une candidatas solapadas (IoU > 0,4) y devuelve (caja, vector de rasgos) por grupo."""
    c = _recursos(); H, W = g.shape; grupos = []
    for caja, fuente, s in sorted(cands, key=lambda t: -(t[2] if t[1] != "haar" else 0.5)):
        for gr in grupos:
            if hs.iou(caja, gr["caja"]) > 0.4:
                gr["miembros"].append((caja, fuente, s)); gr["caja"] = tuple(int(np.mean([m[0][k] for m in gr["miembros"]])) for k in range(4)); break
        else: grupos.append({"caja": caja, "miembros": [(caja, fuente, s)]})
    hog = hs.HOG(); out = []
    for gr in grupos:
        x, y, w, h = gr["caja"]; mejor = {f: AUSENTE for f in fuentes}
        for _, f, s in gr["miembros"]: mejor[f] = max(mejor[f], s)
        r = hs.recorte(g, hs.ventana_de_cara(gr["caja"])); re = float(c["w"] @ hog.compute(r).ravel() + c["b"]) if r is not None else AUSENTE
        n = sum(1 for f in fuentes if mejor[f] > AUSENTE)
        out.append((gr["caja"], [mejor[f] for f in fuentes] + [n, re, h / H, (x + w / 2) / W, (y + h / 2) / H, w / max(1, h), np.log(max(h, 1))]))
    if extra and out: out = _rasgos_extra(out, g, len(fuentes))
    return out

def _rasgos_extra(grupos, g, n_f):
    """Rasgos nuevos (E16): nitidez y contraste dentro de la caja, y contexto: lugar de cada grupo por tamaño y por puntajes frente a los demás de la imagen."""
    H, W = g.shape; dl = lambda f: max(f[0], f[1], f[2], f[3], f[4])   # mejor puntaje de dlib entre sus variantes
    res = [f[n_f + 1] for _, f in grupos]; dls = [dl(f) for _, f in grupos]; hs_ = [c[3] for c, _ in grupos]; sal = []
    orden = lambda v, x: int(sum(1 for o in v if o > x))   # cuántos grupos tienen un valor mayor (0 = el mejor)
    for (c, f), r_, d_, h_ in zip(grupos, res, dls, hs_):
        x, y, w, h = c; rec = g[max(0, y):max(0, y) + max(1, h), max(0, x):max(0, x) + max(1, w)]
        if rec.size < 16: nit = con = med = gra = 0.0
        else:
            rec = cv2.resize(rec, (32, 32), interpolation=cv2.INTER_AREA); nit = float(cv2.Laplacian(rec, cv2.CV_64F).var()); con = float(rec.std()); med = float(rec.mean())
            gra = float(np.mean(np.hypot(cv2.Sobel(rec, cv2.CV_32F, 1, 0), cv2.Sobel(rec, cv2.CV_32F, 0, 1))))
        sal.append((c, f + [nit, con, med, gra, len(grupos), orden(res, r_), orden(dls, d_), orden(hs_, h_), h_ / max(hs_), r_ - max(res), d_ - max(dls)]))
    return sal

def _una(args):
    m = args; g, cajas, _ = hs.cargar_img(m)
    if g is None: return []
    return [(f, int(max((hs.iou(caja, c) for c in cajas), default=0) >= 0.5)) for caja, f in agrupar(candidatas(g), g)]

def cargar_detector(umbral=0.5, ruta=RUTA, max_caras=MAX_CARAS):
    import joblib; obj = joblib.load(ruta)
    clf, usar = (obj["clf"], set(obj["fuentes"])) if isinstance(obj, dict) else (obj, set(FUENTES))   # modelos viejos (E14): solo sus 7 fuentes
    extra = bool(obj.get("extra", False)) if isinstance(obj, dict) else False; orden = tuple(obj.get("orden_fuentes", FUENTES)) if isinstance(obj, dict) else FUENTES
    def det(g):
        gr = agrupar(candidatas(g, usar), g, extra, orden)
        if not gr: return []
        p = clf.predict_proba(np.array([f for _, f in gr]))[:, 1]
        ranking = sorted(zip(p, [c for c, _ in gr]), key=lambda t: -t[0])
        return [c for pp, c in ranking if pp >= umbral][:max_caras]
    return det

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "entrenar": sys.exit(__doc__)
    import joblib; from sklearn.ensemble import HistGradientBoostingClassifier
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000; sem = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    _, _, resto = hs.partir_train(); rng = random.Random(sem); elegidas = []
    for d in ("wider", "openimages"):
        xs = [m for m in resto if m["dataset"] == d]; rng.shuffle(xs); elegidas += xs[:n]
    print(f"imágenes del segundo paso: {len(elegidas)}", flush=True); t0 = time.time()
    with ProcessPoolExecutor() as ex: filas = [f for lst in ex.map(_una, elegidas, chunksize=4) for f in lst]
    X = np.array([f for f, _ in filas]); y = np.array([l for _, l in filas])
    print(f"grupos: {len(y)}, de los cuales caras (IoU>=0,5): {int(y.sum())} ({time.time() - t0:.0f} s)", flush=True)
    clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, random_state=sem).fit(X, y)
    print(f"árboles usados con parada temprana: {clf.n_iter_} de 500; exactitud de entrenamiento {clf.score(X, y):.3f}")
    os.makedirs("modelos", exist_ok=True); joblib.dump(clf, RUTA); print("guardado", RUTA)
