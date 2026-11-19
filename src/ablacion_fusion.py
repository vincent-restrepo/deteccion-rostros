"""Ablación del detector combinado (E15): ¿qué aporta cada fuente de candidatas? Guarda UNA vez las candidatas de cada imagen (datos/cache_fusion.pkl, no va a git)
y luego, para cada subconjunto de fuentes, reagrupa, reentrena el segundo paso (árboles con parada temprana) con las mismas imágenes de entrenamiento
y mide en validación (cara principal, IoU >= 0,5, tope de 3 cajas). Tiempo por fuente medido aparte.
Uso:  .venv/bin/python src/ablacion_fusion.py cache [imgs_entrenamiento_por_dataset=1000] [val_por_dataset=5000]
      .venv/bin/python src/ablacion_fusion.py medir [fuente1+fuente2 …]   (sin argumentos: ablación completa)"""
import sys, os, random, time, pickle, json
import numpy as np, cv2
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hog_svm_cara as hs, detector_fusion as df
from conjunto_principal import cargar
CACHE = os.environ.get("CACHE_FUSION", "datos/cache_fusion.pkl"); AUSENTE = df.AUSENTE; FUENTES = df.FUENTES

def _cands(m):
    g, cajas, _ = hs.cargar_img(m)
    if g is None: return None
    return {"ruta": m["ruta"], "dataset": m["dataset"], "cajas": cajas, "cands": df.candidatas(g)}

def _nuevas(d):
    g, _, _ = hs.cargar_img({"ruta": d["ruta"], "cajas": []})
    return df.candidatas(g, set(df.NUEVAS)) if g is not None else []

def _grupos(args):
    d, usar = args[0], args[1]; tabla = args[2] if len(args) > 2 else None; g, _, _ = hs.cargar_img({"ruta": d["ruta"], "cajas": []})
    if g is None: return None
    c = [x for x in d["cands"] if x[1] in usar]
    if tabla: c = df.corregir(c, tabla)
    return df.agrupar(c, g, "extra" in usar, df.ORDEN) if c else []   # «extra» en la variante = rasgos nuevos (E16)

def _tiempos(m):
    g, _, _ = hs.cargar_img(m); r = df._recursos(); t = {}
    def cron(n, f):
        t0 = time.time(); f(); t[n] = time.time() - t0
    cron("dlib", lambda: df._dlib(g)); cron("dlib_clahe", lambda: df._dlib(r["clahe"].apply(g))); cron("dlib_espejo", lambda: df._dlib(cv2.flip(g, 1)))
    H, W = g.shape; M = cv2.getRotationMatrix2D((W / 2, H / 2), 15, 1.0); cron("dlib_rot", lambda: df._dlib(cv2.warpAffine(g, M, (W, H))))
    cron("haar", lambda: r["haar"].detectMultiScale(g, 1.1, 3, minSize=(40, 40))); cron("hogsvm", lambda: hs.predecir(g, r["w"], r["b"], -0.5)); return t

if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else ""
    if modo == "cache":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000; nv = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
        _, _, resto = hs.partir_train(); rng = random.Random(1); ent = []
        for d in ("wider", "openimages"):
            xs = [m for m in resto if m["dataset"] == d]; rng.shuffle(xs); ent += xs[:n]
        val = cargar("val") if nv > 0 else []; t0 = time.time()   # val_por_dataset = 0: solo entrenamiento (para entrenar los modelos; la ablación necesita validación)
        with ProcessPoolExecutor() as ex:
            e = [x for x in ex.map(_cands, ent, chunksize=4) if x]; print(f"entrenamiento: {len(e)} ({time.time() - t0:.0f} s)", flush=True)
            v = [x for x in ex.map(_cands, val, chunksize=4) if x] if val else []; print(f"validación: {len(v)} ({time.time() - t0:.0f} s)", flush=True)
            muestra = random.Random(3).sample(ent, 40); ts = [t for t in ex.map(_tiempos, muestra)]
        tiempos = {k: round(float(np.mean([t[k] for t in ts[5:]])), 3) for k in ts[0]}   # se descartan las 5 primeras (carga de modelos)
        pickle.dump({"entrenamiento": e, "validacion": v, "tiempos": tiempos}, open(CACHE, "wb")); print("segundos por fuente:", tiempos)
    elif modo == "medir":
        from sklearn.ensemble import HistGradientBoostingClassifier
        z = pickle.load(open(CACHE, "rb")); E, V = z["entrenamiento"], z["validacion"]; print("segundos por fuente:", z["tiempos"], flush=True)
        if os.environ.get("CACHE_ENTRENAMIENTO"): E = pickle.load(open(os.environ["CACHE_ENTRENAMIENTO"], "rb"))["entrenamiento"]; print("entrenamiento desde", os.environ["CACHE_ENTRENAMIENTO"], len(E), "imágenes", flush=True)   # otro caché para entrenar (E16)
        todas = set(FUENTES); variantes = {"todas": todas}
        for q in FUENTES: variantes[f"sin {q}"] = todas - {q}
        variantes["sin rotaciones"] = todas - {"dlib_rot_mas", "dlib_rot_menos"}; variantes["solo dlib normal + hogsvm"] = {"dlib", "hogsvm"}; variantes["solo dlib normal"] = {"dlib"}
        variantes["dlib normal + clahe + espejo"] = {"dlib", "dlib_clahe", "dlib_espejo"}
        if len(sys.argv) > 2: variantes = {a: set(a.split("+")) for a in sys.argv[2:]}   # variantes a medida: fuentes separadas por «+»
        res = {}
        with ProcessPoolExecutor() as ex:
            for nombre, usar in variantes.items():
                tb = df.cargar_correccion() if "corr" in usar else None
                ge = list(ex.map(_grupos, [(d, usar, tb) for d in E], chunksize=8)); gv = list(ex.map(_grupos, [(d, usar, tb) for d in V], chunksize=8))
                X = []; y = []
                for d, gr in zip(E, ge):
                    for caja, f in gr or []: X.append(f); y.append(int(max((hs.iou(caja, c) for c in d["cajas"]), default=0) >= 0.5))
                clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, random_state=1).fit(np.array(X), np.array(y))
                fila = {}
                for umbral in (0.2, 0.1):
                    r = {"wider": [0, 0, 0], "openimages": [0, 0, 0]}   # principal acertada, fp, imágenes
                    for d, gr in zip(V, gv):
                        cajas = d["cajas"]; pr = max(cajas, key=lambda c: c[3]); pred = []
                        if gr:
                            p = clf.predict_proba(np.array([f for _, f in gr]))[:, 1]
                            pred = [c for pp, c in sorted(zip(p, [c for c, _ in gr]), key=lambda t: -t[0]) if pp >= umbral][:df.MAX_CARAS]
                        usado = set(); fp = 0
                        for c in pred:
                            mejor = max(((hs.iou(c, g), k) for k, g in enumerate(cajas) if k not in usado), default=(0, None))
                            if mejor[0] >= 0.5: usado.add(mejor[1]); continue
                            if any(hs.iou(c, g) >= 0.3 for g in cajas if g[3] < 48 * 0.8): continue
                            fp += 1
                        r[d["dataset"]][0] += int(cajas.index(pr) in usado); r[d["dataset"]][1] += fp; r[d["dataset"]][2] += 1
                    fila[str(umbral)] = {k: {"recall_principal": round(a / n, 3), "fp_por_imagen": round(f / n, 3)} for k, (a, f, n) in r.items()}
                res[nombre] = fila; print(nombre, json.dumps(fila, ensure_ascii=False), flush=True)
    elif modo == "agregar":   # ablacion_fusion.py agregar [caché de entrada.pkl ...]: calcula SOLO las fuentes nuevas (df.NUEVAS) y las suma a las candidatas guardadas
        for ruta in (sys.argv[2:] or [CACHE]):
            z = pickle.load(open(ruta, "rb")); t0 = time.time()
            for clave in ("entrenamiento", "validacion"):
                lista = z[clave]
                if not lista: continue
                with ProcessPoolExecutor() as ex: nuevas = list(ex.map(_nuevas, lista, chunksize=4))
                for d, n in zip(lista, nuevas): d["cands"] = [c for c in d["cands"] if c[1] not in df.NUEVAS] + (n or [])
                print(ruta, clave, len(lista), f"({time.time() - t0:.0f} s)", flush=True)
            pickle.dump(z, open(ruta, "wb"))
    elif modo == "correccion":   # ablacion_fusion.py correccion : mide en ENTRENAMIENTO el sesgo de caja de cada fuente y lo guarda en datos/correccion_cajas.json
        E = pickle.load(open(os.environ.get("CACHE_ENTRENAMIENTO", CACHE), "rb"))["entrenamiento"]; est = {}
        for d in E:
            pr = max(d["cajas"], key=lambda c: c[3])
            for c, f, s in d["cands"]:
                if hs.iou(c, pr) >= 0.4: est.setdefault(f, []).append(((c[0] + c[2] / 2 - (pr[0] + pr[2] / 2)) / pr[3], (c[1] + c[3] / 2 - (pr[1] + pr[3] / 2)) / pr[3], c[2] / pr[2], c[3] / pr[3]))
        tabla = {f: [round(float(x), 4) for x in np.mean(v, axis=0)] for f, v in est.items()}
        json.dump(tabla, open(df.RUTA_CORRECCION, "w"), indent=1); print("escrito", df.RUTA_CORRECCION, {f: (v, len(est[f])) for f, v in tabla.items()})
    elif modo == "modelo":   # ablacion_fusion.py modelo fuente1+fuente2 modelos/salida.pkl : entrena con el caché y guarda el segundo paso con su lista de fuentes
        import joblib; from sklearn.ensemble import HistGradientBoostingClassifier
        usar = set(sys.argv[2].split("+")); E = pickle.load(open(CACHE, "rb"))["entrenamiento"]
        tb = df.cargar_correccion() if "corr" in usar else None
        with ProcessPoolExecutor() as ex: ge = list(ex.map(_grupos, [(d, usar, tb) for d in E], chunksize=8))
        X = []; y = []
        for d, gr in zip(E, ge):
            for caja, f in gr or []: X.append(f); y.append(int(max((hs.iou(caja, c) for c in d["cajas"]), default=0) >= 0.5))
        clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, random_state=1).fit(np.array(X), np.array(y))
        joblib.dump({"clf": clf, "fuentes": sorted(usar - {"extra", "corr"}), "extra": "extra" in usar, "orden_fuentes": list(df.ORDEN), "correccion": tb}, sys.argv[3]); print(f"guardado {sys.argv[3]} con {sorted(usar)}; árboles {clf.n_iter_}; grupos {len(y)}")
    else: sys.exit(__doc__)
