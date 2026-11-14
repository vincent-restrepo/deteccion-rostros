"""Revisor (parte B): dado una imagen, dice DÓNDE hay caras alteradas y DE QUÉ TIPO (desenfoque, pixelado, tapado). Solo visión clásica.
Etapa 1: parches 32×32 (paso 16) → «alterado / natural» con un bosque aleatorio sobre rasgos de nitidez, color plano y cuadrícula de bloques.
Etapa 2: cada caja propuesta → tipo, con otro bosque sobre rasgos de la región entera.
Uso:  .venv/bin/python src/revisor.py entrenar [N_img]   |   evaluar <val|test> [N_img]   |   ver <imagen.jpg>"""
import sys, os, json, pickle, time, collections
import cv2, numpy as np
from multiprocessing import Pool

V = 32; PASO = 16
BLOQUES = (4, 6, 8, 12, 16, 24, 32, 48, 64)
TIPOS = ("desenfoque", "pixelado", "tapado")
MODELO = "modelos/revisor.pkl"

def _razon_bloque(dx, dy, b, caja_filtro):
    """Cuánto menor es la diferencia entre vecinos DENTRO de los bloques que EN sus bordes (cuadrícula alineada al (0,0)); ≈ 0 si hay pixelado de lado b."""
    H, W = dx.shape; cx = np.arange(W); cy = np.arange(H)
    bx = (((cx + 1) % b) == 0).astype(np.float32)[None, :]; by = (((cy + 1) % b) == 0).astype(np.float32)[:, None]
    out = []
    for d, m in ((dx, bx), (dy, by)):
        ent = caja_filtro(d * (1 - m)); mi = caja_filtro(np.broadcast_to(1 - m, d.shape).copy())
        bor = caja_filtro(d * m); mb = caja_filtro(np.broadcast_to(m, d.shape).copy())
        a = ent / np.maximum(mi, 1e-6); c = bor / np.maximum(mb, 1e-6)
        out.append(np.where(mb > 1e-4, a / (c + 1.0), 1.0))
    return (out[0] + out[1]) / 2

def rasgos_imagen(img):
    """Mapa de rasgos en una rejilla de paso 16. Devuelve (F[Gy,Gx,n], centros_x, centros_y)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32); H, W = g.shape
    caja = lambda a: cv2.boxFilter(a, -1, (V, V), normalize=True, borderType=cv2.BORDER_REFLECT)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=1)
    dx = np.zeros_like(g); dx[:, :-1] = np.abs(g[:, 1:] - g[:, :-1]); dy = np.zeros_like(g); dy[:-1, :] = np.abs(g[1:, :] - g[:-1, :])
    m = caja(g); v = np.maximum(caja(g * g) - m * m, 0)
    plano = ((dx < 1.5) & (dy < 1.5)).astype(np.float32)
    mapas = [np.log1p(caja(lap * lap)), np.log1p(caja(dx + dy)), caja(plano), np.log1p(np.sqrt(v))]
    razones = [_razon_bloque(dx, dy, b, caja) for b in BLOQUES]
    mapas += razones; mapas.append(np.min(razones, axis=0))
    ys = np.arange(V // 2, H - V // 2 + 1, PASO); xs = np.arange(V // 2, W - V // 2 + 1, PASO)
    if len(ys) == 0 or len(xs) == 0: return None, xs, ys
    F = np.stack([mp[np.ix_(ys, xs)] for mp in mapas], axis=-1)
    # contexto: cada rasgo básico menos el promedio de los vecinos a 2 pasos (32 px)
    P = np.pad(F[..., :4], ((2, 2), (2, 2), (0, 0)), mode="edge"); Gy, Gx = F.shape[:2]
    anillo = sum(P[2 + dy_:2 + dy_ + Gy, 2 + dx_:2 + dx_ + Gx] for dy_ in (-2, 0, 2) for dx_ in (-2, 0, 2) if (dy_, dx_) != (0, 0)) / 8
    return np.concatenate([F, F[..., :4] - anillo], axis=-1), xs, ys

def etiqueta_rejilla(shape, regiones, xs, ys):
    """1 = ventana completamente dentro de una región alterada; 0 = sin nada alterado; -1 = ambigua (se ignora al entrenar)."""
    H, W = shape[:2]; m = np.zeros((H, W), np.float32)
    for r in regiones: m[r[1]:r[3], r[0]:r[2]] = 1
    fr = cv2.boxFilter(m, -1, (V, V), normalize=True, borderType=cv2.BORDER_REFLECT)[np.ix_(ys, xs)]
    return np.where(fr >= 0.99, 1, np.where(fr <= 0.0, 0, -1))

def rasgos_region(img, caja):
    """Rasgos de una región entera (para decidir el tipo)."""
    x0, y0, x1, y1 = caja; r = img[y0:y1, x0:x1]
    if r.shape[0] < 8 or r.shape[1] < 8: return None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32); gr = g[y0:y1, x0:x1]
    lap = cv2.Laplacian(gr, cv2.CV_32F, ksize=1)
    dx = np.abs(gr[:, 1:] - gr[:, :-1]); dy = np.abs(gr[1:, :] - gr[:-1, :])
    ring = g[max(0, y0 - 24):y1 + 24, max(0, x0 - 24):x1 + 24].copy(); lap_r = cv2.Laplacian(ring, cv2.CV_32F, ksize=1)
    hh, ww = gr.shape; ring_mask = np.ones(ring.shape, bool); oy, ox = y0 - max(0, y0 - 24), x0 - max(0, x0 - 24); ring_mask[oy:oy + hh, ox:ox + ww] = False
    lap_fuera = lap_r[ring_mask].var() if ring_mask.any() else 0
    f = [np.log1p(lap.var()), np.log1p(lap_fuera), np.log1p(lap.var()) - np.log1p(lap_fuera),
         float(((dx[:-1, :] < 1.5) & (dy[:, :-1] < 1.5)).mean()), np.log1p(gr.std()),
         np.log(len({tuple(p) for p in (r.reshape(-1, 3) // 8)}) + 1)]
    rs = []
    for b in BLOQUES:   # cuadrícula alineada al (0,0) de la imagen: columnas/filas absolutas
        cx = (np.arange(x0, x1 - 1) + 1) % b == 0; cy = (np.arange(y0, y1 - 1) + 1) % b == 0
        dxx = dx[:, :]; dyy = dy[:, :]
        ent = (dxx[:, ~cx].mean() if (~cx).any() else 0); bor = (dxx[:, cx].mean() if cx.any() else None)
        eny = (dyy[~cy, :].mean() if (~cy).any() else 0); bory = (dyy[cy, :].mean() if cy.any() else None)
        rx = ent / (bor + 1.0) if bor is not None else 1.0; ry = eny / (bory + 1.0) if bory is not None else 1.0
        rs.append((rx + ry) / 2)
    return np.array(f + rs + [min(rs), float(np.argmin(rs))], np.float32)

def procesar_entrenamiento(args):
    ruta, regiones, n_nat = args
    img = cv2.imread(ruta)
    if img is None: return None
    F, xs, ys = rasgos_imagen(img)
    if F is None: return None
    et = etiqueta_rejilla(img.shape, [r["region"] for r in regiones], xs, ys); rng = np.random.RandomState(abs(hash(ruta)) % 2**31)
    X1 = F[et == 1]; nat = F[et == 0]
    X0 = nat[rng.choice(len(nat), min(len(nat), n_nat), replace=False)] if len(nat) else nat
    reg = [(rasgos_region(img, tuple(r["region"])), TIPOS.index(r["tipo"])) for r in regiones]
    return X1, X0, [(f, t) for f, t in reg if f is not None]

def cargar(div, n):
    L = [json.loads(l) for l in open(f"datos/alteradas/{div}/etiquetas.jsonl")][:n]
    for x in L: x["ruta"] = f"datos/alteradas/{div}/imgs/{x['archivo']}"
    return L

def entrenar(n):
    from sklearn.ensemble import RandomForestClassifier
    L = cargar("train", n); t0 = time.time()
    with Pool(3) as p: res = [r for r in p.map(procesar_entrenamiento, [(x["ruta"], x["regiones"], 120) for x in L], chunksize=20) if r]
    X1 = np.concatenate([r[0] for r in res if len(r[0])]); X0 = np.concatenate([r[1] for r in res if len(r[1])])
    X = np.concatenate([X1, X0]); y = np.r_[np.ones(len(X1)), np.zeros(len(X0))]
    print(f"parches: alterados {len(X1)}, naturales {len(X0)}; {round(time.time()-t0)} s", flush=True)
    m1 = RandomForestClassifier(150, max_depth=18, min_samples_leaf=3, n_jobs=3, class_weight="balanced", random_state=0).fit(X, y)
    Xr = np.array([f for r in res for f, t in r[2]]); yr = np.array([t for r in res for f, t in r[2]])
    m2 = RandomForestClassifier(300, max_depth=12, min_samples_leaf=2, n_jobs=3, class_weight="balanced", random_state=0).fit(Xr, yr)
    print("regiones para el tipo:", collections.Counter(TIPOS[i] for i in yr), flush=True)
    os.makedirs("modelos", exist_ok=True); pickle.dump({"m1": m1, "m2": m2, "n_img": len(res)}, open(MODELO, "wb"))

def revisar(img, mod, umbral=0.5, pad=16, area_min=40 * 40):
    F, xs, ys = rasgos_imagen(img)
    if F is None: return []
    Gy, Gx = F.shape[:2]; p = mod["m1"].predict_proba(F.reshape(-1, F.shape[-1]))[:, 1].reshape(Gy, Gx)
    mk = cv2.morphologyEx((p >= umbral).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mk, connectivity=8); H, W = img.shape[:2]; out = []
    for i in range(1, n):
        gx, gy, gw, gh = st[i, :4]
        x0 = max(0, int(xs[gx]) - pad); y0 = max(0, int(ys[gy]) - pad); x1 = min(W, int(xs[gx + gw - 1]) + pad); y1 = min(H, int(ys[gy + gh - 1]) + pad)
        if (x1 - x0) * (y1 - y0) < area_min: continue
        f = rasgos_region(img, (x0, y0, x1, y1))
        if f is None: continue
        pr = mod["m2"].predict_proba(f[None])[0]; out.append({"region": [x0, y0, x1, y1], "tipo": TIPOS[int(pr.argmax())], "confianza": float(pr.max()), "p_alterado": float(p[lab == i].mean())})
    return out

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1); u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i / u if u else 0

def _una(args):
    x, mod = args
    return x, revisar(cv2.imread(x["ruta"]), mod)

def evaluar(div, n):
    mod = pickle.load(open(MODELO, "rb")); L = cargar(div, n); t0 = time.time()
    tp_loc = fp = fn = 0; tipo_ok = 0; por = {t: [0, 0, 0] for t in TIPOS}; conf = collections.Counter(); sin_ed_con_alarma = sin_ed = 0
    from concurrent.futures import ThreadPoolExecutor
    for x in L:
        pred = revisar(cv2.imread(x["ruta"]), mod); gt = x["regiones"]; usado = set()
        for p in pred:
            m = [(iou(p["region"], g["region"]), j) for j, g in enumerate(gt) if j not in usado]
            if m and max(m)[0] >= 0.5:
                j = max(m)[1]; usado.add(j); tp_loc += 1; ok = p["tipo"] == gt[j]["tipo"]; tipo_ok += ok; conf[(gt[j]["tipo"], p["tipo"])] += 1
                por[gt[j]["tipo"]][0] += 1; por[gt[j]["tipo"]][1] += ok
            else: fp += 1
        fn += len(gt) - len(usado)
        for j, g in enumerate(gt): por[g["tipo"]][2] += 1
        if not gt: sin_ed += 1; sin_ed_con_alarma += bool(pred)
    rec = tp_loc / max(1, tp_loc + fn); pre = tp_loc / max(1, tp_loc + fp)
    res = {"division": div, "imagenes": len(L), "regiones_reales": tp_loc + fn, "localizadas_IoU0.5": tp_loc, "falsas": fp,
           "recall_localizacion": round(rec, 3), "precision_localizacion": round(pre, 3),
           "tipo_correcto_entre_localizadas": round(tipo_ok / max(1, tp_loc), 3),
           "recall_con_tipo_correcto": round(tipo_ok / max(1, tp_loc + fn), 3),
           "por_tipo": {t: {"reales": v[2], "localizadas": v[0], "tipo_ok": v[1], "recall_loc": round(v[0] / max(1, v[2]), 3)} for t, v in por.items()},
           "confusion_real_predicho": {f"{a}->{b}": c for (a, b), c in sorted(conf.items())},
           "imagenes_sin_edicion": sin_ed, "con_alarma": sin_ed_con_alarma, "segundos_por_imagen": round((time.time() - t0) / len(L), 2)}
    print(json.dumps(res, ensure_ascii=False, indent=1)); return res

if __name__ == "__main__":
    c = sys.argv[1]
    if c == "entrenar": entrenar(int(sys.argv[2]) if len(sys.argv) > 2 else 3000)
    elif c == "evaluar": evaluar(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 600)
    elif c == "ver":
        mod = pickle.load(open(MODELO, "rb")); img = cv2.imread(sys.argv[2]); col = {"desenfoque": (0, 200, 255), "pixelado": (0, 255, 0), "tapado": (255, 0, 255)}
        for r in revisar(img, mod):
            a, b, c2, d = r["region"]; cv2.rectangle(img, (a, b), (c2, d), col[r["tipo"]], 3); cv2.putText(img, r["tipo"], (a, max(18, b - 6)), 0, 0.8, col[r["tipo"]], 2); print(r)
        cv2.imwrite("salidas_demo/revision.jpg", img)
