"""Revisor por análisis directo de píxeles (parte B, método principal, D6). Sin entrenar ningún modelo: reglas medibles.
  1. «Plano»: píxeles cuyo rango en 3×3 es <= TAU (una región pixelada o tapada es plana por dentro; JPEG deja algo de ruido).
  2. Componentes planos rectangulares = bloques candidatos (se agrandan 1 px: el borde entre bloques no es plano).
  3. Pixelado: bloques cuadrados de lado b (4–64) alineados a la cuadrícula del (0,0) y vecinos entre sí → región = unión; se informa b.
  4. Tapado: un componente plano rectangular grande que no es parte de una cuadrícula.
  5. Desenfoque: zona rectangular con nitidez (log de la energía del Laplaciano) mucho menor que su entorno inmediato.
Uso:  .venv/bin/python src/analisis_pixeles.py evaluar <val|test> [N] | ver <imagen.jpg>"""
import sys, os, json, collections, time
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revisor import cargar, iou

BLOQUES = (4, 6, 8, 12, 16, 24, 32, 48, 64)
TAU = 6            # rango máximo (niveles de gris) en 3×3 para llamar «plano» a un píxel
RELLENO = 0.80     # fracción mínima del rectángulo que debe estar plana
LADO_MIN = 4       # lado mínimo de un bloque (px)
TOL = 2            # tolerancia (px) en tamaño y alineación a la cuadrícula
AREA_TAPADO = 40 * 40
DELTA_BLUR = 1.2   # caída mínima de nitidez frente al entorno (en log de energía)
RELLENO_BLUR = 0.8; LADO_BLUR = 40   # desenfoque: relleno mínimo del rectángulo y lado mínimo (px)
PROP_MIN, PROP_MAX = 0.35, 1.8   # proporción ancho/alto de una región que esconde una cara (en los datos: 0,42–1,4, percentiles 1–99)
MIN_VALIDOS_PEQ = 6      # pixelado con bloque < 24: mínimo de bloques cuadrados alineados (una región ≥ 48 px tiene muchos)
MIN_VALIDOS_GRANDE = 1   # bloque ≥ 24: caben pocos bloques en una cara
CONTRASTE_TAPADO = 20    # diferencia mínima de brillo entre el tapado y sus 6 px de alrededor
# estos tres umbrales se ajustaron en las primeras 150 imágenes de validación y se comprobaron en otras 150 (E10)
PARAMS = {"PROP_MIN": PROP_MIN, "PROP_MAX": PROP_MAX, "MIN_VALIDOS_PEQ": MIN_VALIDOS_PEQ, "MIN_VALIDOS_GRANDE": MIN_VALIDOS_GRANDE, "CONTRASTE_TAPADO": CONTRASTE_TAPADO, "RELLENO_BLUR": RELLENO_BLUR, "LADO_BLUR": LADO_BLUR, "TAU": TAU, "RELLENO": RELLENO, "LADO_MIN": LADO_MIN, "TOL": TOL, "AREA_TAPADO": AREA_TAPADO, "DELTA_BLUR": DELTA_BLUR}

def _bloques_planos(g, tau=None):
    tau = TAU if tau is None else tau
    k = np.ones((3, 3), np.uint8)
    plano = ((cv2.dilate(g, k) - cv2.erode(g, k)) <= tau).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(plano, connectivity=4); H, W = g.shape; out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w < LADO_MIN or h < LADO_MIN or a < RELLENO * w * h: continue
        x0, y0, x1, y1 = max(0, x - 1), max(0, y - 1), min(W, x + w + 1), min(H, y + h + 1)   # el borde entre bloques no es plano
        out.append({"caja": (x0, y0, x1, y1), "color": float(g[y:y + h, x:x + w].mean())})
    return out

def _cuadricula(c):
    """Lado b (de BLOQUES) si la caja es un bloque cuadrado alineado a la cuadrícula del (0,0); si no, None."""
    x0, y0, x1, y1 = c["caja"]; w, h = x1 - x0, y1 - y0
    for b in BLOQUES:
        if abs(w - b) <= TOL and abs(h - b) <= TOL and min(x0 % b, b - x0 % b) <= TOL and min(y0 % b, b - y0 % b) <= TOL: return b
    return None

def _parcial(c, b):
    """Componente que encaja en la cuadrícula de lado b: origen alineado y lados ≤ b o múltiplos de b."""
    x0, y0, x1, y1 = c["caja"]
    lado = lambda l: l <= b + TOL or min(l % b, b - l % b) <= TOL   # recortado (≤ b) o varios bloques del mismo color fundidos (múltiplo de b)
    return lado(x1 - x0) and lado(y1 - y0) and min(x0 % b, b - x0 % b) <= TOL and min(y0 % b, b - y0 % b) <= TOL

def _prop_ok(r):
    w, h = r[2] - r[0], r[3] - r[1]
    return h > 0 and PROP_MIN <= w / h <= PROP_MAX

def _cerca(a, b, d=3):
    return not (a[2] + d < b[0] or b[2] + d < a[0] or a[3] + d < b[1] or b[3] + d < a[1])

def analizar(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); H, W = g.shape
    bl = _bloques_planos(g); sal = []; usado = set()
    for c in bl: c["b"] = _cuadricula(c)
    valido = [i for i, c in enumerate(bl) if c["b"]]
    visto = set()
    for s in valido:   # grupos de bloques vecinos que parten de un bloque válido
        if s in visto: continue
        grupo = [s]; visto.add(s); cola = [s]; bg = bl[s]["b"]
        while cola:
            i = cola.pop()
            for j, c in enumerate(bl):
                if j in visto or not _cerca(bl[i]["caja"], c["caja"]): continue
                if c["b"] == bg or _parcial(c, bg):
                    visto.add(j); grupo.append(j); cola.append(j)
        bs = collections.Counter(bl[i]["b"] for i in grupo if bl[i]["b"]); b = bs.most_common(1)[0][0]
        validos_b = sum(1 for i in grupo if bl[i]["b"] == b)
        colores = len({round(bl[i]["color"] / 4) for i in grupo})
        xs = [bl[i]["caja"] for i in grupo]
        if len(grupo) >= 2 and colores >= 2 and validos_b >= (MIN_VALIDOS_GRANDE if b >= 24 else MIN_VALIDOS_PEQ) and _prop_ok((min(c[0] for c in xs), min(c[1] for c in xs), max(c[2] for c in xs), max(c[3] for c in xs))):
            sal.append({"region": [min(c[0] for c in xs), min(c[1] for c in xs), max(c[2] for c in xs), max(c[3] for c in xs)], "tipo": "pixelado", "bloque": b, "bloques": len(grupo), "validos": validos_b, "colores": colores,
                        "relleno": round(sum((bl[i]["caja"][2]-bl[i]["caja"][0])*(bl[i]["caja"][3]-bl[i]["caja"][1]) for i in grupo) / max(1, (max(c[2] for c in xs)-min(c[0] for c in xs))*(max(c[3] for c in xs)-min(c[1] for c in xs))), 3)})
            usado |= set(grupo)
    for i, c in enumerate(bl):   # tapado: rectángulo plano grande que no forma parte de una cuadrícula
        x0, y0, x1, y1 = c["caja"]
        if i in usado or (x1 - x0) * (y1 - y0) < AREA_TAPADO or not _prop_ok(c["caja"]): continue
        ring = g[max(0, y0 - 8):y1 + 8, max(0, x0 - 8):x1 + 8].astype(np.float32).copy(); ring[8 if y0 >= 8 else y0:, :] = ring[8 if y0 >= 8 else y0:, :]
        dentro = float(g[y0:y1, x0:x1].mean()); borde = np.concatenate([g[max(0, y0 - 6):y0, x0:x1].ravel(), g[y1:y1 + 6, x0:x1].ravel(), g[y0:y1, max(0, x0 - 6):x0].ravel(), g[y0:y1, x1:x1 + 6].ravel()]).astype(np.float32)
        if borde.size == 0 or abs(dentro - float(borde.mean())) < CONTRASTE_TAPADO or x0 <= 0 or y0 <= 0 or x1 >= W or y1 >= H: continue   # plano que toca el borde de la imagen o sin contraste: fondo, no tapado
        sal.append({"region": [x0, y0, x1, y1], "tipo": "tapado", "area": (x1 - x0) * (y1 - y0), "contraste": round(abs(dentro - float(borde.mean())) if borde.size else 0.0, 1),
                    "desv_borde": round(float(borde.std()) if borde.size else 0.0, 1), "toca_borde": int(x0 <= 0 or y0 <= 0 or x1 >= W or y1 >= H)})
    # desenfoque: caída de nitidez frente al entorno
    gf = g.astype(np.float32); lap = cv2.Laplacian(gf, cv2.CV_32F, ksize=1)
    S = np.log1p(cv2.boxFilter(lap * lap, -1, (15, 15), normalize=True, borderType=cv2.BORDER_REFLECT))
    Sg = cv2.GaussianBlur(S, (0, 0), 24)   # nitidez del entorno (suavizada a gran escala)
    mask = ((Sg - S) >= DELTA_BLUR).astype(np.uint8)
    for r in sal: x0, y0, x1, y1 = r["region"]; mask[y0:y1, x0:x1] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)); mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w < LADO_BLUR or h < LADO_BLUR or a < RELLENO_BLUR * w * h or not _prop_ok((x, y, x + w, y + h)): continue
        sal.append({"region": [int(x), int(y), int(x + w), int(y + h)], "tipo": "desenfoque"})
    return sal

def evaluar(div, n, mostrar=True):
    L = cargar(div, n); t0 = time.time(); tp = fp = fn = ok = 0; por = {t: [0, 0, 0] for t in ("desenfoque", "pixelado", "tapado")}
    conf = collections.Counter(); fp_tipo = collections.Counter(); sin_ed = alarma = 0; err_b = []
    for x in L:
        pred = analizar(cv2.imread(x["ruta"])); gt = x["regiones"]; usado = set()
        for p in pred:
            m = [(iou(p["region"], g["region"]), j) for j, g in enumerate(gt) if j not in usado]
            if m and max(m)[0] >= 0.5:
                j = max(m)[1]; usado.add(j); tp += 1; acierto = p["tipo"] == gt[j]["tipo"]; ok += acierto; conf[(gt[j]["tipo"], p["tipo"])] += 1
                por[gt[j]["tipo"]][0] += 1; por[gt[j]["tipo"]][1] += acierto
                if gt[j]["tipo"] == "pixelado" and p["tipo"] == "pixelado": err_b.append(p["bloque"] == gt[j]["params"]["bloque"])
            else: fp += 1; fp_tipo[p["tipo"]] += 1
        fn += len(gt) - len(usado)
        for g in gt: por[g["tipo"]][2] += 1
        if not gt: sin_ed += 1; alarma += bool(pred)
    res = {"metodo": "analisis_pixeles", "parametros": PARAMS, "division": div, "imagenes": len(L), "regiones_reales": tp + fn, "localizadas_IoU0.5": tp, "falsas": fp,
           "falsas_por_tipo": dict(fp_tipo), "recall_localizacion": round(tp / max(1, tp + fn), 3), "precision_localizacion": round(tp / max(1, tp + fp), 3),
           "tipo_correcto_entre_localizadas": round(ok / max(1, tp), 3), "recall_con_tipo_correcto": round(ok / max(1, tp + fn), 3),
           "bloque_correcto_en_pixelados_bien_tipados": round(sum(err_b) / max(1, len(err_b)), 3),
           "por_tipo": {t: {"reales": v[2], "localizadas": v[0], "tipo_ok": v[1], "recall_loc": round(v[0] / max(1, v[2]), 3)} for t, v in por.items()},
           "confusion_real_predicho": {f"{a}->{b}": c for (a, b), c in sorted(conf.items())},
           "imagenes_sin_edicion": sin_ed, "con_alarma": alarma, "segundos_por_imagen": round((time.time() - t0) / len(L), 2)}
    if mostrar: print(json.dumps(res, ensure_ascii=False, indent=1))
    return res

if __name__ == "__main__":
    c = sys.argv[1]
    if c == "evaluar": evaluar(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 300)
    elif c == "ver":
        img = cv2.imread(sys.argv[2]); col = {"desenfoque": (0, 200, 255), "pixelado": (0, 255, 0), "tapado": (255, 0, 255)}
        for r in analizar(img):
            a, b, c2, d = r["region"]; cv2.rectangle(img, (a, b), (c2, d), col[r["tipo"]], 3)
            cv2.putText(img, r["tipo"] + (f" {r['bloque']}x{r['bloque']}" if "bloque" in r else ""), (a, max(18, b - 6)), 0, 0.8, col[r["tipo"]], 2); print(r)
        cv2.imwrite("salidas_demo/revision-pixeles.jpg", img)
