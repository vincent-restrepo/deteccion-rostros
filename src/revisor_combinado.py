"""Revisor combinado (E22): las reglas de píxeles mandan en pixelado y tapado; el bosque v2 (REVISOR_V2=1) pone el resto y lo que las reglas no vieron.
Uso:  REVISOR_V2=1 .venv/bin/python src/revisor_combinado.py <val|test> <N> [desde] [umbral]"""
import sys, os, json, pickle, collections
os.environ["REVISOR_V2"] = "1"
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import revisor, analisis_pixeles
from revisor import iou, TIPOS, MODELO

def revisar(img, mod, umbral=0.95, tipos_reglas=("pixelado", "tapado")):
    reglas = [r for r in analisis_pixeles.analizar(img) if r["tipo"] in tipos_reglas]; out = list(reglas)
    for b in revisor.revisar(img, mod, umbral):
        if all(iou(b["region"], r["region"]) < 0.3 for r in reglas): out.append(b)
    return out

if __name__ == "__main__":
    div, n = sys.argv[1], int(sys.argv[2]); desde = int(sys.argv[3]) if len(sys.argv) > 3 else 0; umb = float(sys.argv[4]) if len(sys.argv) > 4 else 0.95
    mod = pickle.load(open(MODELO, "rb")); L = revisor.cargar(div, n)[desde:]; tp = fp = fn = ok = sin = alarma = 0; por = {t: [0, 0, 0] for t in TIPOS}
    for x in L:
        pred = revisar(cv2.imread(x["ruta"]), mod, umb); gt = x["regiones"]; usado = set()
        for p in pred:
            m = [(iou(p["region"], g["region"]), j) for j, g in enumerate(gt) if j not in usado]
            if m and max(m)[0] >= 0.5:
                j = max(m)[1]; usado.add(j); tp += 1; o = p["tipo"] == gt[j]["tipo"]; ok += o; por[gt[j]["tipo"]][0] += 1; por[gt[j]["tipo"]][1] += o
            else: fp += 1
        fn += len(gt) - len(usado)
        for g in gt: por[g["tipo"]][2] += 1
        if not gt: sin += 1; alarma += bool(pred)
    print(json.dumps({"imagenes": len(L), "reales": tp + fn, "localizadas": tp, "falsas": fp, "recall": round(tp / max(1, tp + fn), 3), "precision": round(tp / max(1, tp + fp), 3), "tipo_ok": round(ok / max(1, tp), 3), "recall_con_tipo": round(ok / max(1, tp + fn), 3), "sin_edicion_con_alarma": f"{alarma}/{sin}", "por_tipo": {t: f"{v[0]}/{v[2]} (tipo ok {v[1]})" for t, v in por.items()}}, ensure_ascii=False, indent=1))
