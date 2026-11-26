"""«Reconocer transformación» (parte B en la app): dada una foto, marca con contorno rojo dónde hubo una alteración y de qué tipo
(desenfoque/difusión, ruido, pixelado, tapado, reducción de colores). Usa el revisor combinado (E22): reglas de píxeles para pixelado y tapado + bosque v2 para el resto.
Modelo: modelos/revisor_v2.pkl (cómo entrenarlo: borradores/guia-reproducir.md). Solo visión clásica.
Independencia (requisito de Vincent): `reconocer(img)` recibe SOLO los píxeles de la foto. No lee el modo, los parámetros ni las cajas de la app ni el detector de caras: nada de lo que se hizo en la edición llega hasta aquí.
Uso suelto:  .venv/bin/python src/reconocer_transformacion.py <foto.jpg> [salida.jpg]"""
import os, sys, pickle
os.environ["REVISOR_V2"] = "1"
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import revisor_combinado, revisor

NOMBRES = {"desenfoque": "Difusión (desenfoque)", "pixelado": "Pixelado", "tapado": "Tapado", "ruido": "Ruido", "colores": "Colores reducidos"}
SIN_TILDE = {"desenfoque": "Difusion (desenfoque)", "pixelado": "Pixelado", "tapado": "Tapado", "ruido": "Ruido", "colores": "Colores reducidos"}   # OpenCV no dibuja tildes
GRUPO = {"tapado": "plano", "colores": "plano"}   # un tapado en óvalo se ve como rectángulos planos sueltos; se unen con los de colores vecinos

HUECO_PLANO = 24   # hueco máximo (px) para unir trozos de tapado/colores

def unir(regiones, hueco=24, hueco_plano=None):
    """Une cajas del mismo grupo (mismo tipo; tapado y colores juntos) que se tocan o están a ≤ hueco píxeles; la unión toma el tipo de la caja más grande."""
    rs = [dict(r) for r in regiones]; cambio = True
    while cambio:
        cambio = False
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                a, b = rs[i], rs[j]
                if GRUPO.get(a["tipo"], a["tipo"]) != GRUPO.get(b["tipo"], b["tipo"]): continue
                A, B = a["region"], b["region"]; hueco_ = (hueco_plano if hueco_plano is not None else HUECO_PLANO) if GRUPO.get(a["tipo"]) else hueco
                if min(A[2], B[2]) + hueco_ < max(A[0], B[0]) or min(A[3], B[3]) + hueco_ < max(A[1], B[1]): continue
                area = lambda r: (r[2] - r[0]) * (r[3] - r[1]); mayor = a if area(A) >= area(B) else b
                rs[i] = dict(mayor, region=[min(A[0], B[0]), min(A[1], B[1]), max(A[2], B[2]), max(A[3], B[3])]); del rs[j]; cambio = True; break
            if cambio: break
    return rs
ROJO = (40, 40, 235)   # BGR
_MODELO = None

def disponible(): return os.path.exists(revisor.MODELO)

def _modelo():
    global _MODELO
    if _MODELO is None: _MODELO = pickle.load(open(revisor.MODELO, "rb"))
    return _MODELO

def reconocer(img, umbral=0.95, juntar=True):
    """Devuelve (imagen con contornos rojos, lista de regiones {region, tipo, nombre, confianza})."""
    regiones = revisor_combinado.revisar(img, _modelo(), umbral)
    if juntar: regiones = unir(regiones)
    for r in regiones: r["nombre"] = NOMBRES.get(r["tipo"], r["tipo"])
    out = img.copy(); grosor = max(2, round(max(img.shape[:2]) / 400)); esc = max(0.5, max(img.shape[:2]) / 1100)
    for r in regiones:
        x0, y0, x1, y1 = (int(v) for v in r["region"]); cv2.rectangle(out, (x0, y0), (x1, y1), ROJO, grosor)
        txt = SIN_TILDE.get(r["tipo"], r["nombre"]); (tw, th), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, esc, max(1, grosor - 1)); ty = y0 - 6 if y0 - th - 10 > 0 else y1 + th + 8
        cv2.rectangle(out, (x0, ty - th - 4), (x0 + tw + 8, ty + base), ROJO, -1)
        cv2.putText(out, txt, (x0 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, esc, (255, 255, 255), max(1, grosor - 1), cv2.LINE_AA)
    return out, regiones

if __name__ == "__main__":
    img = cv2.imread(sys.argv[1]); out, rs = reconocer(img)
    for r in rs: print(r["nombre"], [int(v) for v in r["region"]], round(float(r.get("confianza", 0)), 2))
    print(len(rs), "regiones"); cv2.imwrite(sys.argv[2] if len(sys.argv) > 2 else "/tmp/reconocida.jpg", out)
