"""Detector de caras clásico (sin redes): varios Haar frontales (alt2 y alt) sobre la imagen girada en varios ángulos
(cabeza inclinada) más Haar de perfil (también espejado). Cajas (x, y, w, h) en coordenadas de la imagen original.
Recall alto a propósito (D2): mejor difuminar de más que dejar una cara sin cubrir.
Sin ecualizar el histograma: en la prueba con una cabeza inclinada, ecualizar hacía perder la cara (E3).
Para ir rápido: cada cuadro solo se prueban 0° y los ángulos donde hubo cara hace poco; cada BARRIDO cuadros se prueban todos."""
import cv2, numpy as np

PASO = 15
TODOS = tuple(range(-60, 61, PASO))     # -60, -45, ..., 60
BARRIDO = 5
_cascadas = [cv2.CascadeClassifier(cv2.data.haarcascades + n) for n in
             ("haarcascade_frontalface_alt2.xml", "haarcascade_frontalface_alt.xml")]
_perfil = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

def _girar(gris, ang):
    h, w = gris.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1]); W, H = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += W / 2 - w / 2; M[1, 2] += H / 2 - h / 2
    return cv2.warpAffine(gris, M, (W, H), borderValue=int(gris.mean())), M

def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    i = max(0, x2-x1) * max(0, y2-y1); u = a[2]*a[3] + b[2]*b[3] - i
    return i / u if u else 0

def _nms(items, umbral=0.3):
    items = sorted(items, key=lambda it: -it[0][2] * it[0][3]); fin = []
    for it in items:
        if all(_iou(it[0], f[0]) < umbral for f in fin): fin.append(it)
    return fin

def _en_angulo(gris, ang, minimo, vecinos):
    img, M = (gris, None) if ang == 0 else _girar(gris, ang)
    res = []
    for c in _cascadas:
        for (x, y, w, h) in c.detectMultiScale(img, 1.1, vecinos, minSize=(minimo, minimo)):
            cx, cy = x + w / 2, y + h / 2
            if M is not None:
                cx, cy = cv2.invertAffineTransform(M) @ np.array([cx, cy, 1.0])
            k = 1.0 if ang == 0 else 1.2          # la caja de una cara inclinada se agranda para cubrirla
            res.append(((int(cx - w * k / 2), int(cy - h * k / 2), int(w * k), int(h * k)), ang))
    return res

class Detector:
    def __init__(self, minimo=24, vecinos=3, perfil=True, giros=True, frontal=True):
        self.minimo, self.vecinos, self.perfil, self.giros, self.frontal = minimo, vecinos, perfil, giros, frontal
        self.n = 0; self.activos = set()

    def __call__(self, gris):
        self.n += 1
        if not self.giros: angs = {0}
        elif self.n % BARRIDO == 1: angs = set(TODOS)
        else: angs = {0} | {a + d for a in self.activos for d in (-PASO, 0, PASO) if abs(a + d) <= 60}
        out = []
        if self.frontal:
            for a in sorted(angs): out += _en_angulo(gris, a, self.minimo, self.vecinos)
        if self.perfil:
            for esp in (False, True):
                img = cv2.flip(gris, 1) if esp else gris
                for (x, y, w, h) in _perfil.detectMultiScale(img, 1.1, 4, minSize=(self.minimo, self.minimo)):
                    if esp: x = gris.shape[1] - x - w
                    out.append(((int(x), int(y), int(w), int(h)), 0))
        fin = _nms(out)
        if self.n % BARRIDO == 1 or fin: self.activos = {a for _, a in fin if a != 0}
        return [c for c, _ in fin]
