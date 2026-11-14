"""Ediciones que esconden una cara. Una sola fuente de verdad: las usan la app (parte A) y el generador de imágenes
alteradas (parte B). Tipos: "desenfoque", "pixelado", "tapado". Cada función edita la imagen en el lugar y devuelve la región
realmente modificada (x0, y0, x1, y1)."""
import cv2, numpy as np

TIPOS = ("desenfoque", "pixelado", "tapado")

def pixelar(r, b):
    """Mosaico exacto: bloques de b×b píxeles, cada uno pintado con el promedio de color de sus píxeles."""
    h, w = r.shape[:2]; hh, ww = -(-h // b) * b, -(-w // b) * b
    p = cv2.copyMakeBorder(r, 0, hh - h, 0, ww - w, cv2.BORDER_REPLICATE)
    m = p.reshape(hh // b, b, ww // b, b, p.shape[2]).mean(axis=(1, 3))
    return np.repeat(np.repeat(m, b, 0), b, 1)[:h, :w].astype(r.dtype)

def desenfocar(r, fuerza=1.0):
    """Gauss doble sobre la región reducida a 1/4 (mismo aspecto con mucho menos cálculo). fuerza=1 es el valor de la app."""
    h0, w0 = r.shape[:2]; q = cv2.resize(r, (max(1, w0 // 4), max(1, h0 // 4)), interpolation=cv2.INTER_AREA)
    k = max(3, int(w0 * fuerza / 8) | 1)
    q = cv2.GaussianBlur(cv2.GaussianBlur(q, (k, k), 0), (k, k), 0)
    return cv2.resize(q, (w0, h0), interpolation=cv2.INTER_LINEAR)

def region(forma, caja, margen, bloque=None):
    """Caja (x, y, w, h) agrandada por el margen y recortada a la imagen. Si hay bloque, el origen se alinea a la cuadrícula del (0,0)."""
    H, W = forma[:2]; x, y, w, h = caja; mx, my = w * margen, h * margen
    x0, y0 = int(max(0, x - mx)), int(max(0, y - my)); x1, y1 = int(min(W, x + w + mx)), int(min(H, y + h + my))
    if bloque: x0, y0 = x0 // bloque * bloque, y0 // bloque * bloque
    return x0, y0, x1, y1

def editar(img, caja, tipo, margen=0.25, bloque=16, fuerza=1.0, color=(0, 0, 0)):
    b = bloque if tipo == "pixelado" else None
    x0, y0, x1, y1 = region(img.shape, caja, margen, b)
    r = img[y0:y1, x0:x1]
    if r.size == 0: return None
    if tipo == "pixelado": img[y0:y1, x0:x1] = pixelar(r, bloque)
    elif tipo == "desenfoque": img[y0:y1, x0:x1] = desenfocar(r, fuerza)
    elif tipo == "tapado": img[y0:y1, x0:x1] = np.array(color, dtype=img.dtype)
    else: raise ValueError(tipo)
    return x0, y0, x1, y1
